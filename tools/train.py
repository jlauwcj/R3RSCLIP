import argparse
import torch
import clip
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import ssl
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import cv2
import os
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from config.configs import cfg_from_file
from model.model import ReCLIP
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

def setup_optimizer_with_multiscale(model, cfg):

    multiscale_params = []
    fasb_params = []
    other_params = []
    special_param_ids = set()

    base_model = model.module if hasattr(model, 'module') else model

    # 1. 收集多尺度融合模块参数
    if hasattr(base_model, 'vit') and hasattr(base_model.vit, 'multiscale_fusion'):
        fusion_module = base_model.vit.multiscale_fusion
        print(f"[INFO] Detected MultiScaleFusion module.")
        for param in fusion_module.parameters():
            if param.requires_grad:
                multiscale_params.append(param)
                special_param_ids.add(id(param))

    # 2. 收集 FASB 模块参数
    if hasattr(base_model, 'fasb_module') and cfg.MODEL.get('USE_FASB', False):
        print(f"[INFO] Detected FASB module. Setting up training for Frequency Bias.")
        for param in base_model.fasb_module.parameters():
            if param.requires_grad:
                fasb_params.append(param)
                special_param_ids.add(id(param))

    # 3. 收集其他参数 (如 text_encoder 的提示词等)
    for param in model.parameters():
        if param.requires_grad and id(param) not in special_param_ids:
            other_params.append(param)

    param_groups = []
    if len(other_params) > 0:
        param_groups.append({'params': other_params, 'name': 'base', 'lr': cfg.TRAIN.LR})

    if len(multiscale_params) > 0:
        lr_mult = cfg.MODEL.get('MULTISCALE_LR_MULTIPLIER', 0.1)
        param_groups.append({'params': multiscale_params, 'name': 'multiscale', 'lr': cfg.TRAIN.LR * lr_mult})
        print(f"[INFO] Multiscale params: {len(multiscale_params)}, LR Multiplier: {lr_mult}")

    if len(fasb_params) > 0:
        fasb_lr_mult = cfg.MODEL.get('FASB_LR_MULTIPLIER', 1.0)
        param_groups.append({'params': fasb_params, 'name': 'fasb', 'lr': cfg.TRAIN.LR * fasb_lr_mult})
        print(f"[INFO] FASB params: {len(fasb_params)}, LR Multiplier: {fasb_lr_mult}")

    optimizer = optim.SGD(param_groups, lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)
    return optimizer

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power, cfg):
    current_base_lr = base_lr * (1 - epoch / num_epochs) ** power

    lr_multiplier = cfg.MODEL.get('MULTISCALE_LR_MULTIPLIER', 1.0)
    fasb_lr_mult = cfg.MODEL.get('FASB_LR_MULTIPLIER', 1.0)

    for param_group in optimizer.param_groups:
        if param_group.get('name') == 'multiscale':
            param_group['lr'] = current_base_lr * lr_multiplier
        elif param_group.get('name') == 'fasb':
            param_group['lr'] = current_base_lr * fasb_lr_mult
        else:
            param_group['lr'] = current_base_lr

    return current_base_lr


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/Isaid_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='ReCLIP', type=str)
    args = parser.parse_args()
    return args


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        _, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(cfg)
        print(f"Dataset: {len(self.train_images)} images, {len(self.pseudo_classes)} pseudo labels.")
        assert len(self.train_images) == len(self.pseudo_classes)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def train(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(rank)

    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)

    save_dir = cfg.SAVE_DIR
    # 日志记录
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        log = open(os.path.join(save_dir, 'train_log.txt'), mode='a', buffering=1)
    else:
        log = None

    _, val_filenames, _, _, val_images, val_labels, results_iou, _ = read_file_list(cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    train_data = Train(cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(dataset=train_data, shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=True, batch_size=cfg.TRAIN.BATCH_SIZE,
                              collate_fn=custom_collate_fn, sampler=train_sampler)

    # ==================== 3. 模型初始化 (支持多尺度) ====================
    if args.model_name == 'RECLIPPP':
        # ReCLIP++ 暂不支持此代码中的多尺度配置
        model = ReCLIP(cfg=cfg, clip_model=clip_model, zeroshot_weights=text_weight, rank=rank)
    else:
        model = ReCLIP(
            cfg=cfg,
            clip_model=clip_model,
            zeroshot_weights=text_weight,
            rank=rank,
            use_multiscale=cfg.MODEL.get('USE_MULTISCALE', False),
            compress_dim=cfg.MODEL.get('MULTISCALE_COMPRESS_DIM', 256),
            max_scale=cfg.MODEL.get('MAX_SCALE', 0.5)
        )

    model = model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],
                                                      output_device=rank, find_unused_parameters=True)

    # ==================== 4. 优化器初始化 (使用多尺度版本) ====================
    optimizer = setup_optimizer_with_multiscale(model, cfg)

    max_epoch = cfg.TRAIN.MAX_EPOCH
    stop_epoch = cfg.TRAIN.EPOCH if cfg.TRAIN.EPOCH >= 0 else max_epoch
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0
    best_f1 = 0.0
    start_epoch = 0

    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_iou = ckpt.get('best_iou', 0.0)
        best_f1 = ckpt.get('best_f1', 0.0)
        if rank == 0:
            print(f"[INFO] Resumed from checkpoint: epoch={ckpt['epoch']}, best_iou={best_iou:.4f}")
            print(f"[INFO] Continuing from epoch {start_epoch}...", flush=True)
    else:
        if rank == 0:
            print("[INFO] No checkpoint found, training from scratch.")

    for epoch in range(start_epoch, max_epoch):
        idx = 0
        model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, 0.9, cfg)

        loop = tqdm(train_loader)
        for img, label, img_metas, filenames, pseudo_class in loop:
            time.sleep(0.01)
            gt_cls = []
            for i in range(img.shape[0]):
                temp = [int(t) for t in pseudo_class[i]]
                gt_cls.append(temp)

            if not gt_cls or not gt_cls[0]:
                continue

            output, loss = model(img.to(rank), gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(epoch=epoch, img_loss=loss.item(), avg_loss=running_loss / (idx + 1))
            idx += 1

            if log is not None:
                print('Epoch:[{}] idx:[{}] filenames:{} img_loss:{:.5f} avg_loss:{:.5f} lr:{:.7f}'.format(
                    epoch, idx, filenames, loss.item(), running_loss / idx, lr),
                    file=log, flush=True)


        if rank == 0:
            model.eval()
            success_num = 0

            with torch.no_grad():
                for v_idx in range(len(val_images)):
                    with open(val_images[v_idx], 'rb') as f:
                        img_buf = f.read()
                    img_val = val_preprocess(cfg, img_buf).unsqueeze(dim=0).to(rank)

                    label_ori = Image.open(val_labels[v_idx])
                    ori_shape = (label_ori.size[1], label_ori.size[0])

                    gt_cls_val = list(set(np.asarray(label_ori).flatten()) - {255})

                    output = model(img_val, gt_cls_val, text_weight, cls_name_token, training=False)
                    output = F.interpolate(output, ori_shape, mode='bilinear', align_corners=False)
                    output = torch.argmax(output, dim=1).squeeze(0).cpu()

                    torch.save(output, os.path.join(save_dir, val_filenames[v_idx] + '.pt'))
                    success_num += 1

                iou = mean_iou(results_iou, val_labels, num_classes=c_num, ignore_index=255,
                               nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
                avg_iou = iou['IoU'].mean()
                avg_f1 = iou['F1'].mean()

                if log is not None:
                    print('--- Validation Epoch {} --- mIoU: {:.4f} --- mF1: {:.4f} ---'.format(epoch, avg_iou, avg_f1), file=log, flush=True)

                print(f'Epoch {epoch}| mIoU: {avg_iou:.4f}| mF1: {avg_f1:.4f}')
                if avg_iou > best_iou:
                    best_iou = avg_iou
                    torch.save(model.state_dict(), os.path.join(save_dir, 'best_weight.pth'))
                    print(f"[INFO] Best model updated at epoch {epoch}, mIoU={best_iou:.4f}")

                if avg_f1 > best_f1:
                    best_f1 = avg_f1

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'best_f1': best_f1,
                }, checkpoint_path)
                print(f"[INFO] Checkpoint saved: epoch={epoch}", flush=True)


        if epoch == stop_epoch: break
    if log is not None:
        log.close()

if __name__ == '__main__':
    world_size = 1
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
