import argparse
import torch
import clip
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time
from thop import profile
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from config.configs import cfg_from_file
from model.model import ReCLIP
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, read_file_list, prepare_dataset_cls_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_preprocess = clip.load("ViT-B/16")
clip_model = clip_model.to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/Potsdam_test_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='ReCLIP', type=str)
    args = parser.parse_args()
    return args

def test():
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)

    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(
        cfg)
    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)
    if args.model_name == 'RECLIPPP':
        model = RECLIPPP(cfg=cfg, clip_model=clip_model, rank=0)  # , zeroshot_weights=text_weight)
    else:
        model = ReCLIP(cfg=cfg, clip_model=clip_model, rank=0, zeroshot_weights=text_weight,
                       use_multiscale=cfg.MODEL.get('USE_MULTISCALE', False),
                       compress_dim=cfg.MODEL.get('MULTISCALE_COMPRESS_DIM', 256),
                       max_scale=cfg.MODEL.get('MAX_SCALE', 0.5))
    checkpoint = torch.load(cfg.LOAD_PATH, map_location=device)
    new_weight = {}

    for key, value in checkpoint.items():
        new_key = key[7:] if key.startswith('module.') else key

        if 'proj_net' in new_key:
            print(f"[SKIP] Skipping old weight key: {new_key}")
            continue
        new_weight[new_key] = value

    msg = model.load_state_dict(new_weight, strict=False)
    print(f"[INFO] Load Weight Status: {msg}")
    model = model.to(device)

    c_num = cfg.DATASET.NUM_CLASSES
    model.eval()

    print("\n" + "=" * 20 + " 模型效率评估 " + "=" * 20)
    if hasattr(cfg.DATASET, 'CROP_SIZE'):
        H, W = cfg.DATASET.CROP_SIZE
    else:
        H, W = 224, 224
    dummy_input = torch.randn(1, 3, H, W).to(device)

    # 计算参数量和 FLOPs
    flops, params = profile(model.vit, inputs=(dummy_input,), verbose=False)
    print(f"测试分辨率: {H} x {W}")
    print(f"总参数量 (Params): {params / 1e6:.2f} M")
    print(f"计算复杂度 (FLOPs): {flops / 1e9:.2f} G")
    # 计算推理速度 (FPS)
    iterations = 50
    for _ in range(10):
        _ = model.vit(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model.vit(dummy_input)
    torch.cuda.synchronize()
    fps = iterations / (time.time() - start_time)
    print(f"推理速度 (FPS): {fps:.2f} frames/s")
    print("=" * 50 + "\n")
    # ==============================================================================

    with torch.no_grad():
        idx = 0
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)

            label = Image.open(val_labels[idx])
            ori_shape = tuple((label.size[1], label.size[0]))
            label = np.asarray(label).copy()

            label[label == 0] = 255

            gt_cls = []
            shape = img.shape[2:]
            output = model(img, gt_cls, text_weight, cls_name_token, training=False)

            # pd
            N, C, H, W = output.shape
            if args.model_name == 'RECLIPPP':
                _output = F.softmax(output * 10, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100
            else:
                _output = F.softmax(output * 100, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < cfg.TEST.ReCLIP_PD)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100

            output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
            output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0],
                                                                                       ori_shape[1])

            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1).squeeze(dim=0)
            torch.save(output, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
            # print('filenames:{}, img_idx:{}'.format(val_filenames[idx], idx))

        iou = mean_iou(results_iou, val_labels, num_classes=c_num, ignore_index=255, nan_to_num=0,
                       reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
        print("\n各类别 IoU:", iou['IoU'])
        print("各类别 F1-Score:", iou['F1'])
        avg_iou = iou['IoU'].sum() / c_num
        avg_f1 = iou['F1'].mean()
        print('mIoU:%.4f' % (avg_iou))
        print(f'mF1:  {avg_f1:.4f}')
        print('\n\nfinish with %d/%d\nthe mIOU:%.4lf, the mF1:%.4lf' % (idx + 1, len(val_images), avg_iou, avg_f1))


if __name__ == '__main__':
    test()
