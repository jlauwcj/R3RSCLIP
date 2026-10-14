import torch

import clip
import argparse
from PIL import Image
from tqdm import tqdm
import time
import json

import sys
import os
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 Python 搜索路径
sys.path.insert(0, project_root)


from utils.preprocess import read_file_list, prepare_dataset_cls_tokens, preprocess, val_preprocess
from config.configs import cfg_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loveda_classes = [ "background","building","road","water","barren","forest","agricultural"]
potsdam_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
vaihingen_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
isaid_classes = ["background", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
                 "ground track field", "bridge", "large vehicle","small vehicle", "helicopter", "swimming pool",
                 "roundabout","soccer ball field", "plane", "harbor"]
uavid_classes = ["background", "building", "road", "tree", "low vegetation", "moving car", "static car", "human"]
udd_classes = ["vegetation", "building", "road", "vehicle", "other"]

def print_learnable_params_with_name(model, prefix=''):
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f'{prefix}{name}, Shape: {param.shape}')
    for name, module in model.named_children():
        print_learnable_params_with_name(module, prefix=f'{name}.' if name else '')


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/vaihingen_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='ReCLIP', type=str)
    args = parser.parse_args()
    return args


def ReCLIP(images, labels, cfg):
    clip_model, new_clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    total_accuracy = 0
    total_recall = 0
    idx = 0
    loop = tqdm(zip(images, labels))
    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)
    print(len(images))
    for image, label in loop:
        time.sleep(0.08)
        with open(image, 'rb') as f:
            value_buf = f.read()
        with open(label, 'rb') as f:
            label_buf = f.read()
        _, label, img_metas = preprocess(cfg, value_buf, label_buf, return_meta=True, unlabeled=False)

        idx += 1
        img = Image.open(image)

        img = new_clip_preprocess(img).unsqueeze(dim=0).to(device)
        image_features = clip_model.encode_image(img)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        temp = []
        for j in range(cfg.DATASET.NUM_CLASSES):
            if logits_per_image[0][j] >= cfg.DATASET.THRESHOLD:
                temp.append(j)
        if len(temp) == 0:
            _, pred_label = torch.sort(logits_per_image, descending=True, dim=1)
            temp = pred_label[:, :cfg.DATASET.K].squeeze(dim=0).tolist()
#---------------------数据集text
        with open('text/vaihingen_pseudo_label_ReCLIP.json', mode='a') as cls_json:
            cls_json.write(json.dumps(temp))
            cls_json.write('\n')

#label_cls = set(label.flatten().tolist()[1:])
        label_flat = label.flatten().tolist()
        label_cls = set([x for x in label_flat if x != 255])
        same_values = set(label_cls) & set(temp)
        acc = len(same_values) / max(len(temp),1)#除0风险
        recall = 0.0 if len(label_cls) == 0 else len(same_values) / len(label_cls)

        total_accuracy += acc
        total_recall += recall
        loop.set_postfix(idx=idx, acc=acc, recall=recall, avg_acc=total_accuracy / idx, avg_recall=total_recall / idx)
    total_accuracy = total_accuracy / len(images)
    total_recall = total_recall / len(images)
    print('total acc = {}, total_recall = {}'.format(total_accuracy, total_recall))


def ReCLIPPP(images, labels, cfg, window_size, step_size):
    clip_model, new_clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    _, text = prepare_dataset_cls_tokens(cfg)

    total_accuracy = 0
    total_recall = 0
    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)

    loop = tqdm(zip(images, labels))
    idx = 0
    print(len(images))
    for image, label in loop:
        time.sleep(0.08)
        idx += 1
        with open(image, 'rb') as f:
            value_buf = f.read()
        with open(label, 'rb') as f:
            label_buf = f.read()
        _, label, img_metas = preprocess(cfg, value_buf, label_buf, return_meta=True, unlabeled=False)

        img = Image.open(image)
        width, height = img.size
        cls_dict = {}
        temp = []
        window_size = list(map(int, window_size))
        step_size = int(step_size)
        for y in range(0, height - window_size[1], step_size):
            for x in range(0, width - window_size[0], step_size):
                box = (x, y, x + window_size[0], y + window_size[1])

                cropped_img = img.crop(box)
                cropped_img = new_clip_preprocess(cropped_img).unsqueeze(dim=0).to(device)
                image_features = clip_model.encode_image(cropped_img)
                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp()
                logits_per_crop = logit_scale * image_features @ text_features.t()

                sort_logits, pred_label = torch.sort(logits_per_crop, descending=True, dim=1)
                cls = pred_label[0, 0].item()
                cls_dict[cls] = cls_dict.get(cls, 0) + 1
        for key in cls_dict.keys():
            if cls_dict[key] > 0:
                temp.append(key)
        if len(temp) == 0:
            sorted(cls_dict.items(), key=lambda x: x[1])
            keys = cls_dict.keys()
            temp = list(keys)[:cfg.DATASET.K]
        with open('text/vaihingen_pseudo_label_ReCLIPPP.json', mode='a') as cls_json:
            cls_json.write(json.dumps(temp))
            cls_json.write('\n')
#label_cls = set(label.flatten().tolist()[1:])
        label_flat = label.flatten().tolist()
        label_cls = set([x for x in label_flat if x != 255])
        same_values = set(label_cls) & set(temp)
        acc = len(same_values) / max(len(temp), 1)  # 防除 0
        recall = 0.0 if len(label_cls) == 0 else len(same_values) / len(label_cls)

        total_accuracy += acc
        total_recall += recall
        loop.set_postfix(idx=idx, acc=acc, recall=recall, avg_acc=total_accuracy / idx, avg_recall=total_recall / idx)
    total_accuracy = total_accuracy / len(images)
    total_recall = total_recall / len(images)
    print('total acc = {}, total recall = {}'.format(total_accuracy, total_recall))


if __name__ == '__main__':
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    crop_size = cfg.DATASET.CROP_SIZE
    w = crop_size[0] / 6
    s = w / 2

    _, _, train_images, train_labels, _, _, _, pseudo_classes = read_file_list(cfg)
    if args.model_name == 'RECLIPPP':
        ReCLIPPP(train_images, train_labels, cfg, window_size=(w, w), step_size=s)
    else:
        ReCLIP(train_images, train_labels, cfg)
