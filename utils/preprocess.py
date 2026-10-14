import numpy as np
import cv2
from cv2 import IMREAD_COLOR
import torch
import os
import clip
from PIL import Image
import io
from numpy import random
import json

RS_ENHANCE_MAPPING = {
    "road": "aerial asphalt road",
    "water": "satellite water body",
    "barren": "barren soil land",
    "forest": "dense forest canopy",
    "agricultural": "farmland crop field",

    # --- Potsdam / Vaihingen ---
    "background": "aerial landscape land",
    "impervious surface": "paved concrete surface",
    "building": "building roof aerial",
    "low vegetation": "green grass shrubs",
    "tree": "aerial tree canopy",
    "car": "overhead small car",

    # --- UAVid (侧重低空视角) ---
    "moving car": "moving vehicle road",
    "static car": "parked vehicle lot",
    "human": "aerial person pedestrian"
}
'''
# udd5适配
RS_ENHANCE_MAPPING = {
    "building": "urban building",
    "road": "paved road",
    "vegetation": "greenery",
    "other": "background",
    "vehicle": "car",
    "tree": "tree",
}

# uavid适配
RS_ENHANCE_MAPPING = {
    "building": "building",
    "road": "paved road",
    "low vegetation": "green grass",
    "moving car": "moving car",
    "static car": "static car",
    "tree": "green tree",
    "human": "human",
    "background": "background"
}
'''
prompt_templates = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.',
    'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.',
    'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.',
    'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.',
    'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.',
    'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.',
    'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.',
    'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.',
    'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.',
    'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.',
    'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.',
    'this is the {} in the scene.', 'this is one {} in the scene.']
isaid_classes = ["background", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
                 "ground track field", "bridge", "large vehicle","small vehicle", "helicopter", "swimming pool",
                 "roundabout","soccer ball field", "plane", "harbor"]
loveda_classes = [ "background","building","road","water","barren","forest","agricultural"]

potsdam_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
vaihingen_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
uavid_classes = ["background", "building", "road", "tree", "low vegetation", "moving car", "static car", "human"]
udd_classes = ["vegetation", "building", "road", "vehicle", "other"]

def get_crop_bbox(cfg, img):
    crop_size = cfg.DATASET.CROP_SIZE

    margin_h = max(img.shape[0] - crop_size[0], 0)
    margin_w = max(img.shape[1] - crop_size[1], 0)
    offset_h = np.random.randint(0, margin_h + 1)
    offset_w = np.random.randint(0, margin_w + 1)
    crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
    crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
    return crop_y1, crop_y2, crop_x1, crop_x2


def crop(img, crop_bbox):
    """Crop from ``img``"""
    crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    return img


def preprocess(cfg, image, label, return_meta=False, unlabeled=True):
    img_metas = {}
    ratio_range = cfg.DATASET.RATIO_RANGE
    img_scale = cfg.DATASET.SCALE
    cat_max_ratio = cfg.DATASET.CAT_MAX_RATIO
    crop_size = cfg.DATASET.CROP_SIZE
    brightness_delta = 32
    contrast_range = (0.5, 1.5)
    contrast_lower, contrast_upper = contrast_range
    saturation_range = (0.5, 1.5)
    saturation_lower, saturation_upper = saturation_range
    hue_delta = 18

    img_metas['crop_size'] = crop_size
    img_np = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(img_np, IMREAD_COLOR)

    if label is not None:
        with io.BytesIO(label) as buff:
            label = Image.open(buff)
            label = np.array(label)
        label = label.astype(np.int64)
#---------------loveda数据集特定------------------
        if cfg.DATASET.NAME == 'loveda':
            label[label == 7] = 255
        else:
#------------------------------------------------
            if cfg.DATASET.REDUCE_ZERO_LABEL:
                label[label == 0] = 255
                label = label - 1
                label[label == 254] = 255

    if unlabeled:
        for i in range(cfg.DATASET.NUM_CLASSES):
            label[label == i] = -1
    # Resize
    # image
    h, w = img.shape[:2]
    min_ratio, max_ratio = ratio_range
    ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
    scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)

    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))
    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    resized_img = cv2.resize(img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)
    # label
    resized_label = cv2.resize(label, new_size, dst=None, interpolation=cv2.INTER_NEAREST)
    img_metas['new_size'] = new_size

    # RandomCrop
    crop_bbox = get_crop_bbox(cfg, resized_img)
    for _ in range(10):
        seg_temp = crop(resized_label, crop_bbox)
        labels, cnt = np.unique(seg_temp, return_counts=True)
        cnt = cnt[labels != cfg.DATASET.IGNORE_INDEX]
        if len(cnt) > 1 and np.max(cnt) / np.sum(
                cnt) < cat_max_ratio:
            break
        crop_bbox = get_crop_bbox(cfg, resized_img)
    resized_img = crop(resized_img, crop_bbox)
    resized_label = crop(resized_label, crop_bbox)
    img_metas['crop_bbox'] = crop_bbox

    # RandomFlip
    flip = True if np.random.rand() < 0.5 else False
    if flip:
        resized_img = np.flip(resized_img, axis=1)
        resized_img = resized_img.copy()
        resized_label = np.flip(resized_label, axis=1)
        resized_label = resized_label.copy()

    img = resized_img

    # PhotoMetricDistortion
    if random.randint(2):
        beta = random.uniform(-brightness_delta, brightness_delta)
        img = img.astype(np.float32) + beta
        img = np.clip(img, 0, 255).astype(np.uint8)
    mode = random.randint(2)
    if mode == 1:
        if random.randint(2):
            alpha = random.uniform(contrast_lower, contrast_upper)
            img = img.astype(np.float32) * alpha
            img = np.clip(img, 0, 255).astype(np.uint8)

    if random.randint(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        alpha = random.uniform(saturation_lower, saturation_upper)
        img[:, :, 1] = img[:, :, 1].astype(np.float32) * alpha
        img[:, :, 1] = np.clip(img[:, :, 1], 0, 255)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if random.randint(2):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 0] = (img[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)) % 180
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    if mode == 0:
        if random.randint(2):
            alpha = random.uniform(contrast_lower, contrast_upper)
            img = img.astype(np.float32) * alpha
            img = np.clip(img, 0, 255).astype(np.uint8)

    # Normalize
    img = img.copy().astype(np.float32)
    mean = np.array(cfg.DATASET.IMG_NORM_CFG.MEAN, dtype=np.float32)
    std = np.array(cfg.DATASET.IMG_NORM_CFG.STD, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = img.copy()
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)

    # Pad
    width = max(crop_size[1] - img.shape[1], 0)
    height = max(crop_size[0] - img.shape[0], 0)
    padding = (0, 0, width, height)
    img = cv2.copyMakeBorder(
        img,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=0)
    label = cv2.copyMakeBorder(
        resized_label,
        padding[1],
        padding[3],
        padding[0],
        padding[2],
        cv2.BORDER_CONSTANT,
        value=255)

    # # ImageToTensor
    img = np.ascontiguousarray(img.transpose(2, 0, 1))
    img = torch.from_numpy(img)
    label = torch.from_numpy(label.astype(np.int64))
    if return_meta:
        return img, label, img_metas
    else:
        return img, label


def val_preprocess(cfg, image, label=None, unlabeled=True, distill=False):
    img_scale = cfg.DATASET.SCALE

    img_np = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(img_np, IMREAD_COLOR)

    # Resize
    h, w = img.shape[:2]
    max_long_edge = max(img_scale)
    max_short_edge = min(img_scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))

    new_size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    resized_img = cv2.resize(img, new_size, dst=None, interpolation=cv2.INTER_LINEAR)

    new_h, new_w = resized_img.shape[:2]
    w_scale = new_w / w
    h_scale = new_h / h
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                            dtype=np.float32)
    # label
    if label is not None:
        with io.BytesIO(label) as buff:
            label = Image.open(buff)
            label = np.array(label)
        label = label.astype(np.int64)
#----------------------loveda数据集特定---------------------
        if cfg.DATASET.NAME == 'loveda':
            label[label == 7] = 255
        else:
#-----------------------------------------------------------
            if cfg.DATASET.REDUCE_ZERO_LABEL:
                label[label == 0] = 255
                label = label - 1
                label[label == 254] = 255
        if unlabeled:
            for i in range(cfg.DATASET.NUM_CLASSES):
                label[label == i] = -1
        label = cv2.resize(label, new_size, dst=None, interpolation=cv2.INTER_NEAREST)

    # Normalize
    img = resized_img.copy().astype(np.float32)
    mean = np.array(cfg.DATASET.IMG_NORM_CFG.MEAN, dtype=np.float32)
    std = np.array(cfg.DATASET.IMG_NORM_CFG.STD, dtype=np.float32)

    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    img = img.copy()
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)

    # ImageToTensor
    img = torch.from_numpy(img.transpose(2, 0, 1))

    if distill:
        label = torch.from_numpy(label.astype(np.int64))
        return img, label

    return img


def read_file_list(cfg):
    dataset = cfg.DATASET.NAME
    data_root = cfg.DATASET.DATAROOT #定义
    # 初始化所有可能返回的变量为空列表
    train_filenames = []
    val_filenames = []
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    output = []
    pseudo_classes = []

    if dataset == 'loveda':
        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_filenames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(train_img_dir) if f.lower().endswith('.png')]
        train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.lower().endswith('.png')]
        train_labels = [p.replace('img_dir', 'ann_dir') for p in train_images]
        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.lower().endswith('.png')]
        val_labels = [p.replace('img_dir', 'ann_dir') for p in val_images]
        val_filenames = [os.path.splitext(os.path.basename(p))[0] for p in val_images]

        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]
        pseudo_classes = []
        json_file_path = 'text/loveda_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        pseudo_classes.append([])
                        continue
            # 去掉JSON括号
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1]
            # 过滤空字符串再转 int
                    parts = [p for p in line.split(',') if p.strip()]
                    try:
                        splitted_s = sorted(int(p.strip()) for p in parts)
                    except ValueError as e:
                        print(f'Bad line skipped: {e}')
                        splitted_s = []
                    pseudo_classes.append(splitted_s)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")

    elif dataset == 'potsdam':
        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(train_img_dir) if nm.endswith('.png') and nm.startswith('top_potsdam_')]
        train_images = [os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train', f'{i}.png') for i in train_filenames]
        train_labels = [os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'train', f'{i}.png') for i in train_filenames]

        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(val_img_dir) if nm.endswith('.png') and nm.startswith('top_potsdam_')]
        val_images = [os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val', f'{i}.png') for i in val_filenames]
        val_labels = [os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'val', f'{i}.png') for i in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]
        pseudo_classes = []
        json_file_path = 'text/potsdam_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line[1:-2]
                    if ',' in line:
                        splitted_s = line.split(',')
                        splitted_s = list(map(int, splitted_s))
                        splitted_s.sort()
                    elif len(line) == 0:
                        splitted_s = []
                    else:
                        splitted_s = line
                        splitted_s = [int(splitted_s)]
                    pseudo_classes.append(splitted_s)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")

    elif dataset == 'vaihingen':
        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(train_img_dir) if nm.endswith('.png') and nm.startswith('top_mosaic_09cm_')]
        train_images = [os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train', f'{i}.png') for i in train_filenames]
        train_labels = [os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'train', f'{i}.png') for i in train_filenames]

        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(val_img_dir) if nm.endswith('.png') and nm.startswith('top_mosaic_09cm_')]
        val_images = [os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val', f'{i}.png') for i in val_filenames]
        val_labels = [os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'val', f'{i}.png') for i in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]
        pseudo_classes = []
        json_file_path = 'text/vaihingen_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line[1:-2]
                    if ',' in line:
                        splitted_s = line.split(',')
                        splitted_s = list(map(int, splitted_s))
                        splitted_s.sort()
                    elif len(line) == 0:
                        splitted_s = []
                    else:
                        splitted_s = line
                        splitted_s = [int(splitted_s)]
                    pseudo_classes.append(splitted_s)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")

    elif dataset == 'uavid':
        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_lbl_dir = os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'train')
        train_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(train_img_dir) if nm.endswith('.png')]
        train_images = [os.path.join(train_img_dir, f'{i}.png') for i in train_filenames]
        train_labels = [os.path.join(train_lbl_dir, f'{i}.png') for i in train_filenames]

        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_lbl_dir = os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'val')
        val_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(val_img_dir) if nm.endswith('.png')]
        val_images = [os.path.join(val_img_dir, f'{i}.png') for i in val_filenames]
        val_labels = [os.path.join(val_lbl_dir, f'{i}.png') for i in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]

        pseudo_classes = []
        json_file_path = 'text/uavid_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        pseudo_classes.append([])
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1].strip()
                    if not line:
                        pseudo_classes.append([])
                    else:
                        ids = sorted(int(x.strip()) for x in line.split(',') if x.strip())
                        pseudo_classes.append(ids)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")

    elif dataset == 'udd':
        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_lbl_dir = os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'train')
        train_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(train_img_dir) if nm.endswith('.png')]
        train_images = [os.path.join(train_img_dir, f'{i}.png') for i in train_filenames]
        train_labels = [os.path.join(train_lbl_dir, f'{i}.png') for i in train_filenames]

        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_lbl_dir = os.path.join(cfg.DATASET.DATAROOT, 'ann_dir', 'val')
        val_filenames = [os.path.splitext(nm)[0] for nm in os.listdir(val_img_dir) if nm.endswith('.png')]
        val_images = [os.path.join(val_img_dir, f'{i}.png') for i in val_filenames]
        val_labels = [os.path.join(val_lbl_dir, f'{i}.png') for i in val_filenames]

        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]

        pseudo_classes = []
        json_file_path = 'text/udd_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        pseudo_classes.append([])
                        continue
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1].strip()
                    if not line:
                        pseudo_classes.append([])
                    else:
                        # 提取并排序伪标签类别ID
                        ids = sorted(int(x.strip()) for x in line.split(',') if x.strip())
                        pseudo_classes.append(ids)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")


    elif dataset == 'isaid':

        train_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'train')
        train_filenames = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir(train_img_dir) if f.lower().endswith('.png')]
        train_images = [os.path.join(train_img_dir, f) for f in os.listdir(train_img_dir) if f.lower().endswith('.png')]
        train_labels = [p.replace('img_dir', 'ann_dir') for p in train_images]
        val_img_dir = os.path.join(cfg.DATASET.DATAROOT, 'img_dir', 'val')
        val_images = [os.path.join(val_img_dir, f) for f in os.listdir(val_img_dir) if f.lower().endswith('.png')]
        val_labels = [p.replace('img_dir', 'ann_dir') for p in val_images]
        val_filenames = [os.path.splitext(os.path.basename(p))[0] for p in val_images]


        output = [os.path.join(cfg.SAVE_DIR, f'{i}.pt') for i in val_filenames]
        pseudo_classes = []
        json_file_path = 'text/isaid_pseudo_label_ReCLIP.json'
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        pseudo_classes.append([])
                        continue
            # 去掉JSON括号
                    if line.startswith('[') and line.endswith(']'):
                        line = line[1:-1]
            # 过滤空字符串再转 int
                    parts = [p for p in line.split(',') if p.strip()]
                    try:
                        splitted_s = sorted(int(p.strip()) for p in parts)
                    except ValueError as e:
                        print(f'Bad line skipped: {e}')
                        splitted_s = []
                    pseudo_classes.append(splitted_s)
        except FileNotFoundError:
            print(f"Pseudo label file {json_file_path} not found.")

    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets are: context, voc, ade, stuff, cityscapes, loveda")
    # 添加，检查变量是否被正确初始化
    print(f"train_filenames length: {len(train_filenames)}")
    print(f"val_filenames length: {len(val_filenames)}")

    return train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, output, pseudo_classes  # file list

    
def prepare_dataset_cls_tokens(cfg, noun=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = cfg.DATASET.NAME
    if dataset == 'potsdam':
        dataset_classes = potsdam_classes
    elif dataset == 'uavid':
        dataset_classes = uavid_classes
    elif dataset == 'udd':
        dataset_classes = udd_classes
    elif dataset == 'open':
        noun.append('background')
        dataset_classes = noun
        model, _ = clip.load("ViT-B/16")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in noun:
                texts = [template.format(classname) for template in prompt_templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
            zeroshot_weights = zeroshot_weights.permute(1, 0).float()
    elif dataset == 'cityscapes' or dataset == 'gtav':
        dataset_classes = cityscapes_classes
    #增加LoveDA数据集
    elif dataset == 'loveda':
        dataset_classes = loveda_classes
    elif dataset == 'isaid':
        dataset_classes = isaid_classes
    elif dataset == 'vaihingen':
        dataset_classes = vaihingen_classes

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    cls_name_token = []

    for cls_idx in range(len(dataset_classes)):
        cls_name = dataset_classes[cls_idx]
        enhanced_name = RS_ENHANCE_MAPPING.get(cls_name,f"{cls_name} in satellite")
        token = clip.tokenize(enhanced_name)[0][1: 4]
        cls_name_token.append(token)
    cls_name_token = torch.stack(cls_name_token, dim=0)
    if dataset == 'open':
        return cls_name_token, zeroshot_weights.to(device)

    return cls_name_token, dataset_classes
