import numpy as np
import torch
import clip
import argparse
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

loveda_classes = [ "background","building","road","water","barren","forest","agricultural"]
potsdam_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
vaihingen_classes = ["background", "impervious surface", "building", "low vegetation", "tree", "car"]
isaid_classes = ["background", "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
                 "ground track field", "bridge", "large vehicle","small vehicle", "helicopter", "swimming pool",
                 "roundabout","soccer ball field", "plane", "harbor"]
uavid_classes = ["background", "building", "road", "tree", "low vegetation", "moving car", "static car", "human"]
udd_classes = ["vegetation", "building", "road", "vehicle", "other"]

RS_ENHANCE_MAPPING = {
    # --- LoveDA ---
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

    # --- iSAID (核心是区分小目标) ---
    "ship": "vessel on water",
    "storage tank": "circular oil tank",
    "baseball diamond": "baseball field diamond",
    "tennis court": "aerial tennis court",
    "basketball court": "aerial basketball court",
    "ground track field": "oval running track",
    "bridge": "aerial bridge structure",
    "large vehicle": "overhead truck bus",
    "small vehicle": "overhead small car",
    "helicopter": "parked aerial helicopter",
    "swimming pool": "blue swimming pool",
    "roundabout": "circular road roundabout",
    "soccer ball field": "green soccer field",
    "plane": "airplane on tarmac",
    "harbor": "harbor port terminal",

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
prompt_templates = ['a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.', 'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.', 'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.', 'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.', 'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.', 'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.', 'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.', 'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.', 'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.', 'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.', 'there is a {} in the scene.', 'there is the {} in the scene.', 'this is a {} in the scene.', 'this is the {} in the scene.', 'this is one {} in the scene.']

def parse_args():
    parser = argparse.ArgumentParser(description='Prompt engeering script')
    parser.add_argument('--model', default='ViT16', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT32', 'ViT16'], help='clip model name')
    parser.add_argument('--class-set', default=['stuff'], nargs='+',
        choices=['city', 'ade', 'stuff', 'voc', 'context', 'acontext', 'mickey', 'batman', 'mario', 'gates', 'blur', 'sports', 'car_brands', 'batman_ext', 'car_color', 'loveda','potsdam', 'isaid', 'vaihingen', 'uavid', 'udd'],
        help='the set of class names')
    parser.add_argument('--no-prompt-eng', action='store_true', help='disable prompt engineering')

    args = parser.parse_args()
    return args

def zeroshot_classifier(model_name, classnames, templates):
    model, _ = clip.load(model_name)
    model.cuda().eval()
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            raw_enhanced_name = RS_ENHANCE_MAPPING.get(classname, f"{classname} in satellite")
            all_tokens = clip.tokenize(raw_enhanced_name)[0]
            token_ids = [tid for tid in all_tokens[1:4].tolist() if tid not in [0, 49407]]
            enhanced_name = _tokenizer.decode(token_ids).strip()
            print(f"DEBUG: {classname} -> 最终使用的增强词: '{enhanced_name}'")
            texts = [template.format(enhanced_name) for template in templates]
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


if __name__ == '__main__':
    args = parse_args()

    classes = []
    all_set_name = ''
    name_mapping = {'loveda': loveda_classes, 'potsdam':potsdam_classes, 'isaid':isaid_classes,'vaihingen':vaihingen_classes, 'uavid': uavid_classes, 'udd':udd_classes}
    for set_name in args.class_set:
        if set_name in name_mapping:
            classes += name_mapping[set_name]
            all_set_name += '_{}'.format(set_name)
        if set_name in ['blur'] or args.no_prompt_eng:
            prompt_templates = ['a photo of a {}.']
    # remove redundant classes
    classes = list(dict.fromkeys(classes))
    # remove the first underline
    all_set_name = all_set_name[1:]
    print(classes)

    print(f"{len(classes)} class(es), {len(prompt_templates)} template(s)")

    name_mapping = {'RN50': 'RN50', 'RN101': 'RN101', 'RN50x4': 'RN50x4', 'RN50x16': 'RN50x16', 'ViT32': 'ViT-B/32', 'ViT16': 'ViT-B/16', 'ViT-L': 'ViT-L/14'}
    zeroshot_weights = zeroshot_classifier(name_mapping[args.model], classes, prompt_templates)
    zeroshot_weights = zeroshot_weights.permute(1, 0).float()
    print(zeroshot_weights.shape)

    prefix = f'text/{all_set_name}_{args.model}'
    if args.no_prompt_eng:
        prefix += '_npe'

    torch.save(zeroshot_weights, f'{prefix}_clip_text.pth')
