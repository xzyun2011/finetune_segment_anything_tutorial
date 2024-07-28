import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from utils.general import read_xml
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

## from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, cls):
    color = colors[int(cls)]
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main(opt):
    # original parameters
    sam_checkpoint = opt.sam_weights
    model_type = opt.model_type
    # if you have cuda-based gpu, no DDP
    device = f"cuda:{opt.device}"
    new_decoder_path = opt.decoder_weights
    if new_decoder_path:
        assert os.path.exists(new_decoder_path), f"{new_decoder_path} not exist"
    finetuned = False if not os.path.exists(new_decoder_path) else True

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    if finetuned:
        # update decoder weights
        state_dict = sam.state_dict()
        new_decoder_dict = torch.load(new_decoder_path, map_location=state_dict[list(state_dict.keys())[0]].device)
        state_dict.update(new_decoder_dict)
        res = sam.load_state_dict(state_dict, strict=False)
        print(f"load res: {res}")

    sam.to(device=device)
    predictor = SamPredictor(sam)

    data_folder = opt.data
    image_dir = os.path.join(data_folder, "VOC2007/JPEGImages")
    val_txt = os.path.join(data_folder, "VOC2007/ImageSets/Segmentation/test.txt")
    with open(val_txt, "r") as f:
        file_names = f.readlines()
    file_names = [name.strip() for name in file_names]
    img_lists = [Path(f"{image_dir}/{name}.jpg") for name in file_names]
    # img_lists = list(Path(image_dir).glob("*.jpg"))
    img_lists.sort()
    xml_root = Path(image_dir.replace("JPEGImages","Annotations"))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ## TODO: change  img_lists len
    for img_path in img_lists[:20]:
        img_name = img_path.stem
        xml_path = xml_root/f"{img_name}.xml"
        bboxes = read_xml(str(xml_path))
        save_name = f"{img_name}_finetuned.png" if finetuned else f"{img_name}.png"
        save_path = os.path.join(save_dir, save_name)
        image = cv2.imread(str(img_path))
        image = image[...,::-1]
        predictor.set_image(image, "RGB")

        input_boxes = torch.tensor(bboxes, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        plt.figure(figsize=(9.6, 5.4))
        plt.imshow(image)
        for cls,mask in enumerate(masks):
            show_mask(mask.cpu().numpy(), plt.gca(), cls)
        for box in input_boxes:
            show_box(box.cpu().numpy(), plt.gca())
        plt.axis('off')


        print(f"save to: {save_path}")
        plt.savefig(save_path)
        # plt.show()
        plt.close()



def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-weights', type=str, default=ROOT / 'weights/sam_vit_b_01ec64.pth', help='original sam weights path')
    parser.add_argument('--decoder-weights', type=str, default=ROOT / "weights/sam_decoder_fintune_pointbox.pth", help='finetuned decoder weights path')
    parser.add_argument('--model-type', type=str, default='vit_b', help='sam model type: vit_b, vit_l, vit_h')
    parser.add_argument('--data', type=str, default='/cv/datasets/voc/VOCdevkit', help='VOCdevkit dataset path')
    parser.add_argument('--save_dir', default=ROOT / 'runs/predict_test', help='path to save checkpoint')
    parser.add_argument('--device', default='0', help='cuda device only one, 0 or 1 or 2...')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    colors = np.hstack((np.random.random((21, 3)), np.ones((21, 1)) * 0.6))
    main(opt)
