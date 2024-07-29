import argparse
import os
import torch
import torch.nn as nn
import numpy as np
# set seeds
torch.cuda.is_available()
torch.manual_seed(0)
np.random.seed(0)
import cv2
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.custom_dataset import CustomDataset
from utils.loss import FocalLoss, soft_dice_loss
from utils.general import get_random_prompts, mask2one_hot

from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory


def main(opt):
    # Create a dataset
    data_folder = opt.data  # define your dataset location here
    dataset = CustomDataset(data_folder, txt_name="trainval.txt")  ##  VOC seg is too small, used val for training

    batch_size = opt.batch_size  ## must be 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=CustomDataset.custom_collate)

    # original parameters
    sam_checkpoint = opt.sam_weights
    model_type = opt.model_type
    # if you have cuda-based gpu, no DDP
    device = f"cuda:{opt.device}"

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    num_epochs = opt.epochs
    point_prompt = opt.point_prompt
    box_prompt = opt.box_prompt
    ## if both, random drop one for better generalization ability
    point_box = (point_prompt and box_prompt)

    # model initialization
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.train()
    predictor = SamPredictor(sam)
    print(f"finished loading sam")

    # optimizer and scheduler
    lr = 1e-4
    momentum = 0.937
    weight_decay = 5e-4
    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)

    ## loss
    BCEseg = nn.BCELoss().to(device)
    losses = []
    best_loss = 1e10 ## early end for small loss
    # import monai
    # seg_loss = monai.losses.DiceFocalLoss(to_onehot_y=False, sigmoid=True, squared_pred=True, reduction='mean')

    # voc_classes = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7,
    #                'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
    #                'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

    # train
    print(f"strat training")
    model_transform = ResizeLongestSide(sam.image_encoder.img_size)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for idx, (images, gts, image_names) in enumerate(tqdm(dataloader)):
            valid_classes = []  ## voc 0,255 are ignored
            for i in range(images.shape[0]):
                image = images[i] # h,w,c np.uint8 rgb
                original_size = image.shape[:2] ## h,w
                input_size = model_transform.get_preprocess_shape(image.shape[0], image.shape[1],
                                                                  sam.image_encoder.img_size)  ##h,w
                gt = gts[i].copy() #h,w labels [0,1,2,..., classes-1]
                gt_classes = np.unique(gt)  ##masks classes: [0, 1, 2, 3, 4, 7]
                image_name = image_names[i]

                predictions = []
                ## freeze image encoder
                with torch.no_grad():
                    # gt_channel = gt[:, :, cls]
                    predictor.set_image(image, "RGB")
                    image_embedding = predictor.get_image_embedding()
                for cls in gt_classes:
                    if isinstance(cls, torch.Tensor):
                        cls = cls.item()
                    ## voc 0 is background, 255 is border; ignore 0,255
                    if cls == 0 or cls == 255:
                        continue
                    (foreground_points, background_points), bbox = get_random_prompts(gt, cls)
                    # if the model can't generate any sparse prompts
                    if len(foreground_points) == 0:
                        print(f"======== zero points =============")
                        continue
                    valid_classes.append(cls)
                    if not point_prompt:
                        points = None
                    else:
                        all_points = np.concatenate((foreground_points, background_points), axis=0)
                        all_points = np.array(all_points)
                        point_labels = np.array([1] * foreground_points.shape[0] + [0] * background_points.shape[0], dtype=int)
                        ## image resized to 1024, points also
                        all_points = model_transform.apply_coords(all_points, original_size)

                        all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                        point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                        all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                        points = (all_points, point_labels)

                    if not box_prompt:
                        box_torch=None
                    else:
                        ## preprocess bbox
                        box = model_transform.apply_boxes(bbox, original_size)
                        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                        box_torch = box_torch[None, :]
                    ## if both, random drop one for better generalization ability
                    if point_box and np.random.random()<0.5:
                        if np.random.random()<0.25:
                            points = None
                        elif np.random.random()>0.75:
                            box_torch = None
                    ## freeze prompt encoder
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points = points,
                            boxes = box_torch,
                            # masks=mask_predictions,
                            masks=None,
                        )
                    ## predicted masks, three level
                    mask_predictions, scores = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=True,
                    )
                    # Choose the model's best mask
                    mask_input = mask_predictions[:, torch.argmax(scores),...].unsqueeze(1)
                    with torch.no_grad():
                        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                            points=points,
                            boxes=box_torch,
                            masks=mask_input,
                        )
                    ## predict a better mask, only one mask
                    mask_predictions, scores = sam.mask_decoder(
                        image_embeddings=image_embedding.to(device),
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )
                    best_mask = sam.postprocess_masks(mask_predictions, input_size, original_size)
                    predictions.append(best_mask)

                predictions = torch.cat(predictions, dim=1)
            gts = torch.from_numpy(gts).unsqueeze(1) ## BxHxW ---> Bx1xHxW
            gts_onehot = mask2one_hot(gts, valid_classes)
            gts_onehot = gts_onehot.to(device)

            predictions = torch.sigmoid(predictions)
            # #loss = seg_loss(predictions, gts_onehot)
            loss = BCEseg(predictions, gts_onehot)
            loss_dice = soft_dice_loss(predictions, gts_onehot, smooth = 1e-5, activation='none')
            loss = loss + loss_dice

            print(f"epoch: {epoch} at idx:{idx} --- loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= idx

        losses.append(epoch_loss)
        scheduler.step()
        print(f'EPOCH: {epoch+1}, Loss: {epoch_loss}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            mask_decoder_weighs = sam.mask_decoder.state_dict()
            mask_decoder_weighs = {f"mask_decoder.{k}": v for k,v in mask_decoder_weighs.items() }
            torch.save(mask_decoder_weighs, os.path.join(save_dir, f'sam_decoder_fintune_{str(epoch+1)}_pointbox_monai.pth'))
            print("Saving weights, epoch: ", epoch+1)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-weights', '--w', type=str, default=ROOT / 'weights/sam_vit_b_01ec64.pth', help='original sam weights path')
    parser.add_argument('--model-type', '--type', type=str, default='vit_b', help='sam model type: vit_b, vit_l, vit_h')
    parser.add_argument('--data', type=str, default=ROOT /'data_example/VOCdevkit', help='your VOCdevkit dataset path')
    parser.add_argument('--point-prompt', type=bool, default=True, help='use point prompt')
    parser.add_argument('--box-prompt', type=bool, default=True, help='use box prompt')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, must be 1 for voc')
    parser.add_argument('--save_dir', default=ROOT / 'runs', help='path to save checkpoint')
    parser.add_argument('--device', default='0', help='cuda device only one, 0 or 1 or 2...')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)