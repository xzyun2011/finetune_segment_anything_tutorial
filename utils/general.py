import torch
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def get_random_prompts(mask, mask_value, foreground_nums=1, background_nums=1):
    # Find the indices (coordinates) of the foreground pixels
    foreground_indices = np.argwhere(mask == mask_value)
    ymin, xmin= foreground_indices.min(axis=0)
    ymax, xmax = foreground_indices.max(axis=0)
    bbox = np.array([xmin, ymin, xmax, ymax])
    if foreground_indices.shape[0] < foreground_nums:
        foreground_nums = foreground_indices.shape[0]
        background_nums = int(0.5 * foreground_indices.shape[0])
    background_indices = np.argwhere(mask != mask_value)

    ## random select
    foreground_points = foreground_indices[
        np.random.choice(foreground_indices.shape[0], foreground_nums, replace=False)]
    background_points = background_indices[
        np.random.choice(background_indices.shape[0], background_nums, replace=False)]

    ## 坐标点是(y,x)，输入给网络应该是(x,y),需要翻一下顺序
    foreground_points = foreground_points[:, ::-1]
    background_points = background_points[:, ::-1]

    return (foreground_points, background_points), bbox


def mask2one_hot(label, gt_classes):
    """
    label: 标签图像 # (batch_size, 1, h, w)
    num_classes: 分类类别数
    """
    current_label = label.squeeze(1) # （batch_size, 1, h, w) ---> （batch_size, h, w)
    batch_size, h, w = current_label.shape[0], current_label.shape[1], current_label.shape[2]
    one_hots = []
    for cls in gt_classes:
        if isinstance(cls, torch.Tensor):
            cls = cls.item()
        tmplate = torch.zeros(batch_size, h, w)  # （batch_size, h, w)
        tmplate[current_label == cls] = 1
        tmplate = tmplate.view(batch_size, 1, h, w)  # （batch_size, h, w) --> （batch_size, 1, h, w)
        one_hots.append(tmplate)
    onehot = torch.cat(one_hots, dim=1)
    return onehot


def save_mask_debug(masks, scores):
    scores_np = scores[0].detach().cpu().numpy()
    masks_np = masks[0].detach().cpu().numpy()
    mask_best = masks_np[np.argmax(scores_np), :, :]
    mask_best = (mask_best*255).astype(np.uint8)
    cv2.imwrite("mask_best.png", mask_best)

def save_points_debug(points, gt, cls, image_name):
    tmp_img = np.zeros(gt.shape)
    save_name = f"{image_name.split('.')[0]}_{str(cls)}.jpg"
    all_points, point_labels = points
    for (x, y), label in zip(all_points, point_labels):
        tmp_img[y, x] = int(label*144)
        if len(all_points) < 10:
            color = 80 if int(label)==1 else 200
            cv2.circle(tmp_img, (x,y), radius = 5,color = color,thickness = 6)
            cv2.putText(tmp_img, f"{str(int(label))}", (x,y), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)
    cv2.imwrite(f"{save_name}", tmp_img)


def read_xml(annotation_path):
    '''
    Read xml annotation and return a list of boxes

    :param annotation_path (str): annotation path
           resized (bool): image has been resized
    :return:
        obj_bndbox_set (list):  a list of boxes
    '''

    assert isinstance(annotation_path, str), "input path format error"
    assert annotation_path.split('.')[-1]=="xml", "input file format error, must be .xml file"

    tree = ET.ElementTree(file=annotation_path)  # 打开文件，解析成一棵树型结构
    root = tree.getroot()  # 获取树型结构的根
    obj_set = root.findall('object')  # 找到文件中所有含有object关键字的地方，这些地方含有标注目标
    obj_bndbox_set = [] # 以目标类别为关键字，目标框为值组成的字典结构

    ## image size
    img_path = annotation_path.replace("annotation_voc", "suitcase").replace(".xml", ".jpg")
    # img_w, img_h = imagesize.get(img_path)
    size = root.findall('size')
    width = int(size[0].find('width').text)
    height = int(size[0].find('height').text)
    # assert (img_w, img_h) == (width, height), f"{img_path} original: {img_w,img_h}, xml: {width, height}"

    for obj in obj_set:
        name = obj.find('name')
        find_res = obj.find('name')
        if find_res is None:
            print("no name")
            continue
        obj_name = find_res.text.lower()
        bndbox = obj.find('bndbox')
        x1 = int(round(float(bndbox.find('xmin').text)))  # -1 #-1是因为程序是按0作为起始位置的
        y1 = int(round(float(bndbox.find('ymin').text)))  # -1
        x2 = int(round(float(bndbox.find('xmax').text)))  # -1
        y2 = int(round(float(bndbox.find('ymax').text)))  # -1
        bndbox_loc = [x1, y1, x2, y2]
        obj_bndbox_set.append( bndbox_loc)
    return obj_bndbox_set