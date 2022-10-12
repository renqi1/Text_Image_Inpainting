import os
import numpy as np
import cv2

def random_bbox(img_height, img_width, margin=200, bbox_shape=1600, box_num=10):
    bbox_list = []
    height = bbox_shape
    width = bbox_shape
    ver_margin = margin
    hor_margin = margin
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    for i in range(box_num):
        t = np.random.randint(low=ver_margin, high=maxt)
        l = np.random.randint(low=hor_margin, high=maxl)
        bbox_list.append([t, l, height, width])
    return bbox_list


gt_path = '/home/gzz/机械硬盘/sda3/text_image_inpainting/train/train_data_6/gt'
save_path = '/home/gzz/机械硬盘/sda3/text_image_inpainting/train/train_data_6/gt_crop'
img_list = os.listdir(gt_path)
img_path_list = [os.path.join(gt_path, img) for img in img_list]

for i in range(len(img_list)):
    img_path = img_path_list[i]
    img_name = img_list[i].split('.')[0]

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[0], img.shape[1]
    if img_h < img_w:
        img=img.transpose(1, 0, 2)
    box_list = random_bbox(img_h, img_w)

    for j, box in enumerate(box_list):
        t, l, h, w = box
        result = img[t:t+h, l:l+w, :]
        result = cv2.resize(result, (256, 256))
        save_img = os.path.join(save_path, img_name + str(j)+'.jpg')
        cv2.imwrite(save_img, result)


