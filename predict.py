# 这是比赛提交的predict文件，predict.py接收两个参数：一个是src_image_dir，另一个是结果保存目录
# 创建ZIP压缩包，所有提交内容均放在压缩包的根目录下，比如：result.zip 解压缩之后就是提交的内容文件，而不应该是result文件夹；压缩包中的内容可能如下：
# | - root
#       - predict.py
#       | - model
#            - deeplabv3_resnet50.onnx
#            - deepfillv2_G.onnx

import os
import sys
import glob
import cv2
import numpy as np
import onnxruntime as rt

def crop_region(img, img_mask, l=1536, r=6):
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[-1])
    height, width = img.shape[0], img.shape[1]
    cx, cy = int((x + w / 2)), int((y + h / 2))
    if cx < l / 2:
        cx = l / 2
    if cy < l / 2:
        cy = l / 2
    right = width - cx - l / 2
    bottom = height - cy - l / 2
    if right >= 0:
        if bottom >= 0:
            region = img[int(cy - l / 2):int(cy + l / 2), int(cx - l / 2):int(cx + l / 2), :]
            mask = img_mask[int(cy - l / 2):int(cy + l / 2), int(cx - l / 2):int(cx + l / 2)]
            x, y = int(cx - l / 2), int(cy - l / 2)
        else:
            region = img[height - l:height, int(cx - l / 2):int(cx + l / 2), :]
            mask = img_mask[height - l:height, int(cx - l / 2):int(cx + l / 2)]
            x, y = int(cx - l / 2), int(height - l)
    else:
        if bottom >= 0:
            region = img[int(cy - l / 2):int(cy + l / 2), width - l:width, :]
            mask = img_mask[int(cy - l / 2):int(cy + l / 2), width - l:width]
            x, y = int(width - l ), int(cy - l / 2)
        else:
            region = img[height - l:height, width - l:width, :]
            mask = img_mask[height - l:height, width - l:width]
            x, y = int(width - l), int(height - l)
    length = int(l / r)
    region = cv2.resize(region, (length, length))
    mask = cv2.resize(mask, (length, length))
    return region, mask, [x, y, l]


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        # examine the shape
        trans = False
        img_cover = cv2.imread(image_path)
        if img_cover.shape[0] < img_cover.shape[1]:
            img_cover = img_cover.transpose(1, 0, 2)
            trans = True
        # preprocess to get the shape (1, 3, 480, 640)
        img = cv2.resize(img_cover, (480, 640))
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = (img / 255 - 0.6) / 0.2
        img = img[np.newaxis, :]
        # deeplabv3
        sess = rt.InferenceSession("model/deeplabv3_resnet50.onnx", providers=['CUDAExecutionProvider'])
        # segment result
        result = sess.run(['output'], {'input': img})
        segment = result[0].squeeze(0).argmax(0).astype(np.uint8)
        # dilate target region to remove edge interference
        kernel = np.ones((4, 4), np.uint8)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))  # 获取圆形结构元素
        segment = cv2.dilate(segment, kernel)
        segment = segment * 128
        # deepfillv2
        sess2 = rt.InferenceSession("model/deepfillv2_G.onnx", providers=['CUDAExecutionProvider'])
        # get region image, mask and location
        img_mask = cv2.resize(segment, (img_cover.shape[1], img_cover.shape[0]))
        img, mask, loc = crop_region(img=img_cover, img_mask=img_mask)
        # normalize
        img = img.transpose(2, 0, 1) / 255
        mask = mask[np.newaxis, :] / 128
        masked_img = img * (1 - mask)
        # expand dim
        masked_img = masked_img[np.newaxis, :].astype(np.float32)
        mask = mask[np.newaxis, :].astype(np.float32)
        # result
        result1, result2 = sess2.run(['first_out', 'second_out'], {'img': masked_img, 'mask': mask})
        # get the mask and result of the original image shape
        masko = mask[0][0].astype(np.uint8)
        masko = cv2.resize(masko, (1536, 1536))
        masko = masko[:, :, np.newaxis]
        result2 = result2[0].transpose(1, 2, 0) * 255
        # # be darker
        # result2 = result2 - 5
        result2 = cv2.resize(result2, (1536, 1536))
        # replace the masked region
        x, y, l = loc[0], loc[1], loc[2]
        img_cover[y:y + l, x:x + l, :] = img_cover[y:y + l, x:x + l, :] * (1-masko) + result2 * masko
        if trans:
            img_cover = img_cover.transpose(1, 0, 2)
        # save image
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img_cover)


if __name__ == "__main__":

    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)
