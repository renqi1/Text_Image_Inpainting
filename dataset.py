import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform

ALLMASKTYPES = ['single_bbox', 'bbox', 'free_form']

class InpaintDataset(Dataset):
    def __init__(self, opt):
        assert opt.mask_type in ALLMASKTYPES
        self.opt = opt
        self.save_path = opt.augment_data_path
        self.img_list = os.listdir(self.save_path)
        self.img_path_list = [os.path.join(self.save_path, img) for img in self.img_list]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.img_path_list[index])
        # mask
        if self.opt.mask_type == 'single_bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = 1)
        if self.opt.mask_type == 'bbox':
            mask = self.bbox2mask(shape = self.opt.imgsize, margin = self.opt.margin, bbox_shape = self.opt.bbox_shape, times = self.opt.mask_num)
        if self.opt.mask_type == 'free_form':
            mask = self.random_ff_mask(shape = self.opt.imgsize, max_angle = self.opt.max_angle, max_len = self.opt.max_len, max_width = self.opt.max_width, times = self.opt.mask_num)
        
        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32)).contiguous()
        return img, mask

    def random_ff_mask(self, shape, max_angle = 4, max_len = 40, max_width = 10, times = 15):
        """Generate a random free form mask with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES,
                VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        times = np.random.randint(times)
        for i in range(times):
            start_x = np.random.randint(width)
            start_y = np.random.randint(height)
            for j in range(1 + np.random.randint(5)):
                angle = 0.01 + np.random.randint(max_angle)
                if i % 2 == 0:
                    angle = 2 * 3.1415926 - angle
                length = 10 + np.random.randint(max_len)
                brush_w = 10 + np.random.randint(max_width)
                end_x = (start_x + length * np.sin(angle)).astype(np.int32)
                end_y = (start_y + length * np.cos(angle)).astype(np.int32)
                cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
                start_x, start_y = end_x, end_y
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
    
    def random_bbox(self, shape, margin, bbox_shape):
        """Generate a random tlhw with configuration.
        Args:
            config: Config should have configuration including IMG_SHAPES, VERTICAL_MARGIN, HEIGHT, HORIZONTAL_MARGIN, WIDTH.
        Returns:
            tuple: (top, left, height, width)
        """
        img_height = shape
        img_width = shape
        height = bbox_shape
        width = bbox_shape
        ver_margin = margin
        hor_margin = margin
        maxt = img_height - ver_margin - height
        maxl = img_width - hor_margin - width
        t = np.random.randint(low = ver_margin, high = maxt)
        l = np.random.randint(low = hor_margin, high = maxl)
        h = height
        w = width
        return (t, l, h, w)

    def bbox2mask(self, shape, margin, bbox_shape, times):
        """Generate mask tensor from bbox.
        Args:
            bbox: configuration tuple, (top, left, height, width)
            config: Config should have configuration including IMG_SHAPES,
                MAX_DELTA_HEIGHT, MAX_DELTA_WIDTH.
        Returns:
            tf.Tensor: output with shape [1, H, W, 1]
        """
        bboxs = []
        for i in range(times):
            bbox = self.random_bbox(shape, margin, bbox_shape)
            bboxs.append(bbox)
        height = shape
        width = shape
        mask = np.zeros((height, width), np.float32)
        for bbox in bboxs:
            h = int(bbox[2] * 0.1) + np.random.randint(int(bbox[2] * 0.2 + 1))
            w = int(bbox[3] * 0.1) + np.random.randint(int(bbox[3] * 0.2) + 1)
            mask[(bbox[0] + h) : (bbox[0] + bbox[2] - h), (bbox[1] + w) : (bbox[1] + bbox[3] - w)] = 1.
        return mask.reshape((1, ) + mask.shape).astype(np.float32)
        
class FillDataset(Dataset):
    def __init__(self, opt):
        self.imgsize = opt.imgsize
        self.imgresize = opt.imgresize
        self.root = opt.baseroot
        self.data1 = os.path.join(opt.baseroot, 'train_data_1')
        self.data2 = os.path.join(opt.baseroot, 'train_data_2')
        self.data3 = os.path.join(opt.baseroot, 'train_data_3')
        self.data4 = os.path.join(opt.baseroot, 'train_data_4')
        self.data5 = os.path.join(opt.baseroot, 'train_data_5')
        self.img_real = []
        self.img_mask = []
        self.get_img_and_mask(self.data1)
        self.get_img_and_mask(self.data2)
        self.get_img_and_mask(self.data3)
        self.get_img_and_mask(self.data4)
        self.get_img_and_mask(self.data5)

    def __len__(self):
        return len(self.img_real)

    def __getitem__(self, index):
        # image
        img_real = cv2.imread(self.img_real[index])
        img_mask = cv2.imread(self.img_mask[index])
        if img_real.shape[0] < img_real.shape[1]:
            img_real = img_real.transpose(1, 0, 2)
            img_mask = img_mask.transpose(1, 0, 2)
        img, mask = self.crop_region(img_real, img_mask, l=self.imgsize, r=self.imgresize)

        # the outputs are entire image and mask, respectively
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 128).unsqueeze(0).contiguous()
        return img, mask

    def get_img_and_mask(self, data):
        annotation = os.path.join(data, 'annotation.txt')
        with open(annotation) as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                img = os.path.join(data, splited[0])
                mask = os.path.join(data, splited[2])
                self.img_real.append(img)
                self.img_mask.append(mask)


    def crop_region(self, img, img_mask, l=1536, r=6):
        contours, hierarchy = cv2.findContours(img_mask[:, :, 1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                mask = img_mask[int(cy - l / 2):int(cy + l / 2), int(cx - l / 2):int(cx + l / 2), :]
            else:
                region = img[height - l:height, int(cx - l / 2):int(cx + l / 2), :]
                mask = img_mask[height - l:height, int(cx - l / 2):int(cx + l / 2), :]
        else:
            if bottom >= 0:
                region = img[int(cy - l / 2):int(cy + l / 2), width - l:width, :]
                mask = img_mask[int(cy - l / 2):int(cy + l / 2), width - l:width, :]
            else:
                region = img[height - l:height, width - l:width, :]
                mask = img_mask[height - l:height, width - l:width, :]
        length = int(l/r)
        region = cv2.resize(region, (length, length))
        mask = cv2.resize(mask, (length, length))[:, :, 1]

        return region, mask

class SegmentDataset(Dataset):
    def __init__(self, opt):
        self.root = opt.baseroot
        self.data1 = os.path.join(opt.baseroot, 'train_data_1')
        self.data2 = os.path.join(opt.baseroot, 'train_data_2')
        self.data3 = os.path.join(opt.baseroot, 'train_data_3')
        self.data4 = os.path.join(opt.baseroot, 'train_data_4')
        self.data5 = os.path.join(opt.baseroot, 'train_data_5')
        self.img_cover = []
        self.img_mask = []
        self.get_img_and_mask(self.data1)
        self.get_img_and_mask(self.data2)
        self.get_img_and_mask(self.data3)
        self.get_img_and_mask(self.data4)
        self.get_img_and_mask(self.data5)

    def __len__(self):
        return len(self.img_cover)

    def __getitem__(self, index):
        img_cover = cv2.imread(self.img_cover[index])
        img_mask = cv2.imread(self.img_mask[index])
        if img_cover.shape[0] < img_cover.shape[1]:
            img_cover= img_cover.transpose(1, 0, 2)
            img_mask = img_mask.transpose(1, 0, 2)

        img = cv2.resize(img_cover, (480, 640))
        img = transform.Compose([transform.ToTensor(), transform.Normalize(mean=[0.6, 0.6, 0.6], std=[0.2, 0.2, 0.2])])(img)
        mask = cv2.resize(img_mask[:, :, 1], (480, 640))/128

        return img, mask

    def get_img_and_mask(self, data):
        annotation = os.path.join(data, 'annotation.txt')
        with open(annotation) as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                img = os.path.join(data, splited[1])
                mask = os.path.join(data, splited[2])
                self.img_cover.append(img)
                self.img_mask.append(mask)

