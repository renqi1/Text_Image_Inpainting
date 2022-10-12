import argparse
import cv2
import numpy as np
import torch
import utils
import torchvision.transforms as transform

def crop_region(img, img_mask, l=1536,  r=6):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type = str, default = 'test_img/test1.jpg')
    parser.add_argument('--imgsize', type = int, default = 1536, help = 'size of image')
    parser.add_argument('--imgresize', type=int, default = 6, help='image resize ratio')
    parser.add_argument('--deepfillv2_generator_pretrain', type = str, default = './pytorch_model/deepfillv2_G.pth', help = 'load generator pretrain model')
    parser.add_argument('--deeplabv3_pretrain', type = str, default = './pytorch_model/deeplabv3_resnet50.pth', help='load segmentnet pretrain model')
    parser.add_argument('--backbone', type=str, default='resnet50', help='segmentnet, deeplabv3 backbone')
    parser.add_argument('--in_c', type = int, default = 3, help = 'segmentnet, input RGB image')
    parser.add_argument('--out_c', type = int, default = 2, help = 'segmentnet, output segment foreground and background')
    parser.add_argument('--in_channels', type = int, default = 4, help = 'input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    opt = parser.parse_args()

    img_cover = cv2.imread(opt.image_path)

    segmentnet = utils.create_segmentnet(opt).cuda()
    segmentnet.eval()
    imgr = cv2.resize(img_cover, (480, 640))
    imgrt = transform.Compose([transform.ToTensor(), transform.Normalize(mean=[0.6, 0.6, 0.6], std=[0.2, 0.2, 0.2])])(imgr).cuda().unsqueeze(0)
    result = segmentnet(imgrt)
    segment = result.argmax(1).squeeze(0)
    segment = segment.detach().cpu().numpy().astype(np.uint8) * 128

    imgr[:, :, 2] = segment

    img_mask = cv2.resize(segment, (img_cover.shape[1], img_cover.shape[0]))
    img, mask, loc = crop_region(img=img_cover, img_mask=img_mask, l=opt.imgsize, r=opt.imgresize)


    img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).contiguous().cuda()
    mask = torch.from_numpy(mask.astype(np.float32) / 128).unsqueeze(0).unsqueeze(0).contiguous().cuda()
    masked_img = img * (1 - mask)

    deepfillv2 = utils.create_generator(opt).cuda()
    deepfillv2.eval()
    fake1, fake2 = deepfillv2(masked_img, mask)

    # forward propagation
    fusion_fake1 = img * (1 - mask) + fake1 * mask                      # in range [-1, 1]
    fusion_fake2 = img * (1 - mask) + fake2 * mask                      # in range [-1, 1]

    # convert to visible image format
    fusion_fake1 = fusion_fake1[0].detach().cpu().numpy().transpose(1, 2, 0)
    fusion_fake1 = fusion_fake1 * 255
    fusion_fake1 = fusion_fake1.astype(np.uint8)
    fusion_fake2 = fusion_fake2[0].detach().cpu().numpy().transpose(1, 2, 0)
    fusion_fake2 = fusion_fake2 * 255
    fusion_fake2 = fusion_fake2.astype(np.uint8)

    fusion_fake1 = cv2.resize(fusion_fake1, (opt.imgsize, opt.imgsize))
    fusion_fake2 = cv2.resize(fusion_fake2, (opt.imgsize, opt.imgsize))

    x, y, l = loc[0], loc[1], loc[2]
    img_cover1 = img_cover.copy()
    img_cover2 = img_cover.copy()
    img_cover1[y:y+l, x:x+l, :] = fusion_fake1
    img_cover2[y:y + l, x:x + l, :] = fusion_fake2
    img_cover1 = cv2.resize(img_cover1, (480, 640))
    img_cover2 = cv2.resize(img_cover2, (480, 640))


    # show
    show_img = np.concatenate((imgr, img_cover1, img_cover2), axis=1)
    b, g, r = cv2.split(show_img)
    show_img = cv2.merge([b, g, r])
    cv2.imshow('comparison.jpg', show_img)
    cv2.waitKey(0)
    # cv2.imwrite('result.jpg' % batch_idx, img_cover2)


