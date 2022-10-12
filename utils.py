import os
import numpy as np
import cv2
import torch
from network import network


# ----------------------------------------
#                 Network
# ----------------------------------------

def create_segmentnet(opt):
    segmentnet = network.SegmentNet(opt)
    print('SegmentNet is created!')
    if opt.deeplabv3_pretrain:
        pretrained_dict = torch.load(opt.deeplabv3_pretrain)
        segmentnet.load_state_dict(pretrained_dict)
        # # use this way to load state dict if you load the official pretrained onnx_model
        # segmentnet.load_pretrained_deeplabv3_state_dict(pretrained_dict['model_state'])
    return segmentnet


def create_generator(opt):
    # Initialize the networks
    generator = network.GatedGenerator(opt)
    print('Generator is created!')
    if opt.deepfillv2_generator_pretrain:
        pretrained_dict = torch.load(opt.deepfillv2_generator_pretrain)
        generator.load_state_dict(pretrained_dict)
    else:
        # Init the networks
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    if opt.deepfillv2_discriminator_pretrain:
        pretrained_dict = torch.load(opt.deepfillv2_discriminator_pretrain)
        discriminator.load_state_dict(pretrained_dict)
    else:
        # Init the networks
        network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator



# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def save_sample_png(sample_folder, sample_name, img_list, name_list, pixel_max_cnt = 255):
    # Save image one-by-one
    for i in range(len(img_list)):
        img = img_list[i]
        # Recover normalization: * 255 because last layer is tanh activated
        img = img * 255
        # Process img_copy and do not destroy the data of img
        img_copy = img.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        img_copy = np.clip(img_copy, 0, pixel_max_cnt)
        img_copy = img_copy.astype(np.uint8)
        # Save to certain path
        save_img_name = sample_name + '_' + name_list[i] + '.png'
        save_img_path = os.path.join(sample_folder, save_img_name)
        cv2.imwrite(save_img_path, img_copy)

def save_sample(sample_folder, epoch, img, mask, first_out, second_out):
    masked_img = img * (1 - mask) + mask
    mask = torch.cat((mask, mask, mask), 1)
    img_list = [img, mask, masked_img, first_out, second_out]
    name_list = ['gt', 'mask', 'masked_img', 'first_out', 'second_out']
    save_sample_png(sample_folder=sample_folder, sample_name='epoch%d' % (epoch + 1), img_list=img_list,
                              name_list=name_list, pixel_max_cnt=255)


def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim
