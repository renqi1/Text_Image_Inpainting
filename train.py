import argparse

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--train_model', type = str, default = 'deepfillv2', help='which model you want to train')
    parser.add_argument('--save_path', type = str, default = './pytorch_model', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0, 1", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--deepfillv2_generator_pretrain', type = str, default = '', help = 'load generator pretrain model')
    parser.add_argument('--deepfillv2_discriminator_pretrain', type = str, default = '', help='load discriminator pretrain odel')
    parser.add_argument('--deeplabv3_pretrain', type = str, default = '', help='load segmentnet pretrain model')
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 3, help = 'number of epochs of training')
    parser.add_argument('--batch_size1', type = int, default = 24, help = 'size of the deepfillv2 batches, if you use autocast you can set a larger batch size')
    parser.add_argument('--batch_size2', type=int, default=16, help='size of the deeplabv3 batches')
    parser.add_argument('--autocast', type=bool, default=True, help='Semi precision training')
    parser.add_argument('--lr_s', type = float, default = 1e-4, help= 'Adam: SegmentNet learning rate')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: Generator learning rate')
    parser.add_argument('--lr_d', type = float, default = 4e-4, help = 'Adam: Discriminator learning rate')
    parser.add_argument('--b1', type = float, default = 0.5, help = 'Adam: deepfillv2 beta 1')
    parser.add_argument('--b2', type = float, default = 0.999, help = 'Adam: deepfillv2 beta 2')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.98, help = 'lr decrease factor')
    parser.add_argument('--lr_decrease_step', type=int, default=100, help='lr decrease step for deepfiilv2')
    parser.add_argument('--lambda_l1', type = float, default = 1000, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_gan', type = float, default = 1, help = 'the parameter of gan loss')
    parser.add_argument('--num_workers', type = int, default = 8, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--backbone', type = str, default = 'mobilenetv2', help = 'segmentnet, deeplabv3 backbone')
    parser.add_argument('--in_c', type = int, default = 3, help = 'segmentnet, input RGB image')
    parser.add_argument('--out_c', type = int, default = 2, help = 'segmentnet, output segment foreground and background')
    parser.add_argument('--in_channels', type = int, default = 4, help = 'deepfillv2, input RGB image + 1 channel mask')
    parser.add_argument('--out_channels', type = int, default = 3, help = 'deepfillv2, output RGB image')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'deepfillv2, latent channels')
    parser.add_argument('--pad_type', type = str, default = 'zero', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--init_type', type = str, default = 'xavier', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = '/home/gzz/机械硬盘/sda3/text_image_inpainting/train', help = 'the training folder')
    parser.add_argument('--augment_data_path', type = str, default = '/home/gzz/机械硬盘/sda3/text_image_inpainting/train/train_data_6/gt_crop', help = 'the augment data folder')
    parser.add_argument('--augment', type=bool, default= False, help='augment dataset')
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--imgsize', type = int, default = 1536, help = 'size of image')
    parser.add_argument('--imgresize', type=int, default = 6, help='image resize ratio')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--mask_num', type = int, default = 15, help = 'number of mask')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    parser.add_argument('--max_angle', type = int, default = 4, help = 'parameter of angle for free form mask')
    parser.add_argument('--max_len', type = int, default = 50, help = 'parameter of length for free form mask')
    parser.add_argument('--max_width', type = int, default = 15, help = 'parameter of width for free form mask')
    opt = parser.parse_args()
    print(opt)

    
    '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
    import trainer
    # Enter main function
    if opt.train_model == 'deeplabv3':
        trainer.Segment_trainer(opt)

    if opt.train_model == 'deepfillv2':
        if opt.gan_type == 'WGAN':
            trainer.WGAN_trainer(opt)
        if opt.gan_type == 'LSGAN':
            trainer.LSGAN_trainer(opt)
