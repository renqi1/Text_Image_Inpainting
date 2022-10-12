import os
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import dataset
import utils
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.FillDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))
    if opt.augment:
        augmentset = dataset.InpaintDataset(opt)
        trainset = ConcatDataset([trainset, augmentset])

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size1, shuffle = True, num_workers = opt.num_workers, pin_memory = True)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, step, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor"""
        lr = lr_in * (opt.lr_decrease_factor ** step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    # Save the onnx_model
    def save_model(generator, discriminator, epoch, opt):
        model_g = 'deepfillv2_G_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size1)
        model_d = 'deepfillv2_D_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size1)
        model_g = os.path.join(save_folder, model_g)
        model_d = os.path.join(save_folder, model_d)
        if opt.multi_gpu == True:
            torch.save(generator.module.state_dict(), model_g)
            torch.save(discriminator.module.state_dict(), model_d)
            print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            torch.save(generator.state_dict(), model_g)
            torch.save(discriminator.state_dict(), model_d)
            print('The trained model is successfully saved at epoch %d' % (epoch))

    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()
    step = 0
    scaler = GradScaler()
    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img = img.cuda()
            mask = mask.cuda()
            optimizer_d.zero_grad()
            optimizer_g.zero_grad()
            if opt.autocast:
                with autocast():
                    ## Train Discriminator
                    first_out, second_out = generator(img, mask)
                    first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
                    second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]
                    # Fake samples
                    fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
                    # True samples
                    true_scalar = discriminator(img, mask)
                    # Overall Loss and optimize
                    loss_D = 0.5 * torch.mean(torch.relu(1 - true_scalar)) + 0.5 * torch.mean(torch.relu(1 + fake_scalar))
                    # loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
                    scaler.scale(loss_D).backward()
                    scaler.step(optimizer_d)
                    scaler.update()
                    ## Train Generator
                    # Mask L1 Loss
                    first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
                    second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
                    # GAN Loss
                    fake_scalar = discriminator(second_out_wholeimg, mask)
                    GAN_Loss = -torch.mean(fake_scalar)
                    # Compute losses
                    loss = opt.lambda_l1 * (second_MaskL1Loss + 0.5*first_MaskL1Loss) + opt.lambda_gan * GAN_Loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_g)
                    scaler.update()
            else:
                ## Train Discriminator
                first_out, second_out = generator(img, mask)
                first_out_wholeimg = img * (1 - mask) + first_out * mask  # in range [0, 1]
                second_out_wholeimg = img * (1 - mask) + second_out * mask  # in range [0, 1]
                # Fake samples
                fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
                # True samples
                true_scalar = discriminator(img, mask)
                # Overall Loss and optimize
                loss_D = 0.5 * torch.mean(torch.relu(1 - true_scalar)) + 0.5 * torch.mean(torch.relu(1 + fake_scalar))
                # loss_D = - torch.mean(true_scalar) + torch.mean(fake_scalar)
                loss_D.backward()
                optimizer_d.step()
                ## Train Generator
                # Mask L1 Loss
                first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
                second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
                # GAN Loss
                fake_scalar = discriminator(second_out_wholeimg, mask)
                GAN_Loss = -torch.mean(fake_scalar)
                # Compute losses
                loss = opt.lambda_l1 * (second_MaskL1Loss + 0.5*first_MaskL1Loss) + opt.lambda_gan * GAN_Loss
                loss.backward()
                optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if batch_idx % 5 == 0:
            # Print log
                print("\r[Epoch %d/%d] [Batch %d/%d] L1Loss: %.5f  %.5f  DLoss: %.5f GLoss: %.5f time_left: %s" %
                    ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(), second_MaskL1Loss.item(),
                    loss_D.item(), GAN_Loss.item(), time_left))

                # print("\r[Epoch %d/%d] [Batch %d/%d] L1Loss: %.5f  %.5f  DLoss: %.5f GLoss: %.5f time_left: %s" %
                #       ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(),
                #        second_MaskL1Loss.item(),
                #        loss_D.item(), GAN_Loss.item(), time_left))
            if (batch_idx+1) % opt.lr_decrease_step == 0:
                # Save the onnx_model
                step += 1
                save_model(generator, discriminator, (epoch + 1), opt)

                utils.save_sample(sample_folder, epoch, img, mask, first_out, second_out)

                # Learning rate decrease
                adjust_learning_rate(opt.lr_g, optimizer_g, step, opt)
                adjust_learning_rate(opt.lr_d, optimizer_d, step, opt)



def Segment_trainer(opt):

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # configurations
    save_folder = opt.save_path

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Define the dataset
    trainset = dataset.SegmentDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size=opt.batch_size2, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    # Build networks
    segmentnet = utils.create_segmentnet(opt)

    # To device
    if opt.multi_gpu == True:
        segmentnet = nn.DataParallel(segmentnet)
        segmentnet = segmentnet.cuda()

    else:
        segmentnet = segmentnet.cuda()

    # Optimizers
    optimizer = torch.optim.Adam(segmentnet.parameters(), lr=opt.lr_s)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, step, opt):
        """Set the learning rate every step"""
        lr = opt.lr_s * (opt.lr_decrease_factor ** step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    # Save the onnx_model
    def save_model(net, epoch, opt):
        model_name = 'deeplabv3_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size2)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            torch.save(net.module.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            torch.save(net.state_dict(), model_name)
            print('The trained model is successfully saved at epoch %d' % (epoch))

    # ----------------------------------------
    #            Training
    # ----------------------------------------
    prev_time = time.time()
    weights = [1, 10]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    step = 0
    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (img, mask) in enumerate(dataloader):
            # img (shape: [B, 3, H, W]), masked (shape: [B, H, W]) and put it to cuda
            img = img.cuda()
            mask = torch.tensor(mask, dtype=torch.int64).cuda()

            ## Train SegmentNet
            optimizer.zero_grad()
            segment = segmentnet(img)
            loss = loss_fn(segment, mask)
            loss.backward()
            optimizer.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            if batch_idx % 5 == 0:
                # Print log
                print("\r[Epoch %d/%d] [Batch %d/%d]   Loss: %.5f   time_left: %s" %
                      ((epoch + 1), opt.epochs, batch_idx, len(dataloader), loss.item(), time_left))

                # print("\r[Epoch %d/%d] [Batch %d/%d] L1Loss: %.5f  %.5f  DLoss: %.5f GLoss: %.5f time_left: %s" %
                #       ((epoch + 1), opt.epochs, batch_idx, len(dataloader), first_MaskL1Loss.item(),
                #        second_MaskL1Loss.item(),
                #        loss_D.item(), GAN_Loss.item(), time_left))
            if (batch_idx + 1) % opt.lr_decrease_step == 0:
                step += 1
                # Save the onnx_model
                save_model(segmentnet, (epoch + 1), opt)
                # Learning rate decrease
                adjust_learning_rate(optimizer, step, opt)

        save_model(segmentnet, (epoch + 1), opt)

