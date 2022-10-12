import torch
import numpy as np
import argparse
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deeplabv3_pretrain', type = str, default = './pytorch_model/deeplabv3_resnet50.pth')
    parser.add_argument('--deepfillv2_generator_pretrain', type = str, default = './pytorch_model/deepfillv2_G.pth')
    parser.add_argument('--save_path1', type = str, default = './onnx_model/deeplabv3_resnet50.onnx')
    parser.add_argument('--save_path2', type = str, default = './onnx_model/deepfillv2_G.onnx')
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

    # 构建输入
    input = np.random.rand(1, 3, 640, 480).astype("float32")
    input = torch.tensor(input)
    # 获取PyTorch Module
    segmentnet = utils.create_segmentnet(opt)
    segmentnet.eval()
    # onnx
    torch.onnx.export(segmentnet, input, opt.save_path1, opset_version=11, verbose=True, input_names=['input'], output_names=['output'])

    print('segmentnet export successfully')

    deepfillv2 = utils.create_generator(opt)
    deepfillv2.eval()
    input1 = np.random.rand(1, 3, 256, 256).astype("float32")
    input2 = np.random.rand(1, 1, 256, 256).astype("float32")

    input = (torch.tensor(input1), torch.tensor(input2))
    input_names = ('img', 'mask')
    output_names = ["first_out", "second_out"]
    torch.onnx.export(deepfillv2, input, opt.save_path2, opset_version=11, verbose=True, input_names=input_names, output_names=output_names)
    print('fillnet export successfully')