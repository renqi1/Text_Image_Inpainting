# 百度网盘AI大赛——图像处理挑战赛:文档图片去遮挡

[比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/479/0/introduction)

## 1 方案

图像分割+图像修复

有空再细说

##  2 代码说明

### 2.1 Train Deeplabv3

首先训练图像分割网络, 骨干网络有resnet50, resnet101, mobilenetv2可选，
mobilenetv2和resnet50效果都差不多，mobilenetv2参数量少，但GPU推理慢，resnet50参数大反而推理快一些。
权重衰减为每100个step，lr = 0.95*lr, batch size为16， trainloader长度为900. 训练2个epoch基本就差不多了。


```bash
python train.py     --train_model 'deeplabv3'
                    --epochs 2
                    --batch_size2 16
                    --lr_s 0.0001
                    --lr_decrease_factor 0.95
                    --lr_decrease_step 100
                    --backbone 'resnet50'
                    --baseroot ''
```
### 2.2 Train Deepfillv2

再训练deepfillv2, 生成器和判别器默认原项目作者的学习率效果较好，权重衰减为每100个step，lr = 0.95*lr, 
采用混合精度训练，batch size为增大至24，trainloader长度为600. L1损失系数太小没有效果，因此设置为1000.
图像imgsize是在原图裁剪的原始图像大小，训练时需要resize，这里裁剪图像尺寸1536×1536，缩放为1/6即256×256较为合理。

```bash
python train.py     --train_model 'deepfillv2'
                    --lr_g 0.0001
                    --lr_d 0.0004
                    --lr_decrease_factor 0.95
                    --lr_decrease_step 100
                    --lambda_l1 1000
                    --lambda_gan 1
                    --autocast True
                    --batch_size1 24
                    --imgsize 1536
                    --imgresize 6
```
```bash
if you have more than one GPU, please change following codes:
python train.py     --multi_gpu True
                    --gpu_ids [the ids of your multi-GPUs]
```

### 2.3 预测
#### 2.3.1 预测单张图片查看效果
运行predict_single.py可得到预测结果对比图，分别为：图像分割图，粗修图，精修图。
你需要设置你训练的权重文件路径，注意imgsize, imgresize, backbone都要与训练时一样。
```bash
python predict_single.py    --image_path 'test_img/test.jpg'
                            --imgsize 1536
                            --imgresize 6
                            --deepfillv2_generator_pretrain './models/deepfillv2_G.pth'
                            --deeplabv3_pretrain  './models/deeplabv3_resnet50.pth'
                            --backbone 'resnet50'
```
### 2.3.2 导出onnx并预测多张图片
运行convert导出onnx模型，你需要设置你的pytorch模型路径，以及onnx模型保存的路径(和名字)
```bash
python convert.py   --deepfillv2_generator_pretrain './models/deepfillv2_G.pth'
                    --deeplabv3_pretrain  './models/deeplabv3_resnet50.pth'
                    --backbone 'resnet50'
                    --save_path1 './onnx_model/deeplabv3_resnet50.onnx'
                    --save_path2 './onnx_model/deepfillv2_G.onnx'
```
运行predict_onnx.py预测图片，你需要设置要预测的图片文件夹路径，以及onnx模型保存的路径(和名字)，注意imgsize, imgresize要与前面一样。
```bash
python predict_onnx.py   --src_image_dir 'test_img'
                    --save_dir test_img_predict
                    --imgsize 1536
                    --imgresize 6
                    --deeplabv3_onnx './onnx_model/deeplabv3_resnet50.onnx'
                    --deepfillv2_onnx './onnx_model/deepfillv2_G.onnx'
```

## 3 代码运行环境
* GPU RTX3090
* ubuntu 20.04.1
* cuda == 11.3
* python == 3.8.13 
* pytorch == 1.10.1 
* torchvision == 0.11.2
* cv2 == 4.5.5
* transformers==4.21.1   
* numpy==1.22.4