import torch

ckpt = torch.load('./ckpt/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth', map_location='cpu')
print(list(ckpt.keys()))