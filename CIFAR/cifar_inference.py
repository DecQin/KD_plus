# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from Models import resnet56_cifar   # 你的模型定义

# # ---------------- 参数 ----------------
# TEACHER_PTH = './ckpt/cifar_teachers/resnet56_vanilla/resnet56.pth'   # 改成本地路径
# BATCH_SIZE  = 128
# NUM_CLASS   = 100
# DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# # ---------------- 数据 ----------------
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5071, 0.4867, 0.4408),
#                          (0.2675, 0.2565, 0.2761))
# ])
# test_set = datasets.CIFAR100(root='./Dataset/data', train=False,
#                              download=True, transform=transform)
# loader   = DataLoader(test_set, batch_size=BATCH_SIZE,
#                       shuffle=False, num_workers=4, pin_memory=True)

# # ---------------- 模型 ----------------
# model = resnet56_cifar().to(DEVICE)
# ckpt  = torch.load(TEACHER_PTH, map_location=DEVICE)
# # 兼容 DataParallel / 单卡
# if 'module.' in list(ckpt['model'].keys())[0]:
#     from collections import OrderedDict
#     new_state = OrderedDict((k[7:], v) for k, v in ckpt['model'].items())
#     model.load_state_dict(new_state)
# else:
#     model.load_state_dict(ckpt['model'])
# model.eval()

# # ---------------- 推理 ----------------
# correct = total = 0
# with torch.no_grad():
#     for x, y in loader:
#         x, y = x.to(DEVICE), y.to(DEVICE)
#         out = model(x)
#         logits = out[0] if isinstance(out, tuple) else out
#         pred = logits.argmax(1)
#         total += y.size(0)
#         correct += (pred == y).sum().item()

# acc = 100. * correct / total
# print(f'ResNet56 on CIFAR-100: {acc:.2f}%  (correct/total = {correct}/{total})')

import torch, torchvision.transforms as T
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from resnet import resnet56          # 确保结构一致

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 数据
mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]
transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
test_set  = CIFAR100('./Dataset/data', train=False, transform=transform)
loader    = DataLoader(test_set, 128, shuffle=False, num_workers=4)

# 模型
net = resnet56(num_classes=100).to(device)
ckpt = torch.load('./ckpt/cifar_teachers/resnet56_vanilla/resnet56.pth', map_location=device)
net.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
net.eval()

# 推理
correct = total = 0
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = net(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

print(f'Acc = {100*correct/total:.2f}%  ({correct}/{total})')