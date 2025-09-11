import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from resnet import resnet56           # 与你生成 .pth 时用的定义一致

# 1. 数据
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4867,0.4408),
                         (0.2675,0.2565,0.2761))
])
train_set = datasets.CIFAR100('./Dataset/data', train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR100('./Dataset/data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, 128, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_set,  128, shuffle=False, num_workers=4)

# 2. 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = resnet56(num_classes=100).to(device)

# 3. 加载已有权重
ckpt = torch.load('./ckpt/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth', map_location=device)

# 取真正的权重字典
state_dict = ckpt['model']          # 关键一步！

# 兼容 DataParallel（若保存时用了）
if list(state_dict.keys())[0].startswith('module.'):
    from collections import OrderedDict
    new_state = OrderedDict((k[7:], v) for k, v in state_dict.items())
    state_dict = new_state

# 4. 加载
net.load_state_dict(state_dict, strict=True)
print('Loaded successfully. Acc =', ckpt.get('accuracy', 'N/A'))


# 4. 训练 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1,
                            momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[100,150],
                                                 gamma=0.1)

for epoch in range(200):          # 再跑 50 epoch
    net.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(net(x), y)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # 快速验证
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (net(x).argmax(1) == y).sum().item()
            total   += y.size(0)
            val_acc = 100*correct/total
    print(f'Epoch {epoch} Acc {val_acc:.2f}%')

    best_acc = 0.0
    best_path = "./ckpt/cifar_teachers/resnet56_vanilla/resnet56.pth"
    # 更新并保存
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch'      : epoch + 1,
            'model'      : net.state_dict(),
            'optimizer'  : optimizer.state_dict(),
            'scheduler'  : scheduler.state_dict(),
            'accuracy'   : best_acc
        }, best_path)
        print(f"保存新最佳权重：{best_acc:.2f}%  -> {best_path}")

# 5. 保存继续训练后的权重
# torch.save(net.state_dict(), './ckpt/cifar_teachers/resnet56_vanilla/resnet56.pth')