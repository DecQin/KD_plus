# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# import torch
# import torchvision
# from torch import nn
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn

# import MODELS
# from MODELS.rekdr50 import build_review_kd, hcl
# from Dataset import CIFAR
# from utils import colorstr, Save_Checkpoint, AverageMeter, DirectNormLoss
# from utils_plus2_rekdr50 import CKDConfig, CKDLoss, CanonicalProj
# # from torchsummaryX import summary

# import numpy as np
# from pathlib import Path
# import os
# import time
# import json
# import random
# import logging
# import argparse
# import warnings
# # from torch.utils.tensorboard import SummaryWriter
# import pdb
# import wandb
# # add
# from collections import OrderedDict


# def train(model, teacher, T_EMB, train_dataloader, optimizer, criterion, scheduler, args, epoch):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     train_loss = AverageMeter()
#     train_error = AverageMeter()

#     # Cls_loss = AverageMeter()
#     # Div_loss = AverageMeter()
#     # Norm_Dir_loss = AverageMeter()

#     # Model on train mode
#     model.train()
#     teacher.eval()
#     step_per_epoch = len(train_dataloader)
    
#     # pdb.set_trace()
#     for step, (images, labels) in enumerate(train_dataloader):
#         start = time.time()
#         images, labels = images.cuda(), labels.cuda()

#         # compute output
#         s_features, s_emb, s_logits = model(images)

#         with torch.no_grad():
#             t_features, t_emb, t_logits = teacher(images, is_feat=True, preact=True)
#             # t_features, t_emb, t_logits = teacher(images, is_feat=True, preact=False)
#             t_features = t_features[1:]
        
#         # cls loss
#         # cls_loss = criterion(s_logits, labels) * args.cls_loss_factor
#         # Kd loss
#         # kd_loss = hcl(s_features, t_features) * min(1, epoch/args.warm_up) * args.kd_loss_factor
#         # ND loss
#         # norm_dir_loss = nd_loss(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels)

#         # loss = cls_loss + kd_loss + norm_dir_loss
#         cfg = CKDConfig(
#                     lam_kd=args.lam_kd,     
#                     sum_lam=args.sum_lam,   
#                     lam_mmd=args.lam_mmd,
#         )
#         criterion = CKDLoss(cfg)

#         # norm 
#         Ds = s_emb.shape[1]
#         Dt = t_emb.shape[1]
#         proj_s = CanonicalProj(Ds, args.Dc).to(device)
#         proj_t = CanonicalProj(Dt, args.Dc).to(device)

#         # 统一 loss
#         loss_dict = criterion(
#                         logits_s=s_logits,
#                         labels=labels,
#                         logits_t=t_logits,
#                         feat_s=s_emb,
#                         feat_t=t_emb,
#                         tz_s=s_features,
#                         tz_t=t_features,
#                         proj_s=proj_s,
#                         proj_t=proj_t,
#                         epoch=epoch,
#         )
#         loss = loss_dict['loss_total']
#         # measure accuracy and record loss
#         batch_size = images.size(0)
#         _, pred = s_logits.data.cpu().topk(1, dim=1)
#         train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
#         train_loss.update(loss.item(), batch_size)

#         # Cls_loss.update(cls_loss.item(), batch_size)
#         # Div_loss.update(kd_loss.item(), batch_size)
#         # Norm_Dir_loss.update(norm_dir_loss.item(), batch_size)

#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
#         s1 = '\r{} [{}/{}]'.format(t, step+1, step_per_epoch)
#         s2 = ' - {:.2f}ms/step - ce_loss: {:.3f} - kd_loss: {:.3f} - nd_loss: {:.3f} - mmd_loss: {:.3f} - orth_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}'.format(
#              1000 * (time.time() - start), loss_dict.get('loss_sup').item(), loss_dict.get('loss_kd').item(), loss_dict.get('loss_nd').item(), loss_dict.get('loss_mmd').item(), loss_dict.get('loss_orth').item(), train_loss.val, 1-train_error.val)

#         print(s1+s2, end='', flush=True)
#         scheduler.step()

#     print()

#     # 直接取最后一个 batch 的值（tensorboard 用）
#     sup = loss_dict.get('loss_sup', torch.tensor(0.)).item()
#     kd = loss_dict.get('loss_kd',  torch.tensor(0.)).item()
#     nd = loss_dict.get('loss_nd', torch.tensor(0.)).item()
#     mmd = loss_dict.get('loss_mmd', torch.tensor(0.)).item()
#     orth = loss_dict.get('loss_orth', torch.tensor(0.)).item()

#     return sup, kd, nd, mmd, orth, train_loss.avg, train_error.avg


# def test(model, test_dataloader, criterion):
#     test_loss = AverageMeter()
#     test_error = AverageMeter()

#     # Model on eval mode
#     model.eval()

#     with torch.no_grad():
#         for images, labels in test_dataloader:
#             images, labels = images.cuda(), labels.cuda()

#             # compute logits
#             _, _, logits = model(images)

#             loss = F.cross_entropy(logits, labels)
#             # loss = criterion(logits, labels)

#             # measure accuracy and record loss
#             batch_size = images.size(0)
#             _, pred = logits.data.cpu().topk(1, dim=1)
#             test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
#             test_loss.update(loss.item(), batch_size)

#     return test_loss.avg, test_error.avg


# def epoch_loop(model, teacher, train_set, test_set, args):
#     # 初始化 wandb 实验
#     wandb.init(
#         project="res50_rekd+++",  # 必选：你的项目名称（可自定义）
#         name=f"{args.teacher}-{args.model_name}",  # 可选：实验名称（便于区分）
#         config=vars(args),  # 自动记录所有命令行参数（args）
#         save_code=True,  # 可选：保存当前代码快照（便于复现）
#         dir=str("./run")  # 可选：wandb 日志文件保存路径（与你的 save_dir 一致）
#     )

#     # data loaders
#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.workers)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)
    
#     # model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = nn.DataParallel(model, device_ids=args.gpus)
#     model.to(device)
#     teacher = nn.DataParallel(teacher, device_ids=args.gpus)
#     teacher.to(device)

#     # loss
#     # criterion = nn.CrossEntropyLoss().to(device)
#     # nd_loss = DirectNormLoss(num_class=100, nd_loss_factor=args.nd_loss_factor).to(device)

#     cfg = CKDConfig()
#     criterion = CKDLoss(cfg).to(device)

#     # optimizer
#     optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1 * args.batch_size / 256, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

#     # CosineAnnealingLR
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer=optimizer,
#         T_max=args.epochs*len(train_loader),  # 总epoch数作为周期
#     )
    
#     # 权重
#     save_dir = Path(args.save_dir)
#     weights = save_dir / 'weights'
#     weights.mkdir(parents=True, exist_ok=True)
#     last = weights / 'last'
#     best = weights / 'best'

#     # acc,loss
#     acc_loss = save_dir / 'acc_loss'
#     acc_loss.mkdir(parents=True, exist_ok=True)

#     train_acc_savepath = acc_loss / 'train_acc.npy'
#     train_loss_savepath = acc_loss / 'train_loss.npy'
#     val_acc_savepath = acc_loss / 'val_acc.npy'
#     val_loss_savepath = acc_loss / 'val_loss.npy'

#     # tensorboard
#     # logdir = save_dir / 'logs'
#     # logdir.mkdir(parents=True, exist_ok=True)
#     # summary_writer = SummaryWriter(logdir, flush_secs=120)
    
#     # resume
#     if args.resume:
#         checkpoint = torch.load(args.resume)
#         start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         best_error = checkpoint['best_error']
#         train_acc = checkpoint['train_acc']
#         train_loss = checkpoint['train_loss']
#         test_acc = checkpoint['test_acc']
#         test_loss = checkpoint['test_loss']
#         logger.info(colorstr('green', 'Resuming training from {} epoch'.format(start_epoch)))
#     else:
#         start_epoch = 0
#         best_error = 0
#         train_acc = []
#         train_loss = []
#         test_acc = []
#         test_loss = []

#     # Train model
#     best_error = 1
#     for epoch in range(start_epoch, args.epochs):
#         # if epoch in [150, 180, 210]:
#         #     for param_group in optimizer.param_groups:
#         #         param_group['lr'] *= 0.1
#         print("Epoch {}/{}".format(epoch + 1, args.epochs))
#         sup_loss, kd_loss, nd_loss, mmd_loss, orth_loss, train_epoch_loss, train_error = train(model=model,
#                                                                                  teacher=teacher,
#                                                                                  T_EMB=T_EMB,
#                                                                                  train_dataloader=train_loader,
#                                                                                  optimizer=optimizer,
#                                                                                  criterion=criterion,
#                                                                                  scheduler=scheduler,
#                                                                                  args=args,
#                                                                                  epoch=epoch)
#         test_epoch_loss, test_error = test(model=model, 
#                                            test_dataloader=test_loader,
#                                            criterion=criterion)
        
#         s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.5f}".format(
#             train_epoch_loss, 1-train_error, test_epoch_loss, 1-test_error, optimizer.param_groups[0]['lr'])
#         logger.info(colorstr('green', s))

#         # save acc,loss
#         train_loss.append(train_epoch_loss)
#         train_acc.append(1-train_error)
#         test_loss.append(test_epoch_loss)
#         test_acc.append(1-test_error)

#         # save model
#         is_best = test_error < best_error
#         best_error = min(best_error, test_error)
#         state = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_error': best_error,
#                 'train_acc': train_acc,
#                 'train_loss': train_loss,
#                 'test_acc': test_acc,
#                 'test_loss': test_loss,
#             }
        
#         last_path = last / 'epoch_{}_loss_{:.3f}_acc_{:.3f}'.format(
#             epoch + 1, test_epoch_loss, 1-test_error)
#         best_path = best / 'epoch_{}_acc_{:.3f}'.format(
#                 epoch + 1, 1-best_error)

#         Save_Checkpoint(state, last, last_path, best, best_path, is_best)

#         wandb.log({
#             # 学习率
#             "lr": optimizer.param_groups[0]['lr'],
#             # 训练指标
#             "train_acc": 1 - train_error,
#             "train_loss": train_epoch_loss,
#             "train_error": train_error,
#             # 验证指标
#             "test_acc": 1 - test_error,
#             "val_loss": test_epoch_loss,
#             "val_error": test_error,
#             "best_test_acc": 1 - best_error,
#             # 细分损失
#             "sup_loss": sup_loss,
#             "kd_loss": kd_loss,
#             "nd_loss": nd_loss,
#             "mmd_loss": mmd_loss,
#             "orth_loss": orth_loss,
#         }, step=epoch)  # step 对应 epoch，确保曲线按 epoch 排列

#     # 新增：跟踪保存的模型文件（last 和 best 模型都记录）
#     wandb.save(str(last_path))  # 跟踪最新模型
#     if is_best:
#         wandb.save(str(best_path))  # 仅跟踪最优模型

#     wandb.finish()  # 关闭 wandb 实验，确保日志完整保存

#         # tensorboard
#         # pdb.set_trace()
#     #     if epoch == 1:
#     #         images, labels = next(iter(train_loader))
#     #         img_grid = torchvision.utils.make_grid(images)
#     #         summary_writer.add_image('Cifar Image', img_grid)
#     #     summary_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
#     #     summary_writer.add_scalar('train_loss', train_epoch_loss, epoch)
#     #     summary_writer.add_scalar('train_error', train_error, epoch)
#     #     summary_writer.add_scalar('val_loss', test_epoch_loss, epoch)
#     #     summary_writer.add_scalar('val_error', test_error, epoch)

#     #     summary_writer.add_scalar('nd_loss', norm_dir_loss, epoch)
#     #     summary_writer.add_scalar('kd_loss', div_loss, epoch)
#     #     summary_writer.add_scalar('cls_loss', cls_loss, epoch)

#     # summary_writer.close()
#     if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
#         np.save(train_acc_savepath, train_acc)
#         np.save(train_loss_savepath, train_loss)
#         np.save(val_acc_savepath, test_acc)
#         np.save(val_loss_savepath, test_loss)


# if __name__ == "__main__":
#     model_names = sorted(name for name in MODELS.__dict__ 
#                          if name.islower() and not name.startswith("__") 
#                          and callable(MODELS.__dict__[name]))

#     parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
#     parser.add_argument("--model_name", type=str, default="resnet20_cifar", choices=model_names, help="model architecture")
#     parser.add_argument("--dataset", type=str, default='cifar100')
#     parser.add_argument("--epochs", type=int, default=240)
#     parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
#     parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
#     parser.add_argument("--lr", type=float, default=0.1)
#     parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
#     parser.add_argument("--weight_decay", type=float, default=5e-4)

#     parser.add_argument("--teacher", type=str, default="resnet50_cifar", help="teacher architecture")
#     parser.add_argument("--teacher_weights", type=str, default="", help="teacher weights path")
#     parser.add_argument("--cls_loss_factor", type=float, default=1.0, help="cls loss weight factor")
#     parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="KL loss weight factor")
#     parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="ND loss weight factor")
#     parser.add_argument("--warm_up", type=float, default=20.0, help='loss weight warm up epochs')
#     parser.add_argument("--Dc", type=int, default=256, help="Dc")
#     parser.add_argument("--lam_kd", type=float, default=1.0, help="lam_kd")
#     parser.add_argument("--sum_lam", type=float, default=0.5, help="sum_lam")
#     parser.add_argument("--lam_mmd", type=float, default=0.05, help="lam_mmd")

#     parser.add_argument("--gpus", type=list, default=[0,1])
#     parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
#     parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
#     parser.add_argument("--save_dir", type=str, default="./run", help="save path, eg, acc_loss, weights, tensorboard, and so on")
#     args = parser.parse_args()

#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s [line:%(lineno)d] %(message)s',
#                         datefmt='%d %b %Y %H:%M:%S')
#     logger = logging.getLogger(__name__)

#     args.batch_size = args.batch_size * len(args.gpus)

#     logger.info(colorstr('green', "Distribute train, gpus:{}, total batch size:{}, epoch:{}".format(args.gpus, args.batch_size, args.epochs)))
    
#     train_set, test_set, num_class = CIFAR(name=args.dataset)
#     model = build_review_kd(student_name=args.model_name, num_class=num_class, teacher_name=args.teacher)
#     teacher = MODELS.__dict__[args.teacher](num_class=num_class)

#     if args.teacher_weights:
#         print('Load Teacher Weights')
#         teacher_ckpt = torch.load(args.teacher_weights, weights_only=False, map_location='cpu')['model']

#         # wrn40_2
#         # new_state_dict = OrderedDict(
#         #     (k.replace('bn1.', 'bn.', 1) if k.startswith('bn1.') else k, v)
#         #     for k, v in teacher_ckpt.items()
#         # )
#         # teacher.load_state_dict(new_state_dict)
#         teacher.load_state_dict(teacher_ckpt)

#         for param in teacher.parameters():
#             param.requires_grad = False

#     # res56    ./ckpt/teacher/resnet56/center_emb_train.json
#     # res32x4  ./ckpt/teacher/resnet32x4/center_emb_train.json
#     # wrn40_2  ./ckpt/teacher/wrn_40_2/center_emb_train.json
#     # res50    ./ckpt/teacher/resnet50/center_emb_train.json
#     # class-mean
#     with open("./ckpt/teacher/resnet50/center_emb_train.json", 'r') as f:
#         T_EMB = json.load(f)
#     f.close()
    
#     logger.info(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.model_name + ' ...'))
#     # Train the model
#     epoch_loop(model=model, teacher=teacher, train_set=train_set, test_set=test_set, args=args)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import MODELS
from MODELS.rekdr50 import build_review_kd  # ✅ 不再从这里 import hcl（避免误用每步new conv的版本）
from Dataset import CIFAR
from utils import colorstr, Save_Checkpoint, AverageMeter, DirectNormLoss
from utils_plus2_rekdr50 import CKDConfig, CKDLoss, CanonicalProj

import numpy as np
from pathlib import Path
import os
import time
import json
import random
import logging
import argparse
import warnings
import pdb
import wandb
from collections import OrderedDict
import inspect


# ✅ 如果你的 MODELS.rekdr50 里已经实现了 HCLAlign，可直接 from MODELS.rekdr50 import HCLAlign
#   否则先在这里给一个本地可用的 HCLAlign（只初始化一次，参数可训练，绝不在每步 new）
class HCLAlign(nn.Module):
    def __init__(self, s_channels, t_channels, pool_levels=(4, 2, 1)):
        super().__init__()
        assert len(s_channels) == len(t_channels), "HCLAlign: s_channels/t_channels 长度必须一致"
        self.pool_levels = pool_levels
        self.adapt = nn.ModuleList()
        for sc, tc in zip(s_channels, t_channels):
            if sc == tc:
                self.adapt.append(nn.Identity())
            else:
                self.adapt.append(nn.Conv2d(sc, tc, kernel_size=1, bias=False))

    @torch.no_grad()
    def _align_spatial(self, fs, ft):
        if fs.shape[-2:] == ft.shape[-2:]:
            return fs, ft
        hs, ws = fs.shape[-2:]
        ht, wt = ft.shape[-2:]
        area_s = hs * ws
        area_t = ht * wt
        # 只做下采样（更大的一侧 pool 到更小的一侧）
        if area_s > area_t:
            fs = F.adaptive_avg_pool2d(fs, (ht, wt))
        else:
            ft = F.adaptive_avg_pool2d(ft, (hs, ws))
        return fs, ft

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for i, (fs, ft) in enumerate(zip(fstudent, fteacher)):
            fs, ft = self._align_spatial(fs, ft)
            fs = self.adapt[i](fs)
            loss = F.mse_loss(fs, ft, reduction='mean')
            n, c, h, w = fs.shape
            cnt = 1.0
            tot = 1.0
            for l in self.pool_levels:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction='mean') * cnt
                tot += cnt
            loss_all = loss_all + loss / tot
        return loss_all


def train(model, teacher, train_dataloader, optimizer, criterion, scheduler,
          proj_s, proj_t, feat_len, args, epoch):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_loss = AverageMeter()
    train_error = AverageMeter()

    model.train()
    teacher.eval()
    step_per_epoch = len(train_dataloader)

    for step, (images, labels) in enumerate(train_dataloader):
        start = time.time()
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        # student
        s_features, s_emb, s_logits = model(images)

        # teacher
        with torch.no_grad():
            t_features, t_emb, t_logits = teacher(images, is_feat=True, preact=True)
            t_features = t_features[1:]

        # ✅ 固定对齐长度（只在 epoch_loop 里算一次）
        if feat_len is not None:
            s_features_use = s_features[:feat_len]
            t_features_use = t_features[:feat_len]
        else:
            L = min(len(s_features), len(t_features))
            s_features_use = s_features[:L]
            t_features_use = t_features[:L]

        # ✅ 统一 loss（不再每步 new CKDConfig / CKDLoss / CanonicalProj）
        loss_dict = criterion(
            logits_s=s_logits,
            labels=labels,
            logits_t=t_logits,
            feat_s=s_emb,
            feat_t=t_emb,
            tz_s=s_features_use,
            tz_t=t_features_use,
            proj_s=proj_s,
            proj_t=proj_t,
            epoch=epoch,
        )
        loss = loss_dict['loss_total']

        batch_size = images.size(0)
        _, pred = s_logits.data.cpu().topk(1, dim=1)
        train_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
        train_loss.update(loss.item(), batch_size)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        s1 = '\r{} [{}/{}]'.format(t, step + 1, step_per_epoch)
        s2 = (' - {:.2f}ms/step - ce_loss: {:.3f} - kd_loss: {:.3f} - nd_loss: {:.3f} - '
              'mmd_loss: {:.3f} - orth_loss: {:.3f} - train_loss: {:.3f} - train_acc: {:.3f}').format(
            1000 * (time.time() - start),
            loss_dict.get('loss_sup').item(),
            loss_dict.get('loss_kd').item(),
            loss_dict.get('loss_nd').item(),
            loss_dict.get('loss_mmd').item(),
            loss_dict.get('loss_orth').item(),
            train_loss.val,
            1 - train_error.val
        )
        print(s1 + s2, end='', flush=True)

        scheduler.step()

    print()

    sup = loss_dict.get('loss_sup', torch.tensor(0.)).item()
    kd = loss_dict.get('loss_kd', torch.tensor(0.)).item()
    nd = loss_dict.get('loss_nd', torch.tensor(0.)).item()
    mmd = loss_dict.get('loss_mmd', torch.tensor(0.)).item()
    orth = loss_dict.get('loss_orth', torch.tensor(0.)).item()

    return sup, kd, nd, mmd, orth, train_loss.avg, train_error.avg


def test(model, test_dataloader, criterion):
    test_loss = AverageMeter()
    test_error = AverageMeter()

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

            _, _, logits = model(images)
            loss = F.cross_entropy(logits, labels)

            batch_size = images.size(0)
            _, pred = logits.data.cpu().topk(1, dim=1)
            test_error.update(torch.ne(pred.squeeze(), labels.cpu()).float().sum().item() / batch_size, batch_size)
            test_loss.update(loss.item(), batch_size)

    return test_loss.avg, test_error.avg


def epoch_loop(model, teacher, train_set, test_set, args):
    wandb.init(
        project="res50_rekd+++",
        name=f"{args.teacher}-{args.model_name}",
        config=vars(args),
        save_code=True,
        dir=str("./run")
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=args.workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.workers)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    teacher = nn.DataParallel(teacher, device_ids=args.gpus).to(device)

    # ✅ 只创建一次 cfg/criterion（不在 train() 里反复 new）
    cfg = CKDConfig(
        lam_kd=args.lam_kd,
        sum_lam=args.sum_lam,
        lam_mmd=args.lam_mmd,
    )

    # ✅ 先“偷看”一个 batch 来初始化：HCLAlign + proj_s/proj_t（不会影响后续for-loop）
    images0, labels0 = next(iter(train_loader))
    images0 = images0.cuda(non_blocking=True)

    with torch.no_grad():
        s_features0, s_emb0, s_logits0 = model(images0)
        t_features0, t_emb0, t_logits0 = teacher(images0, is_feat=True, preact=True)
        t_features0 = t_features0[1:]

    feat_len = min(len(s_features0), len(t_features0))
    s_features0 = s_features0[:feat_len]
    t_features0 = t_features0[:feat_len]

    s_channels = [f.shape[1] for f in s_features0]
    t_channels = [f.shape[1] for f in t_features0]

    hcl_align = HCLAlign(s_channels, t_channels).to(device)

    # ✅ 兼容两种 CKDLoss 写法：__init__(cfg, hcl_align=...) 或 set_hcl_align(...)
    try:
        sig = inspect.signature(CKDLoss.__init__)
        if 'hcl_align' in sig.parameters:
            criterion = CKDLoss(cfg, hcl_align=hcl_align).to(device)
        else:
            criterion = CKDLoss(cfg).to(device)
            if hasattr(criterion, "set_hcl_align"):
                criterion.set_hcl_align(hcl_align)
    except Exception:
        criterion = CKDLoss(cfg).to(device)
        if hasattr(criterion, "set_hcl_align"):
            criterion.set_hcl_align(hcl_align)

    # ✅ proj 只创建一次（不在 train() 里反复 new）
    Ds = s_emb0.shape[1]
    Dt = t_emb0.shape[1]
    proj_s = CanonicalProj(Ds, args.Dc).to(device)
    proj_t = CanonicalProj(Dt, args.Dc).to(device)

    # ✅ optimizer：把 hcl_align + proj_s/proj_t 的参数也加进去（关键，否则映射永远学不会）
    extra_params = list(hcl_align.parameters()) + list(proj_s.parameters()) + list(proj_t.parameters())
    optimizer = torch.optim.SGD(
        params=list(model.parameters()) + extra_params,
        lr=args.lr * args.batch_size / 256,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs * len(train_loader),
    )

    save_dir = Path(args.save_dir)
    weights = save_dir / 'weights'
    weights.mkdir(parents=True, exist_ok=True)
    last = weights / 'last'
    best = weights / 'best'

    acc_loss = save_dir / 'acc_loss'
    acc_loss.mkdir(parents=True, exist_ok=True)
    train_acc_savepath = acc_loss / 'train_acc.npy'
    train_loss_savepath = acc_loss / 'train_loss.npy'
    val_acc_savepath = acc_loss / 'val_acc.npy'
    val_loss_savepath = acc_loss / 'val_loss.npy'

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # ✅ 兼容：旧ckpt没有这些key也不报错
        if 'proj_s_state_dict' in checkpoint:
            proj_s.load_state_dict(checkpoint['proj_s_state_dict'])
        if 'proj_t_state_dict' in checkpoint:
            proj_t.load_state_dict(checkpoint['proj_t_state_dict'])
        if 'hcl_align_state_dict' in checkpoint:
            hcl_align.load_state_dict(checkpoint['hcl_align_state_dict'])

        best_error = checkpoint.get('best_error', 1.0)
        train_acc = checkpoint.get('train_acc', [])
        train_loss = checkpoint.get('train_loss', [])
        test_acc = checkpoint.get('test_acc', [])
        test_loss = checkpoint.get('test_loss', [])
        logger.info(colorstr('green', 'Resuming training from {} epoch'.format(start_epoch)))
    else:
        start_epoch = 0
        best_error = 1.0
        train_acc, train_loss, test_acc, test_loss = [], [], [], []

    for epoch in range(start_epoch, args.epochs):
        print("Epoch {}/{}".format(epoch + 1, args.epochs))

        sup_loss, kd_loss, nd_loss, mmd_loss, orth_loss, train_epoch_loss, train_error = train(
            model=model,
            teacher=teacher,
            train_dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            proj_s=proj_s,
            proj_t=proj_t,
            feat_len=feat_len,
            args=args,
            epoch=epoch
        )

        test_epoch_loss, test_error = test(model=model, test_dataloader=test_loader, criterion=criterion)

        s = "Train Loss: {:.3f}, Train Acc: {:.3f}, Test Loss: {:.3f}, Test Acc: {:.3f}, lr: {:.5f}".format(
            train_epoch_loss, 1 - train_error, test_epoch_loss, 1 - test_error, optimizer.param_groups[0]['lr']
        )
        logger.info(colorstr('green', s))

        train_loss.append(train_epoch_loss)
        train_acc.append(1 - train_error)
        test_loss.append(test_epoch_loss)
        test_acc.append(1 - test_error)

        is_best = test_error < best_error
        best_error = min(best_error, test_error)

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_error': best_error,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
            # ✅ 新增保存（方便 resume 后继续稳定蒸馏）
            'proj_s_state_dict': proj_s.state_dict(),
            'proj_t_state_dict': proj_t.state_dict(),
            'hcl_align_state_dict': hcl_align.state_dict(),
            'feat_len': feat_len,
        }

        last_path = last / 'epoch_{}_loss_{:.3f}_acc_{:.3f}'.format(epoch + 1, test_epoch_loss, 1 - test_error)
        best_path = best / 'epoch_{}_acc_{:.3f}'.format(epoch + 1, 1 - best_error)

        Save_Checkpoint(state, last, last_path, best, best_path, is_best)

        wandb.log({
            "lr": optimizer.param_groups[0]['lr'],
            "train_acc": 1 - train_error,
            "train_loss": train_epoch_loss,
            "train_error": train_error,
            "test_acc": 1 - test_error,
            "val_loss": test_epoch_loss,
            "val_error": test_error,
            "best_test_acc": 1 - best_error,
            "sup_loss": sup_loss,
            "kd_loss": kd_loss,
            "nd_loss": nd_loss,
            "mmd_loss": mmd_loss,
            "orth_loss": orth_loss,
        }, step=epoch)

    wandb.save(str(last_path))
    if is_best:
        wandb.save(str(best_path))
    wandb.finish()

    if not os.path.exists(train_acc_savepath) or not os.path.exists(train_loss_savepath):
        np.save(train_acc_savepath, train_acc)
        np.save(train_loss_savepath, train_loss)
        np.save(val_acc_savepath, test_acc)
        np.save(val_loss_savepath, test_loss)


if __name__ == "__main__":
    model_names = sorted(name for name in MODELS.__dict__
                         if name.islower() and not name.startswith("__")
                         and callable(MODELS.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
    parser.add_argument("--model_name", type=str, default="resnet20_cifar", choices=model_names, help="model architecture")
    parser.add_argument("--dataset", type=str, default='cifar100')
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch_size", type=int, default=128, help="batch size per gpu")
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--teacher", type=str, default="resnet50_cifar", help="teacher architecture")
    parser.add_argument("--teacher_weights", type=str, default="", help="teacher weights path")
    parser.add_argument("--cls_loss_factor", type=float, default=1.0, help="cls loss weight factor")
    parser.add_argument("--kd_loss_factor", type=float, default=1.0, help="KL loss weight factor")
    parser.add_argument("--nd_loss_factor", type=float, default=1.0, help="ND loss weight factor")
    parser.add_argument("--warm_up", type=float, default=20.0, help='loss weight warm up epochs')
    parser.add_argument("--Dc", type=int, default=256, help="Dc")
    parser.add_argument("--lam_kd", type=float, default=1.0, help="lam_kd")
    parser.add_argument("--sum_lam", type=float, default=0.5, help="sum_lam")
    parser.add_argument("--lam_mmd", type=float, default=0.05, help="lam_mmd")

    parser.add_argument("--gpus", type=list, default=[0, 1])
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
    parser.add_argument("--resume", type=str, help="best ckpt's path to resume most recent training")
    parser.add_argument("--save_dir", type=str, default="./run", help="save path")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. This will turn on deterministic CUDNN.')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [line:%(lineno)d] %(message)s',
                        datefmt='%d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    args.batch_size = args.batch_size * len(args.gpus)
    logger.info(colorstr('green', "Distribute train, gpus:{}, total batch size:{}, epoch:{}".format(
        args.gpus, args.batch_size, args.epochs)))

    train_set, test_set, num_class = CIFAR(name=args.dataset)
    model = build_review_kd(student_name=args.model_name, num_class=num_class, teacher_name=args.teacher)
    teacher = MODELS.__dict__[args.teacher](num_class=num_class)

    if args.teacher_weights:
        print('Load Teacher Weights')
        teacher_ckpt = torch.load(args.teacher_weights, weights_only=False, map_location='cpu')['model']
        teacher.load_state_dict(teacher_ckpt)
        for param in teacher.parameters():
            param.requires_grad = False

    with open("./ckpt/teacher/resnet50/center_emb_train.json", 'r') as f:
        T_EMB = json.load(f)

    logger.info(colorstr('green', 'Use ' + args.teacher + ' Training ' + args.model_name + ' ...'))
    epoch_loop(model=model, teacher=teacher, train_set=train_set, test_set=test_set, args=args)
