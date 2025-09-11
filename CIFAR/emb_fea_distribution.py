"""
Visualized the embedding feature of the pre-train model on the training set.
"""
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict

import Models
from Dataset import CIFAR

import numpy as np
import argparse
import json

def emb_fea(model, data, args):
    # model to evaluate mode
    model.eval()
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    EMB = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.cuda(), labels.cuda()

            # compute output
            emb_fea, logits = model(images, embed=True)

            for emb, i in zip(emb_fea, labels):
                i = i.item()
                assert len(emb) == args.emb_size
                if str(i) in EMB:
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))
                else:
                    EMB[str(i)] = [[] for _ in range(len(emb))]
                    for j in range(len(emb)):
                        EMB[str(i)][j].append(round(emb[j].item(), 4))

    
    for key, value in EMB.items():
        for i in range(args.emb_size):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB


if __name__ == "__main__":
    model_names = sorted(name for name in Models.__dict__ 
                         if name.islower() and not name.startswith("__") 
                         and callable(Models.__dict__[name]))

    parser = argparse.ArgumentParser(description='Visualized the embedding feature of the model on the train set.')
    parser.add_argument("--model_name", type=str, default="resnet56_cifar", choices=model_names, help="model architecture")
    parser.add_argument("--model_weights", type=str, default="", help="model weights path")
    parser.add_argument("--emb_size", type=int, default=64, help="emb fea size")
    parser.add_argument("--dataset", type=str, default='cifar10')
    parser.add_argument("--batch_size", type=int, default=64, help="total batch size")
    parser.add_argument('--workers', default=16, type=int, help='number of data loading workers')
    args = parser.parse_args()
    
    # dataset
    if args.dataset in ['cifar10', 'cifar100']:
        train_set, test_set, num_class = CIFAR(name=args.dataset)
    else:
        print("No Dataset!!!")
    
    model = Models.__dict__[args.model_name](num_class=num_class)
        
    if args.model_weights:
        print('Visualized the embedding feature of the {} model on the train set'.format(args.model_name))
        
        # model_ckpt = torch.load(args.model_weights)['model_state_dict']
        model_ckpt = torch.load(args.model_weights)['model']
        # new_state_dict = OrderedDict()
        # for k, v in model_ckpt.items():
        #     name = k[7:]   # remove 'module.'
        #     new_state_dict[name] = v
        new_state_dict = OrderedDict()
        for k, v in model_ckpt.items():
            # 1. 如果 key 本身带 'module.' → 直接去掉
            if k.startswith('module.'):
                new_key = k[7:]
            # 2. 如果 key 是不带层级的扁平 key → 补层级
            elif k.startswith(('0.', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                # 根据数字前缀判断属于哪一层
                idx = int(k[0])
                if idx < 9:          # 0-8 → layer1.0-8
                    new_key = f'layer1.{k}'
                elif idx < 18:       # 9-17 → layer2.0-8
                    new_key = f'layer2.{k[2:]}'
                else:                # 18-26 → layer3.0-8
                    new_key = f'layer3.{k[2:]}'
            # 把最顶层的 bn1.* 改为 bn.*
            # elif k.startswith('bn1.'):
            #     new_key = k.replace('bn1.', 'bn.', 1)
            # 3. 其他顶层 key（conv1、bn1、fc）保持不变
            else:
                new_key = k
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict)

        for param in model.parameters():
            param.requires_grad = False

    else:
        print('No load Pre-trained weights!')
    
    model = model.cuda()

    emb = emb_fea(model=model, data=train_set, args=args)
    emb_json = json.dumps(emb, indent=4)
    with open("./run/{}_embedding_fea/{}.json".format(args.dataset, args.model_name), 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()