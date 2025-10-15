#!/usr/bin/env python3
import os
import csv
import subprocess
from datetime import datetime
import glob, torch

# CSV_FILE = "wrn40_2-wrn40_1_kd+++_Dc.csv"
# CSV_FILE = "res32x4-res8x4_kd+++_Dc.csv"
CSV_FILE = "res50-MV2_kd+++_Dc.csv"
GPU_IDS  = "0,1"                      # 按需改卡号
# BASE_ARGS = [
#     "--model_name", "resnet8x4_cifar",
#     "--teacher", "resnet32x4_cifar",
#     "--teacher_weights", "./ckpt/cifar_teachers/resnet32x4_vanilla/ckpt_epoch_240.pth",
#     "--dataset", "cifar100",
#     "--epoch", "240",
#     "--batch_size", "64",
# ]
BASE_ARGS = [
    "--model_name", "mobilenetv2",
    "--teacher", "resnet50_cifar",
    "--teacher_weights", "./ckpt/cifar_teachers/ResNet50_vanilla/ckpt_epoch_240.pth",
    "--dataset", "cifar100",
    "--epoch", "240",
    "--batch_size", "64",
    "--lr", "0.02",
]
# BASE_ARGS = [
#     "--model_name", "resnet20_cifar",
#     "--teacher", "resnet56_cifar",
#     "--teacher_weights", "./ckpt/cifar_teachers/resnet56_vanilla/ckpt_epoch_240.pth",
#     "--dataset", "cifar100",
#     "--epoch", "240",
#     "--batch_size", "64",
#     "--lr", "0.05",
# ]
# BASE_ARGS = [
#     "--model_name", "wrn40_1_cifar",
#     "--teacher", "wrn40_2_cifar",
#     "--teacher_weights", "./ckpt/cifar_teachers/wrn_40_2_vanilla/ckpt_epoch_240.pth",
#     "--dataset", "cifar100",
#     "--epoch", "240",
#     "--batch_size", "64",
#     "--lr", "0.02",
# ]

# 1. 手动写每次要跑的参数组合，想跑几组写几组
MY_EXPS = [
    # res56-res20
    # (5e-4, 256, 1.2, 0.4, 0.04),
    # (5e-4, 512, 1.2, 0.4, 0.04),
    # (5e-4, 1024, 1.2, 0.4, 0.04),
    # (5e-4, 2048, 1.2, 0.4, 0.04),
    # wrn40_2-wrn40_1
    # (3e-4, 256, 1.2, 0.4, 0.04),
    # (3e-4, 512, 1.2, 0.4, 0.04),
    # (3e-4, 1024, 1.2, 0.4, 0.04),
    # (3e-4, 2048, 1.2, 0.4, 0.04),
    # res32x4-res8x4
    # (7e-4, 256, 2.0, 0.4, 0.04),
    # (7e-4, 512, 2.0, 0.4, 0.04),
    # (7e-4, 1024, 2.0, 0.4, 0.04),
    # (7e-4, 2048, 2.0, 0.4, 0.04),
    # res50-MV2
    (8e-4, 256, 2.5, 0.8, 0.10),
    (8e-4, 512, 2.5, 0.8, 0.10),
    (8e-4, 1024, 2.5, 0.8, 0.10),
    (8e-4, 2048, 2.5, 0.8, 0.10),
]

# 2. 如果 csv 已存在，已跑过的组合自动跳过
import pandas as pd
done = set(pd.read_csv(CSV_FILE)[["weight_decay","Dc","lam_kd","sum_lam","lam_mmd"]].itertuples(index=False, name=None)) if os.path.exists(CSV_FILE) else set()

with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not done:
        writer.writerow(["weight_decay","Dc","lam_kd","sum_lam","lam_mmd","best_test_acc","time"])

    for wd, dc, kd, su, mmd in MY_EXPS:
        if (wd, dc, kd, su, mmd) in done:
            print(f"skip {(wd, dc, kd, su, mmd)} — already in csv")
            continue

        run_name = f"wd{wd}_dc{dc}_kd{kd}_su{su}_mmd{mmd}".replace(".","")
        cmd = ["python3", "train_cifar_ckd.py"] + BASE_ARGS + [
            f"--weight_decay={wd}",
            f"--Dc={dc}",
            f"--lam_kd={kd}",
            f"--sum_lam={su}",
            f"--lam_mmd={mmd}",
            f"--save_dir=./run/wrn40/{run_name}"
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = GPU_IDS

        print(f"\n>>>> launching {run_name} …")
        subprocess.run(cmd, env=env, check=True)          # 等待跑完

        # 读取结果写入表格
        # 找到 best 目录下最新一个 epoch_*_acc_* 目录里的 ckpt.pth
        log_dir = os.path.join(".", "run", run_name)   # 相对路径
        best_dir  = os.path.join(log_dir, "weights", "best")
        ckpt_list = glob.glob(os.path.join(best_dir, "epoch_*_acc_*", "ckpt.pth"))
        if ckpt_list:
            latest_ckpt = max(ckpt_list, key=os.path.getmtime)   # 取最新
            state_dict  = torch.load(latest_ckpt, map_location="cpu")
            best_error  = state_dict["best_error"]               # 直接读
            best_acc    = 1. - best_error
        else:
            best_acc = -1.0

        writer.writerow([wd, dc, kd, su, mmd, best_acc, datetime.now().strftime("%m%d-%H:%M")])
        f.flush()
