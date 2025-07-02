#!/usr/bin/env python
import os
import json
import argparse
import sys
from glob import glob
from tqdm import tqdm

import torch
import numpy as np
import wandb
from ptflops import get_model_complexity_info
import netron
import torchvision

# 导入自定义模块和第三方包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import *
from report import report_comparison


def train(model, train_loader, device, criterion, optimizer, scheduler,
          early_stopping, val_loader, epochs, lr, binary, outdir):
    """
    模型训练函数，遍历多个epoch进行训练和验证，
    自动保存最优模型，并根据验证损失调整学习率和进行早停。
    """
    train_losses = []
    val_losses = []
    lr_change_epoch = 0
    lr_history = {lr_change_epoch: lr}
    best_val_loss = np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        print('Epoch:', epoch)

        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs = data['X'].to(device)
            labels = data['y'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            total_loss += loss_value
            total_count += labels.size(0)
            tqdm.write(f"Loss: {loss_value / labels.size(0):.7f}")

            # 可选：batch 级别日志
            wandb.log({
                "train/batch_loss": loss_value / labels.size(0),
                "train/global_step": epoch * len(train_loader) + tqdm.format_dict['n'],
            })

        avg_train_loss = total_loss / total_count
        print(f'Training loss: {avg_train_loss:.7f}')
        train_losses.append(avg_train_loss)
        wandb.log({
            "train/loss": avg_train_loss,
            "train/epoch": epoch,
            "lr": scheduler._last_lr[0]
        })

        # 验证
        val_dir = os.path.join(outdir, 'val', str(epoch))
        os.makedirs(val_dir, exist_ok=True)
        val_loss = evaluate(model, val_loader, device, criterion, val_dir,
                            epoch, binary, mode='val')
        val_losses.append(val_loss)
        print(f'Validation loss: {val_loss:.7f}')
        wandb.log({
            "val/loss": val_loss,
            "val/epoch": epoch,
        })

        # 保存最佳
        if val_loss < best_val_loss:
            model_path = os.path.join(outdir, 'unet_best_epoch.pt')
            opt_path = os.path.join(outdir, 'optimizer_best_epoch.pt')
            if hasattr(args, 'gpus') and args.gpus:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), opt_path)
            best_val_loss = val_loss
            best_model = model
            best_epoch = epoch

        # 学习率调度与早停
        scheduler.step(val_loss)
        early_stopping(val_loss)

        last_lr = scheduler._last_lr[0]
        if last_lr < lr_history[lr_change_epoch]:
            lr_change_epoch = epoch
            lr_history[lr_change_epoch] = last_lr
            print('Restoring best model weights.')
            if hasattr(args, 'gpus') and args.gpus:
                model.module.load_state_dict(
                    torch.load(os.path.join(outdir, 'unet_best_epoch.pt'),
                               map_location=device))
            else:
                model.load_state_dict(
                    torch.load(os.path.join(outdir, 'unet_best_epoch.pt'),
                               map_location=device))
            optimizer.load_state_dict(
                torch.load(os.path.join(outdir, 'optimizer_best_epoch.pt'),
                           map_location='cpu'))

        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    return best_model, best_epoch, epoch, lr_history, train_losses, val_losses


def evaluate(model, data_loader, device, criterion, outdir, epoch, binary, mode):
    """
    模型评估函数，用于在验证或测试阶段计算模型损失和保存预测结果。
    """
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), desc=f"Evaluating epoch {epoch}"):
            inputs = data['X'].to(device)
            truth = data['y'].to(device)
            label = data['label'].to(device)
            fname = data['fname']

            outputs = model(inputs)
            loss_value = criterion(outputs, truth).item()
            total_loss += loss_value
            batch_size = truth.size(0)
            total_count += batch_size

            # 计算准确率
            if binary:
                preds = (outputs > 0.5).long()
                correct = (preds == truth[:, 0]).sum().item()
            else:
                _, preds = outputs.max(1)
                correct = (preds == truth[:, 0]).sum().item()

            batch_loss = loss_value / batch_size
            batch_acc = correct / (batch_size * truth.size(2) * truth.size(3))
            tqdm.write(f"Loss: {batch_loss:.7f}, Acc: {batch_acc:.7f}")

            wandb.log({
                f"{mode}/batch_loss": batch_loss,
                f"{mode}/batch_acc": batch_acc,
                f"{mode}/batch": i,
            })

            # 保存可视化对比图
            if mode == 'test' or i == 0:
                num_plots = 1 if mode == 'val' else batch_size
                for n in range(num_plots):
                    preds_np = preds[n].cpu().numpy().squeeze()
                    truth_np = truth[n, 0].cpu().numpy().squeeze()
                    label_np = label[n, 0].cpu().numpy().squeeze()
                    plot_file = os.path.join(outdir, f'{fname[n]}.png')
                    report_comparison(preds=preds_np, truth=truth_np, label=label_np,
                                      file=plot_file, epoch=epoch)
                    np.savez(os.path.join(outdir, f'{fname[n]}.npz'),
                             outputs=preds_np, truth=truth_np, label=label_np)

                    wandb.log({
                        f"{mode}/comparison_ep{epoch}_{fname[n]}":
                            wandb.Image(plot_file,
                                        caption=f"{mode} ep{epoch} {fname[n]}")
                    })

    return total_loss / total_count


def visualize_model(model, input):
    import torchlens as tl
    model_history = tl.log_forward_pass(model, input, layers_to_save='all', vis_opt='rolled')
    print(model_history)


if __name__ == '__main__':
    os.environ["PYTHONPATH"] = "/content/ML-BH-reconnection"

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--indir', required=True, type=str)
    parser.add_argument('-o', '--outdir', required=True, type=str)
    parser.add_argument('-f', '--file-fraction', default=1.0, type=float)
    parser.add_argument('-d', '--data-splits', default=[0.8, 0.1, 0.1],
                        nargs='+', type=float)
    parser.add_argument('-e', '--epochs', default=10, type=int)
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('-l', '--learning-rate', default=1.e-5, type=float)
    parser.add_argument('-c', '--num-classes', default=1, type=int)
    parser.add_argument('-k', '--kernel-size', default=3, type=int)
    parser.add_argument('-y', '--height', default=5000, type=int)
    parser.add_argument('-x', '--width', default=625, type=int)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--standardize', action='store_true')
    parser.add_argument('-g', '--gpus', nargs='+', help='GPUs to run on')
    parser.add_argument('-w', '--num-workers', default=0, type=int)
    parser.add_argument('--model', default='ViTUnet', type=str)
    parser.add_argument('--loss', default='focal', type=str)
    parser.add_argument('--TwoDG', action='store_true')
    parser.add_argument('--gradB', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 数据拆分
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(project_root, args.indir)
    files = glob(os.path.join(data_dir, "*.npz"))
    train_files, val_files, test_files = split_data(files, args.file_fraction, args.data_splits)

    features = ['b1', 'b2', 'b3', 'e1', 'e2', 'e3', 'rho', 'p']
    binary = args.num_classes == 1

    # 2D Gaussian label smoothing & gradB
    gaussian = args.TwoDG
    gradB_flag = args.gradB

    # Datasets and loaders
    train_dataset = NPZDataset(train_files, features, args.normalize, args.standardize, binary, gaussian, gradB_flag)
    val_dataset   = NPZDataset(val_files,   features, args.normalize, args.standardize, binary, gaussian, gradB_flag)
    test_dataset  = NPZDataset(test_files,  features, args.normalize, args.standardize, binary, gaussian, gradB_flag)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               drop_last=True, num_workers=args.num_workers)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size,
                                               drop_last=False, num_workers=args.num_workers)
    test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=args.batch_size,
                                               drop_last=False, num_workers=args.num_workers)

    # 选择模型
    en_ch = len(features)
    if args.model == 'UNet':
        model = UNet(down_chs=(en_ch,64,128), up_chs=(128,64),
                     num_class=args.num_classes, retain_dim=True,
                     out_sz=(args.height,args.width), kernel_size=args.kernel_size)
    else:
        model = ViTUNet(down_chs=(en_ch,64,128), up_chs=(128,64),
                        num_class=args.num_classes, retain_dim=True,
                        out_sz=(args.height,args.width), kernel_size=args.kernel_size)

    # 打印和计算复杂度
    macs, params = get_model_complexity_info(model, (en_ch, args.height, args.width),
                                             as_strings=True, print_per_layer_stat=False, verbose=False)
    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')

    # 初始化 W&B
    wandb.init(
        project="ML-BH-reconnection",
        config=vars(args),
        name=f"{args.model}_bs{args.batch_size}_lr{args.learning_rate}"
    )
    wandb.watch(model, log="all", log_freq=100)

    # 多GPU or CPU
    if args.gpus:
        assert torch.cuda.is_available(), "CUDA not available."
        device = torch.device(f'cuda:{args.gpus[0]}')
        model = torch.nn.DataParallel(model, device_ids=[int(x) for x in args.gpus])
    else:
        device = torch.device('cpu')
    model.to(device)
    print('Device:', device)

    # 损失、优化器、调度器、早停
    if args.loss == 'focal':
        criterion = FocalLoss(gamma=1.5, alpha=0.85)
    elif args.loss == 'crossentropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss == 'l2':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'focall2':
        criterion = FocalMSELoss(gamma=1.5, alpha=0.85)
    elif args.loss == 'focall2+':
        criterion = FocalMSELoss(gamma=1.5, alpha=0.85, f_weight=0.5, composition_method="sum")
    elif args.loss == 'posfocus':
        criterion = PosFocusLoss()
    elif args.loss == 'posfocal':
        criterion = PosFocal(gamma=1.5, alpha=0.85)
    elif args.loss == 'hausdorff':
        criterion = SoftHausdorffLoss(alpha=5.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1.e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1.e-5, verbose=True)
    early_stopping = EarlyStopping(patience=10, min_delta=0)

    # 训练和评估
    print('Starting training...')
    best_model, best_epoch, last_epoch, lr_history, train_losses, val_losses = train(
        model, train_loader, device, criterion, optimizer, scheduler,
        early_stopping, val_loader, args.epochs, args.learning_rate, binary, args.outdir
    )
    print('Finished training!')

    print(f'Evaluating best model from epoch {best_epoch}...')
    test_dir = os.path.join(args.outdir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    test_loss = evaluate(best_model, test_loader, device, criterion, test_dir, best_epoch, binary, mode='test')
    print(f'Test loss: {test_loss:.7f}')
    wandb.log({"test/loss": test_loss})

    # 保存 metadata
    metadata = {
        'args': vars(args),
        'features': features,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'last_epoch': last_epoch,
        'best_epoch': best_epoch,
        'lr_history': lr_history,
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
    }
    with open(os.path.join(args.outdir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    wandb.finish()
