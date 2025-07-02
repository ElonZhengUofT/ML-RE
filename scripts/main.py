from scripts.train import *
from scripts.report import *

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

if __name__ == '__main__':
    # 设置命令行参数解析
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-d', '--dir', required=True, type=str,
                            help="结果文件所在目录")
    arg_parser.add_argument('-g', '--gif', action='store_true',
                            help="是否生成gif动画")
    arg_parser.add_argument('-m', '--modeldir', help="模型文件目录")
    # f 表示固定阈值（若需要）
    arg_parser.add_argument('-f')
    args = arg_parser.parse_args()

    ############################################################################
    # 绘制各向异性图及重连点
    ############################################################################
    # 这里使用一个示例数据文件 'sample/data/3600.npz'
    # if the following file is founded
    if os.path.exists('smaller_data/data_time_0032_smaller.npz'):
        earth_center_x, xmin, xmax, zmin, zmax = report_reconnection_points('smaller_data/data_time_0032_smaller.npz')
    elif os.path.exists('/Users/zhengshizhao/PycharmProjects/ML-BH-reconnection/smaller_data/data_time_0032_smaller.npz'):
        earth_center_x, xmin, xmax, zmin, zmax = report_reconnection_points('/Users/zhengshizhao/PycharmProjects/ML-BH-reconnection/smaller_data/data_time_0032_smaller.npz')



    ############################################################################
    # 读取元数据
    ############################################################################
    if args.modeldir:
        # 若提供了模型目录，则从该目录中读取元数据
        metadata_path = os.path.join(args.modeldir, 'metadata.json')
    else:
        metadata_path = os.path.join(args.dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    ############################################################################
    # Accoring to the metadata, report the args of training
    ############################################################################
    print(metadata['args'])

    ############################################################################
    # 绘制损失曲线（仅在未指定模型目录时）
    ############################################################################
    if not args.modeldir:
        if 'lr_history' in metadata.keys():
            lr_history = [int(epoch) for epoch in metadata['lr_history'].keys()]
        else:
            lr_history = None
        report_loss(metadata['train_losses'], metadata['val_losses'], lr_history,
                  args.dir)

        ############################################################################
        # 若指定生成gif动画，则按几何序列生成验证结果动画帧
        ############################################################################

        if args.gif:
            frames = []
            fname = Path(metadata['val_files'][0]).stem
            sequence = generate_geom_seq(metadata['last_epoch'])
            for i in sequence:
                # 加载对应epoch的预测结果与真实标签
                data = np.load(
                    os.path.join(args.dir, 'val', str(i), f'{fname}.npz'))
                preds, truth = data['outputs'], data['label']
                frame = report_gif_frame(preds, truth, i, xmin, xmax, zmin, zmax)
                frames.append(frame)
            gif.save(frames, os.path.join(args.dir, 'epochs.gif'), duration=200)

    ############################################################################
    # 处理测试集预测结果
    ############################################################################
    test_list = glob(os.path.join(args.dir, 'test', '*.npz'))
    num_test_files = len(test_list)
    all_preds = np.zeros(
        (num_test_files, metadata['args']['height'], metadata['args']['width']))
    all_truth = np.zeros(
        (num_test_files, metadata['args']['height'], metadata['args']['width']))
    # 遍历所有测试文件，收集预测与真实结果，并保存对比图
    for i, test_file in enumerate(test_list):
        data = np.load(test_file)
        preds, truth = data['outputs'], data['label']
        if args.modeldir:
            test_plot_file = os.path.join(args.dir, 'test',
                                          Path(test_file).stem)
            plot_comparison(preds, truth, test_plot_file,
                            metadata['best_epoch'])
        all_preds[i] = preds
        print(f'all_truth shape: {all_truth.shape}')
        print(f'truth shape: {truth.shape}')
        all_truth[i] = truth

    # 根据地球中心划分夜侧和日侧数据
    nightside_preds = all_preds[:, :, :earth_center_x]
    nightside_truth = all_truth[:, :, :earth_center_x]
    dayside_preds = all_preds[:, :, earth_center_x:]
    dayside_truth = all_truth[:, :, earth_center_x:]

    ############################################################################
    # 分别对整体、夜侧和日侧计算Precision-Recall及混淆矩阵
    ############################################################################
    f1 = {}
    f2 = {}
    tpr = {}
    for preds, truth, side in [
        (all_preds.ravel(), all_truth.ravel(), 'both_sides'),
        (nightside_preds.ravel(), nightside_truth.ravel(), 'nightside'),
        (dayside_preds.ravel(), dayside_truth.ravel(), 'dayside')
    ]:
        # if there is no positive in truth, skip, and set f1 and f2 to 0
        if len(np.where(truth != 0)[0]) == 0:
            continue
        side_dir = os.path.join(args.dir, side)
        os.makedirs(side_dir, exist_ok=True)

        # find where is positive in truth
        where_positive = np.where(truth == 1)
        # 计算Precision-Recall曲线
        precision, recall, thresholds = metrics.precision_recall_curve(truth,
                                                                       preds, pos_label=1)
        d = {'precision': precision, 'recall': recall}
        np.savez(os.path.join(side_dir, 'precision_recall.npz'), **d)

        # 根据F1得分确定最佳阈值
        max_f1_score, max_f1_index, max_f1_thresh = pick_best_threshold_by_f_beta(
            precision, recall, thresholds, 1)
        f1[side] = {'score': max_f1_score, 'threshold': max_f1_thresh}
        binary_preds = np.where(preds < max_f1_thresh, 0, 1)
        report_confusion_matrix(binary_preds, truth, 'f1', side_dir)

        # 根据F2得分确定最佳阈值
        max_f2_score, max_f2_index, max_f2_thresh = pick_best_threshold_by_f_beta(
            precision, recall, thresholds, 2)
        f2[side] = {'score': max_f2_score, 'threshold': max_f2_thresh}
        binary_preds = np.where(preds < max_f2_thresh, 0, 1)
        report_confusion_matrix(binary_preds, truth, 'f2', side_dir)

        # Get the best threshold by the intersection of precision and recall
        intersection_diff, intersection_index, intersection_thresh = pick_best_threshold_by_intersection(
            precision, recall, thresholds)
        intersection = {}  # Dictionary to hold the intersection-based threshold result if needed
        intersection[side] = {'diff': intersection_diff,
                              'threshold': intersection_thresh}

        # Generate binary predictions using the intersection-based threshold
        binary_preds = np.where(preds < intersection_thresh, 0, 1)
        report_confusion_matrix(binary_preds, truth, 'intersection', side_dir)

        #         max_tpr, max_tpr_index, best_threshold = pick_best_threshold_by_high_tpr(
        #             precision, recall, thresholds, min_precision=1e-6)
        #         tpr[side] = {'score': max_tpr, 'threshold': best_threshold}
        #         binary_preds = np.where(preds < best_threshold, 0, 1)
        #         report_confusion_matrix(binary_preds, truth, 'tpr', side_dir)

        # 绘制Precision-Recall曲线及阈值曲线
        report_precision_recall(precision, recall, max_f1_score, max_f1_index,
                                max_f1_thresh,
                                max_f2_score, max_f2_index, max_f2_thresh,
                                side_dir)
        report_thresholds(precision, recall, thresholds, max_f1_thresh,
                          max_f2_thresh,
                          side_dir)
        report_precision_recall_by_threshold(precision, recall, thresholds,
                                             intersection_index,
                                             intersection_thresh,
                                                side_dir)

    if 'dayside' in f1:
        binary_preds_f1_nightside = np.where(
            nightside_preds < f1['nightside']['threshold'], 0, 1)
        binary_preds_f1_dayside = np.where(
            dayside_preds < f1['dayside']['threshold'], 0, 1)
        all_binary_preds = np.concatenate(
            (binary_preds_f1_nightside, binary_preds_f1_dayside), axis=2)
    else:
        print(
            "dayside f1 threshold not available, using nightside only for f1.")
        all_binary_preds = np.where(
            nightside_preds < f1['nightside']['threshold'], 0, 1)

    report_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(),
                            'f1', args.dir)

    if 'dayside' in f2:
        binary_preds_f2_nightside = np.where(
            nightside_preds < f2['nightside']['threshold'], 0, 1)
        binary_preds_f2_dayside = np.where(
            dayside_preds < f2['dayside']['threshold'], 0, 1)
        all_binary_preds = np.concatenate(
            (binary_preds_f2_nightside, binary_preds_f2_dayside), axis=2)
    else:
        print(
            "dayside f2 threshold not available, using nightside only for f2.")
        all_binary_preds = np.where(
            nightside_preds < f2['nightside']['threshold'], 0, 1)

    report_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(),
                            'f2', args.dir)



#     if 'dayside' in tpr:
    #         binary_preds_tpr_nightside = np.where(
    #             nightside_preds < tpr['nightside']['threshold'], 0, 1)
    #         binary_preds_tpr_dayside = np.where(
    #             dayside_preds < tpr['dayside']['threshold'], 0, 1)
    #         all_binary_preds = np.concatenate(
    #             (binary_preds_tpr_nightside, binary_preds_tpr_dayside), axis=2)
    #     else:
    #         print(
    #             "dayside tpr threshold not available, using nightside only for tpr.")
    #         all_binary_preds = np.where(
    #             nightside_preds < tpr['nightside']['threshold'], 0, 1)
    #
    #     report_confusion_matrix(all_binary_preds.ravel(), all_truth.ravel(),
    #                             'tpr', args.dir)
    ############################################################################
    #
    ############################################################################

    ############################################################################
    # 绘制ROC曲线，并输出分类指标
    ############################################################################
    report_roc(all_preds.ravel(), all_truth.ravel(), args.dir)
    metrics_dict = evaluate_classifier(all_binary_preds.ravel(),
                                       all_truth.ravel())
    metrics_dict['F1'] = f1
    metrics_dict['F2'] = f2
    metrics_dict['TPR'] = tpr
    metrics_dict['Intersection'] = intersection
    print(json.dumps(metrics_dict, indent=2))
    metrics_dict['args'] = metadata['args']

    # 将计算得到的指标保存到metrics.json文件中
    with open(os.path.join(args.dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

