# ==================== 公共配置 ====================

# 复用 SegMAN-T 的模型结构、优化器和学习率策略。
_base_ = './segman_t_ade.py'

# General 数据集的绝对路径。各阶段的目录以此路径为根。
data_root = 'C:/ZZworkspace/zyworkspace/project/datasets/dis/General'
# 50 轮实验的日志和权重目录。
work_dir = 'work_dirs/segman_t_general_50epochs'

# 类别索引与 TIFF 掩码像素值对应，0 为背景，1 为光伏面板。
classes = ('background', 'solar_panel')
palette = [[0, 0, 0], [255, 80, 0]]
# 输入尺寸按（宽，高）填写，对应 General 原始 TIFF 的尺寸。
input_size = (512, 512)
# 转为 RGB 并使用 ImageNet 的均值和标准差归一化。
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# 三个数据集拆分共用 TIFF 后缀、类别和标签规则。
dataset = dict(
    type='CustomDataset',
    data_root=data_root,
    img_suffix='.tif',
    seg_map_suffix='.tif',
    classes=classes,
    palette=palette,
    reduce_zero_label=False)

# 替换继承的 ADE20K 数据设置，以下各阶段分别填入对应拆分。
data = dict(
    _delete_=True,
    workers_per_gpu=2)  # DataLoader 默认每张 GPU 启动 2 个数据加载进程。

model = dict(
    # 使用 SegMAN-T 在 ImageNet-1K 上预训练的编码器权重。
    backbone=dict(
        pretrained='C:/ZZworkspace/zyworkspace/project/weights/cv/SegMAN/SegMAN_Encoder_t.pth.tar'),
    decode_head=dict(
        num_classes=len(classes),
        # 全局池化分支的特征图为 1 × 1，使用 GroupNorm 支持单张训练。
        norm_cfg=dict(_delete_=True, type='GN', num_groups=32,
                      requires_grad=True)))

# 验证和测试共用单尺度推理流程。输入宽高都需要能被 32 整除。
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=False, size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]),
]

# ==================== 训练 ====================

# 联合读取影像和 GT，缩放后进行随机翻转和颜色扰动。
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=input_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

data.update(
    samples_per_gpu=8,  # 训练 batch size，单张 GPU 每步处理 8 张图。
    train_dataloader=dict(
        drop_last=False),  # 保留最后一个不足 batch size 的批次。
    train=dict(
        **dataset, img_dir='train/image', ann_dir='train/label',
        pipeline=train_pipeline))

# 完整遍历训练集 50 次，每个训练 batch 更新一次参数。
runner = dict(_delete_=True, type='EpochBasedRunner', max_epochs=50)
# 学习率策略沿用父配置，将线性预热缩短到 1 步。
lr_config = dict(warmup_iters=1)
# 每轮验证后保存常规 checkpoint，最多保留最近 3 个。
checkpoint_config = dict(
    by_epoch=True, interval=1, save_last=True, max_keep_ckpts=3)
# 每 50 次参数更新记录训练日志和 TensorBoard 标量，轮末也记录。
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True, ignore_last=False),
        dict(type='TensorboardLoggerHook', by_epoch=True, ignore_last=False),
    ])

# ==================== 验证 ====================

# 默认评估完整验证集。可通过 data.val_dataloader.indices 指定评估样本。
data.update(
    val_dataloader=dict(samples_per_gpu=1, drop_last=False),
    val=dict(
        **dataset, img_dir='val/image', ann_dir='val/label',
        pipeline=test_pipeline))

# 每次验证为这四张固定样本各输出一张四联图，并写入 TensorBoard。
# 样本用文件名主干指定，读取范围为完整验证集。
fixed_vis_names = [
    'F47Mosaic_F47D001008_00000018',
    'F48Mosaic_F48D001002_00000043',
    'F48Mosaic_F48D002001_00000057',
    'F48Mosaic_F48D002003_00000091',
]
evaluation = dict(
    interval=1,  # 每个 epoch 结束后验证一次并输出四联图。
    priority='ABOVE_NORMAL',  # 先验证，再保存当轮的常规 checkpoint。
    metric=['mIoU', 'mDice', 'mFscore'],
    pre_eval=True,  # 按样本累计指标，减少保留整批预测的内存占用。
    save_best='mIoU',  # 按验证集 mIoU 选出最佳权重。
    fixed_vis_names=fixed_vis_names)

# ==================== 测试 ====================

# tools/test.py 读取 test 拆分。测试指标通过入口的 --eval 参数指定。
data.update(
    test_dataloader=dict(samples_per_gpu=1),
    test=dict(
        **dataset, img_dir='test/image', ann_dir='test/label',
        pipeline=test_pipeline))
