norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SegMANEncoder_t',
        style='pytorch',
        pretrained=
        'C:/ZZworkspace/zyworkspace/project/weights/cv/SegMAN/SegMAN_Encoder_t.pth.tar'
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        decoder_params=dict(embed_dim=128),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        feat_proj_dim=192),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'ADE20KDataset'
data_root = 'C:/ZZworkspace/zyworkspace/project/datasets/dis/General'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile', imdecode_backend='pillow'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='AlignedResize', keep_ratio=False, size_divisor=32),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    workers_per_gpu=2,
    samples_per_gpu=1,
    train_dataloader=dict(drop_last=False),
    train=dict(
        type='CustomDataset',
        data_root='C:/ZZworkspace/zyworkspace/project/datasets/dis/General',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        classes=('background', 'solar_panel'),
        palette=[[0, 0, 0], [255, 80, 0]],
        reduce_zero_label=False,
        img_dir='train/image',
        ann_dir='train/label',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='pillow'),
            dict(type='LoadAnnotations', reduce_zero_label=False),
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ]),
    val_dataloader=dict(samples_per_gpu=1, drop_last=False),
    val=dict(
        type='CustomDataset',
        data_root='C:/ZZworkspace/zyworkspace/project/datasets/dis/General',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        classes=('background', 'solar_panel'),
        palette=[[0, 0, 0], [255, 80, 0]],
        reduce_zero_label=False,
        img_dir='val/image',
        ann_dir='val/label',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='pillow'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(
                        type='AlignedResize',
                        keep_ratio=False,
                        size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test_dataloader=dict(samples_per_gpu=1),
    test=dict(
        type='CustomDataset',
        data_root='C:/ZZworkspace/zyworkspace/project/datasets/dis/General',
        img_suffix='.tif',
        seg_map_suffix='.tif',
        classes=('background', 'solar_panel'),
        palette=[[0, 0, 0], [255, 80, 0]],
        reduce_zero_label=False,
        img_dir='test/image',
        ann_dir='test/label',
        pipeline=[
            dict(type='LoadImageFromFile', imdecode_backend='pillow'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(
                        type='AlignedResize',
                        keep_ratio=False,
                        size_divisor=32),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10)
checkpoint_config = dict(
    by_epoch=False, interval=1, save_last=True, max_keep_ckpts=3)
evaluation = dict(
    interval=1,
    metric=['mIoU', 'mDice', 'mFscore'],
    save_best='mIoU',
    priority='LOW',
    pre_eval=True,
    fixed_vis_names=[
        'F47Mosaic_F47D001008_00000018', 'F48Mosaic_F48D001002_00000043',
        'F48Mosaic_F48D002001_00000057', 'F48Mosaic_F48D002003_00000091'
    ])
classes = ('background', 'solar_panel')
palette = [[0, 0, 0], [255, 80, 0]]
input_size = (512, 512)
dataset = dict(
    type='CustomDataset',
    data_root='C:/ZZworkspace/zyworkspace/project/datasets/dis/General',
    img_suffix='.tif',
    seg_map_suffix='.tif',
    classes=('background', 'solar_panel'),
    palette=[[0, 0, 0], [255, 80, 0]],
    reduce_zero_label=False)
fixed_vis_names = [
    'F47Mosaic_F47D001008_00000018', 'F48Mosaic_F48D001002_00000043',
    'F48Mosaic_F48D002001_00000057', 'F48Mosaic_F48D002003_00000091'
]
find_unused_parameters = False
work_dir = './work_dirs\segman_t_general'
gpu_ids = [0]
auto_resume = False
