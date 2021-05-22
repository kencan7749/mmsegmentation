# dataset settings
dataset_type = "ParticleDataset"#'Particle_detection'
data_root ="/var/datasets/rain_filtering/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512,32), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiPhotoMetricDistortion'),
    dict(type='MultiNormalize', **img_norm_cfg), #muzu
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 32),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='range_images/train',
        ann_dir='ann_dir/train/first',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='range_images/val',
        ann_dir='particle_labels/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='range_images/train',
        ann_dir='particle_labels/train/first',
        pipeline=test_pipeline))
