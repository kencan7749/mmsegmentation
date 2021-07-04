# dataset settings
dataset_type = "ParticleDataset"
data_root ="data/rain_filtering_v2"#"/var/datasets/rain_filtering/"
#the order of mean or std should be ['first_depth', 'first_intensity', 'first_return_type',
# 'last_depth', 'last_intensity', 'last_return_type']
img_norm_cfg = dict(
    mean=[5038.093, 8.658958, 5.960323, 6241.105,
       9.026124, 5.805868], std=[2592.0776, 5.1905198, 1.5205307, 2186.4004,
       5.1824026, 1.6501048], to_rgb=False) # to_rgb is False since each image are loaded gray
crop_size = (32, 32) # crop 32x32 from 40x1800 
img_scale = (40, 1800)
train_pipeline = [
    dict(type='LoadMultiImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='MultiNormalize', **img_norm_cfg), #muzu
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    # do LoadMultiIageFromFile but it is just to use mmseg apis. (img_ratio is just 1.0)
    dict(type='LoadMultiImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            #dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0.0),
            dict(type='MultiNormalize', **img_norm_cfg),
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
        ann_dir='ann_dir/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='range_images/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='range_images/test',
        ann_dir='ann_dir/test',
        pipeline=test_pipeline))
