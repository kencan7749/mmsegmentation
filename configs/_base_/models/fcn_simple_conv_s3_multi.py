# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SimpleConvNet',
        in_channels=6,
        out_channels=64,
        base_channels=64,
        num_convs=3,
        strides=(1,1,1),
        dilations=(1,1,1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        ),
    decode_head=dict(
        type='FCNHead',
        in_channels=64,
        in_index=0,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    #auxiliary_head=dict(
    #    type='FCNHead',
    #    in_channels=64,
    #    in_index=0,
    #    channels=64,
    #    num_convs=1,
    #    concat_input=False,
    #    dropout_ratio=0.1,
    #    num_classes=4,
    #    norm_cfg=norm_cfg,
    #    align_corners=False,
    #    loss_decode=dict(
    #        type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=32, stride=20))