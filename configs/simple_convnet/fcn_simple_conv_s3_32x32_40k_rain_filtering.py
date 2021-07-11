_base_ = [
    '../_base_/models/fcn_simple_conv_s3_multi.py', '../_base_/datasets/rain_filtering.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
model = dict(test_cfg=dict(mode='whole',crop_size=(32, 32), stride=(20, 20)))
#evaluation = dict(metric='mDice')#dict(metric='mIoU')#

cudnn_benchmark = False