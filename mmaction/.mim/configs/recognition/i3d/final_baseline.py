_base_ = ['./i3d_r50_32x2x1_100e_kinetics400_rgb.py']

model = dict(
    cls_head=dict(
        type='I3DHead',
        num_classes=3,
        in_channels=2048,
        spatial_type='avg',
        dropout_ratio=0.5,
        init_std=0.01),
        #multi_class=True),
)

dataset_type = 'VideoDataset'
data_root = '/u/ybkim95/Triadic'
data_root_val = data_root
ann_file_train = '/u/ybkim95/mmaction2/data/oversampled_triadic/custom_train_list.txt'
ann_file_val = '/u/ybkim95/mmaction2/data/oversampled_triadic/custom_val_list.txt'
ann_file_test = '/u/ybkim95/mmaction2/data/oversampled_triadic/custom_test_list.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 256)),
    #dict(
    #    type='MultiScaleCrop',
    #    input_size=224,
    #    scales=(1, 0.8),
    #    random_crop=False,
    #    max_wh_scale_gap=0),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 256)),
    #dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=32,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(-1, 256)),
    #dict(type='ThreeCrop', crop_size=256),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline))

total_epochs = 75

# runtime settings
work_dir = './work_dirs/i3d_custom/'
load_from = 'https://download.openmmlab.com/mmaction/recognition/i3d/i3d_r50_video_32x2x1_100e_kinetics400_rgb/i3d_r50_video_32x2x1_100e_kinetics400_rgb_20200826-e31c6f52.pth'
