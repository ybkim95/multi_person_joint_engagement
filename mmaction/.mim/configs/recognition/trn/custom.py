_base_ = [
    '../../_base_/models/trn_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# model settings
model = dict(cls_head=dict(num_classes=3))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/u/ybkim95/Triadic'
data_root_val = data_root
ann_file_train = '/u/ybkim95/mmaction2/data/triadic/custom_train_list.txt'
ann_file_val = '/u/ybkim95/mmaction2/data/triadic/custom_val_list.txt'
ann_file_test = '/u/ybkim95/mmaction2/data/triadic/custom_test_list.txt'

sthv1_flip_label_map = {2: 4, 4: 2, 30: 41, 41: 30, 52: 66, 66: 52}

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),   
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv1_flip_label_map),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        twice_sample=True,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        filename_tmpl='{:05}.jpg',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        filename_tmpl='{:05}.jpg',
        pipeline=test_pipeline))
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(lr=0.002, paramwise_cfg=dict(fc_lr5=False), weight_decay=5e-4)
# learning policy
lr_config = dict(policy='step', step=[30, 45])
total_epochs = 50

# runtime settings
load_from = "https://download.openmmlab.com/mmaction/recognition/trn/trn_r50_1x1x8_50e_sthv1_rgb/trn_r50_1x1x8_50e_sthv1_rgb_20210401-163704a8.pth"
find_unused_parameters = True
work_dir = './work_dirs/trn_custom'

