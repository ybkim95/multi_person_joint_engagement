_base_ = ['../../_base_/models/x3d.py']

model = dict(
    type='Recognizer3D',
    cls_head=dict(
        type='X3DHead',
        num_classes=3,
        spatial_type='avg',
        dropout_ratio=0.5,
        fc1_bias=False),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

# dataset settings
dataset_type = 'VideoDataset'
data_root = '/u/ybkim95/CutOutTriadic'
data_root_val = data_root
ann_file_train = '/u/ybkim95/mmaction2/data/cutouttriadic/custom_train_list.txt'
ann_file_val = '/u/ybkim95/mmaction2/data/cutouttriadic/custom_val_list.txt'
ann_file_test = '/u/ybkim95/mmaction2/data/cutouttriadic/custom_test_list.txt'


img_norm_cfg = dict(
    mean=[114.75, 114.75, 114.75], std=[57.38, 57.38, 57.38], to_bgr=False)

train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=2, num_clips=1),
    dict(type='DecordDecode'),
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
        clip_len=16,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
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
        clip_len=16,
        frame_interval=2,
        num_clips=10,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
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


log_level = "INFO"
dist_params = dict(backend='nccl')

optimizer = dict(
        type="Adam",
        lr=0.001,
        weight_decay=0.0001
        )

optimizer_config = dict(grad_clip=None)

lr_config=dict(policy='step', step=7)

total_epochs=20

log_config=dict(interval=10, hooks=[dict(type='TextLoggerHook')])

resume_from=None

work_dir = './work_dirs/x3d_custom_cutout'
#output_config = dict(out=f'{work_dir}/result.json', output_format='json')

checkpoint_config=dict(interval=5)
workflow = [('train', 1)]
load_from = 'https://download.openmmlab.com/mmaction/recognition/x3d/facebook/x3d_m_facebook_16x5x1_kinetics400_rgb_20201027-3f42382a.pth'
