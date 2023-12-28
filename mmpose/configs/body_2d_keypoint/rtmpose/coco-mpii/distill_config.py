_base_ = [
    '../../../rtmpose/coco/rtmpose-m_8xb256-420e_mpii-coco-256x192.py'
]

# config settings
logit = True

train_cfg = dict(max_epochs=60, val_interval=10)

# method details
model = dict(
    _delete_=True,
    type='MultiTeacherDistiller',
    teacher1_cfg='',
    teacher2_cfg='',
    student_cfg='',
    distill_cfg=[
        dict(methods=[
            dict(
                type='KDLoss',
                name='loss_logit',
                use_this=logit,
                weight=1,
            )
        ]),
    ],
    teacher1_pretrained='work_dirs/..',
    teacher2_pretrained='',
    train_cfg=train_cfg,
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
)

optim_wrapper = dict(clip_grad=dict(max_norm=1., norm_type=2))