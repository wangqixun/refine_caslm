classes = [
    'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard', 'sports ball',
    'baseball bat', 'skateboard', 'tennis racket'
]
num_classes = 9
model = dict(
    type='MaskRCNN',
    pretrained=None,
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='CBFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_decoded_bbox=True,
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=9,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
        train_cfg=None,
        test_cfg=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=-1),
        pretrained=None),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='OHEMSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.1,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='InstaBoost',
        action_candidate=('normal', 'horizontal', 'skip'),
        action_prob=(1, 0, 0),
        scale=(0.8, 1.2),
        dx=15,
        dy=15,
        theta=(-1, 1),
        color_prob=0.5,
        hflag=False,
        aug_ratio=0.5),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=[(640, 360), (640, 180)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=640)
]
train_pipeline = [
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train_data = [
    dict(
        type='CocoDataset',
        classes=[
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
            'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
        ],
        ann_file=
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0512.json',
        img_prefix='/share/yanzhen/tools/internal-data/0425/0512/dataset/new/',
        pipeline=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    dict(
        type='CocoDataset',
        classes=[
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
            'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
        ],
        ann_file=
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/0517/annotations/split1/dataset/new/',
        pipeline=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    dict(
        type='CocoDataset',
        classes=[
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
            'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
        ],
        ann_file=
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp2.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/0517/annotations/split2/dataset/new/',
        pipeline=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ])
]
test_data = dict(
    type='CocoDataset',
    classes=[
        'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
        'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
    ],
    ann_file=
    '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_test_xhs9.json',
    img_prefix=
    '/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/dataset/new/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(640, 640),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Pad', size_divisor=32),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=[
                'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
                'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
            ],
            ann_file=
            '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0512.json',
            img_prefix=
            '/share/yanzhen/tools/internal-data/0425/0512/dataset/new/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='InstaBoost',
                    action_candidate=('normal', 'horizontal', 'skip'),
                    action_prob=(1, 0, 0),
                    scale=(0.8, 1.2),
                    dx=15,
                    dy=15,
                    theta=(-1, 1),
                    color_prob=0.5,
                    hflag=False,
                    aug_ratio=0.5),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(640, 360), (640, 180)],
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Pad', size_divisor=640)
            ]),
        pipeline=[
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=[
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
            'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
        ],
        ann_file=
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_test_xhs9.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/dataset/new/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=[
            'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard',
            'sports ball', 'baseball bat', 'skateboard', 'tennis racket'
        ],
        ann_file=
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_test_xhs9.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/dataset/new/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(metric=['bbox', 'segm'])
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
fp16 = None
optimizer_config = dict(
    grad_clip=None,
    type='DistOptimizerHook',
    update_interval=1,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=False)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[27, 33])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=36)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
resume_from = None
work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/mask_rcnn_cb2_swimtiny_coco80_0512_0517sp1_0517sp2_ms_giou_ohem_InstaBoost'
gpu_ids = range(0, 8)