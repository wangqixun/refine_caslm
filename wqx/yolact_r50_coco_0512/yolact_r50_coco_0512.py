classes = [
    'person', 'bicycle', 'motorcycle', 'frisbee', 'snowboard', 'sports ball',
    'baseball bat', 'skateboard', 'tennis racket'
]
num_classes = 9
find_unused_parameters = True
img_size = 640
model = dict(
    type='YOLACT',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        zero_init_residual=False,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=9,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[8, 16, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[
                9.27536231884058, 18.285714285714285, 35.55555555555556,
                71.11111111111111, 128.0
            ],
            centers=[(4.63768115942029, 4.63768115942029),
                     (9.142857142857142, 9.142857142857142),
                     (17.77777777777778, 17.77777777777778),
                     (35.55555555555556, 35.55555555555556), (64.0, 64.0)]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=256,
        num_protos=32,
        num_classes=9,
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=9,
        in_channels=256,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        top_k=200,
        max_per_img=100))
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.4, 57.12, 57.38], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=[123.68, 116.78, 103.94],
        to_rgb=True,
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size_divisor=640),
    dict(
        type='Normalize',
        mean=[123.68, 116.78, 103.94],
        std=[58.4, 57.12, 57.38],
        to_rgb=True),
    dict(type='RandomFlip', flip_ratio=0.5),
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
            dict(type='Resize', keep_ratio=False),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
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
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=[123.68, 116.78, 103.94],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=640),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1_full.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/0517/annotations/split1_full/dataset/new/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=[123.68, 116.78, 103.94],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=640),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=[123.68, 116.78, 103.94],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=640),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='RandomFlip', flip_ratio=0.5),
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
        '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp3.json',
        img_prefix=
        '/share/yanzhen/tools/internal-data/0517/annotations/split3/dataset/new/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(
                type='Expand',
                mean=[123.68, 116.78, 103.94],
                to_rgb=True,
                ratio_range=(1, 4)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size_divisor=640),
            dict(
                type='Normalize',
                mean=[123.68, 116.78, 103.94],
                std=[58.4, 57.12, 57.38],
                to_rgb=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=[
        dict(
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
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='Expand',
                    mean=[123.68, 116.78, 103.94],
                    to_rgb=True,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
                dict(type='Pad', size_divisor=640),
                dict(
                    type='Normalize',
                    mean=[123.68, 116.78, 103.94],
                    std=[58.4, 57.12, 57.38],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.5),
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
            '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1_full.json',
            img_prefix=
            '/share/yanzhen/tools/internal-data/0517/annotations/split1_full/dataset/new/',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='Expand',
                    mean=[123.68, 116.78, 103.94],
                    to_rgb=True,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
                dict(type='Pad', size_divisor=640),
                dict(
                    type='Normalize',
                    mean=[123.68, 116.78, 103.94],
                    std=[58.4, 57.12, 57.38],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.5),
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
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='Expand',
                    mean=[123.68, 116.78, 103.94],
                    to_rgb=True,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
                dict(type='Pad', size_divisor=640),
                dict(
                    type='Normalize',
                    mean=[123.68, 116.78, 103.94],
                    std=[58.4, 57.12, 57.38],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.5),
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
            '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp3.json',
            img_prefix=
            '/share/yanzhen/tools/internal-data/0517/annotations/split3/dataset/new/',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
                dict(
                    type='PhotoMetricDistortion',
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                dict(
                    type='Expand',
                    mean=[123.68, 116.78, 103.94],
                    to_rgb=True,
                    ratio_range=(1, 4)),
                dict(
                    type='MinIoURandomCrop',
                    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
                    min_crop_size=0.3),
                dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
                dict(type='Pad', size_divisor=640),
                dict(
                    type='Normalize',
                    mean=[123.68, 116.78, 103.94],
                    std=[58.4, 57.12, 57.38],
                    to_rgb=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
            ])
    ],
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
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
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
                    dict(type='Resize', keep_ratio=False),
                    dict(
                        type='Normalize',
                        mean=[123.68, 116.78, 103.94],
                        std=[58.4, 57.12, 57.38],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=55)
cudnn_benchmark = True
evaluation = dict(metric=['bbox', 'segm'], classwise=True)
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[20, 42, 49, 52])
load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/yolact_r50_1x8_coco_20200908-f38d58df.pth'
resume_from = None
work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/yolact_r50_coco_0512'
gpu_ids = range(0, 8)
