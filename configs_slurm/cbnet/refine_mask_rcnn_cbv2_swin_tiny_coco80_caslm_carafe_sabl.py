_base_ = [
    '../swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
]
find_unused_parameters=True

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN_CARAFE',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='carafe',
            up_kernel=5,
            up_group=1,
            encoder_kernel=3,
            encoder_dilation=1,
            compressed_channels=64),
    ),
    roi_head=dict(
        type='CascadeLastMaskRefineRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_head=[
            dict(
                type='SABLHead',
                num_classes=80,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                                loss_weight=1.0)
            ),            dict(
                type='SABLHead',
                num_classes=80,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                                loss_weight=1.0)
            ),            dict(
                type='SABLHead',
                num_classes=80,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                                loss_weight=1.0)
            ),
            # dict(
            #     type='Shared2FCBBoxHead',
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=80,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=[0., 0., 0., 0.],
            #         target_stds=[0.1, 0.1, 0.2, 0.2]),
            #     reg_class_agnostic=False,
            #     loss_cls=dict(
            #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            # ),
            # dict(
            #     type='Shared2FCBBoxHead',
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=80,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=[0., 0., 0., 0.],
            #         target_stds=[0.05, 0.05, 0.1, 0.1]),
            #     reg_class_agnostic=False,
            #     loss_cls=dict(
            #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            # ),
            # dict(
            #     type='Shared2FCBBoxHead',
            #     in_channels=256,
            #     fc_out_channels=1024,
            #     roi_feat_size=7,
            #     num_classes=80,
            #     bbox_coder=dict(
            #         type='DeltaXYWHBBoxCoder',
            #         target_means=[0., 0., 0., 0.],
            #         target_stds=[0.033, 0.033, 0.067, 0.067]),
            #     reg_class_agnostic=False,
            #     loss_cls=dict(
            #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            #     loss_bbox=dict(type='L1Loss', loss_weight=1.0)
            # ),
        ],

        mask_head=dict(
            _delete_=True,
            type='RefineMaskHead',
                 num_convs_instance=2,
                 num_convs_semantic=4,
                 conv_in_channels_instance=256,
                 conv_in_channels_semantic=256,
                 conv_kernel_size_instance=3,
                 conv_kernel_size_semantic=3,
                 conv_out_channels_instance=256,
                 conv_out_channels_semantic=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 dilations=[1, 3, 5],
                 semantic_out_stride=4,
                 mask_use_sigmoid=True,
                 stage_num_classes=[80, 80, 80, 80],
                 stage_sup_size=[14, 28, 56, 112],
                 upsample_cfg=dict(type='bilinear', scale_factor=2),
                #  upsample_cfg=dict(
                #     type='carafe',
                #     scale_factor=2,
                #     up_kernel=5,
                #     up_group=1,
                #     encoder_kernel=3,
                #     encoder_dilation=1,
                #     compressed_channels=64
                # ),
                 loss_cfg=dict(
                    type='RefineCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)
        ),    ),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    gpu_assign_thr=8,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    gpu_assign_thr=8,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    gpu_assign_thr=8,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ],
        # rcnn=dict(
        #     sampler=dict(
        #         num=384,
        #     ),
        # ),
    ),
)



data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)


evaluation = dict(metric=['bbox', 'segm'], classwise=True)

# fp16 = None
# optimizer_config = dict(
#     grad_clip=dict(max_norm=35, norm_type=2),
#     type='DistOptimizerHook',
#     update_interval=1,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

# fp16 = dict(_delete_=True, loss_scale=128.)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2),)

# sit
# load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
# resume_from = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'
# work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco'


# 平台
load_from = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/refine_mask_rcnn_cbv2_swin_tiny_coco80_0512_0517sp1_full_0517sp2_0517sp3_giou_ohem_bigsize_coloraug_copypaste_sablhead/combination.pth'
# resume_from = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/refine_mask_rcnn_cbv2_swin_tiny_coco80_caslm_carafe_sabl/latest.pth'
resume_from = None
work_dir = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/refine_mask_rcnn_cbv2_swin_tiny_coco80_caslm_carafe_sabl'





