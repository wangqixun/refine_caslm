_base_ = [
    '../swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py'
]
find_unused_parameters=True

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN',
        upsample_cfg=dict(mode='bilinear'),
    ),
    roi_head=dict(
        type='RefineRoIHead',
        bbox_roi_extractor=dict(
            roi_layer=dict(type='ModulatedDeformRoIPoolPack', output_size=7, output_channels=256),
        ),
        mask_roi_extractor=dict(
            roi_layer=dict(type='ModulatedDeformRoIPoolPack', output_size=14, output_channels=256),
        ),
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
                 loss_cfg=dict(
                    type='RefineCrossEntropyLoss',
                    stage_instance_loss_weight=[0.25, 0.5, 0.75, 1.0],
                    semantic_loss_weight=1.0,
                    boundary_width=2,
                    start_stage=1)
        ),    ),
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                num=128,
            ),
        ),
    ),
)



evaluation = dict(metric=['bbox', 'segm'], classwise=True)

# fp16 = None
optimizer_config = dict(
    use_fp16=False)


# sit
# load_from = '/share/wangqixun/workspace/github_project/mmdetection/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
# resume_from = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco/latest.pth'
# work_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/u2mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco'


# 平台
load_from = '/mnt/mmtech01/usr/guiwan/workspace/model_dl/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth'
# resume_from = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/refine_mask_rcnn_cbv2_swin_tiny_coco80_v3/latest.pth'
resume_from = None
work_dir = '/mnt/mmtech01/usr/guiwan/workspace/instance_segm/wqx/refine_mask_rcnn_cbv2_swin_tiny_coco80_dmp_v5'





