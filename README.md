# Refine Cascade-Last-Mask RCNN


结合了[refin-mask](https://github.com/zhanggang001/RefineMask)、[CBNetv2](https://github.com/VDIGPKU/CBNetV2)，整体框架还是基于[mmdet](https://github.com/open-mmlab/mmdetection)





## checkpoint
|model |train|val| bbox map | segm map | checkpoint|
|:--- | :-----: |:-----: |:-----: | ----: | :----:|
|cbv2_swimtiny_mask_rcnn|coco train|coco minival|50.2|44.5|[链接](https://github.com/CBNetwork/storage/releases/download/v1.0.0/mask_rcnn_cbv2_swin_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.pth.zip)(from [repo](https://github.com/VDIGPKU/CBNetV2))|
|refine_cbv2_swimtiny_mask_rcnn|coco train|coco val|50.7|46.5|[链接](https://cloud.189.cn/t/iMbINfRRfER3)(访问码:fj4k)|
|refine_cbv2_swimtiny_cascade-last-mask_rcnn |coco train|coco val| 52.8 |46.8| [链接](https://cloud.189.cn/t/BJBZjanuERR3)(访问码:qr0n)

<br>

## 安装环境
安装参考 [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md)

此外，还需要安装apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

<br>


## 训练

###### refine_cbv2_swimtiny_mask_rcnn
```
bash tools/dist_train.sh configs/refine/refine_mask_rcnn_cbv2_swin_tiny_coco80.py 8 
```

###### refine_cbv2_swimtiny_cascade-last-mask_rcnn
```
bash tools/dist_train.sh configs/refine/refine_mask_rcnn_cbv2_swin_tiny_coco80_caslm.py 8 
```


<br>


#### 有事没事常联系
```
	🆔 挠挠
	☎️ 18611684528
	📮 253817124@qq.com
	
```