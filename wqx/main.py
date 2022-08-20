from xsma.generel_utils.tool_json import load_json, write_json
import numpy as np
import json
from rich import print
import pycocotools.mask as maskUtils
import cv2
import os
from tqdm import tqdm


def load_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    if isinstance(data, str):
        data = eval(data)
    return data




def f1(raw_json_file=None, img_dir=None, new_json_file=None, mode='coco80'):
    if img_dir is None:
        img_dir = '/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/dataset/new'

    # raw_json_file = '/share/data/coco/annotations/instances_train2017.json'
    if new_json_file is None:
        new_json_file = raw_json_file.replace('.json', '_coco80.json')

    raw_json = load_json(raw_json_file)
    print(raw_json.keys())
    print(raw_json['categories'])
    
    new_json = {}
    for k in ['info', 'licenses']:
        if k not in raw_json:
            continue
        new_json[k] = raw_json[k]

    new_json['images'] = []
    for img_info in raw_json['images']:
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        # img = cv2.imread(f"{img_dir}/{file_name}")
        # h,w = img.shape[:2]
        img_info_new = {
            'file_name': file_name,
            # 'height': h,
            # 'width': w,
            'height': height,
            'width': width,
            'id': img_info['id']
        }
        new_json['images'].append(img_info_new)

    if mode=='coco80':
        coco_name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    elif mode=='xhs9':
        coco_name_list = ['person', 'bicycle', 'motorcycle', 'frisbee',
                'snowboard', 'sports ball', 'baseball bat',
                'skateboard', 'tennis racket',
                ]
    new_json['categories'] = []
    mapping_old_to_new = {}
    idx_new = 0
    for idx in range(len(raw_json['categories'])):
        supercategory = raw_json['categories'][idx]['supercategory']
        id_raw = raw_json['categories'][idx]['id']
        name = raw_json['categories'][idx]['name']
        if name in coco_name_list:
            idx_new += 1
            cate_info = {
                'supercategory': supercategory,
                'id': idx_new,
                'name': name
            }
            mapping_old_to_new[id_raw] = idx_new
            new_json['categories'].append(cate_info)
    # print(new_json['categories'])
    # print(mapping_old_to_new)


    new_json['annotations'] = []
    for idx, ann_info in enumerate(raw_json['annotations']):
        if idx == 0:
            print(ann_info)
        segmentation = ann_info['segmentation']
        area = ann_info['area']
        iscrowd = ann_info['iscrowd']
        image_id = ann_info['image_id']
        bbox = ann_info['bbox']
        category_id_old = ann_info['category_id']
        id_raw = ann_info['id']
        if category_id_old not in mapping_old_to_new:
            continue
        category_id_new = mapping_old_to_new[category_id_old]

        if not isinstance(segmentation['counts'], list):
            segmentation['counts'] = str(segmentation['counts'])
        ann_info_new = {
            'segmentation': segmentation,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id_new,
            'id': id_raw
        }
        new_json['annotations'].append(ann_info_new)
    # print(new_json)
    write_json(new_json, new_json_file)


def f1_del_person(raw_json_file=None, img_dir=None, new_img_dir=None, new_json_file=None, mode='coco80'):
    raw_json = load_json(raw_json_file)
    print(raw_json.keys())
    print(raw_json['categories'])

    os.makedirs(new_img_dir, exist_ok=True)
    
    new_json = {}
    for k in ['info', 'licenses']:
        if k not in raw_json:
            continue
        new_json[k] = raw_json[k]

    img_id_list_exist_ann = []
    new_json['images'] = []
    for img_info in tqdm(raw_json['images']):
        file_name = img_info['file_name']
        img = cv2.imread(f"{img_dir}/{file_name}")
        new_json['images'].append(img_info)

        nb_ann = 0
        ann_list = []
        for ann_info in raw_json['annotations']:
            if (ann_info['image_id'] == img_info['id']) and ann_info['category_id'] == 1:
                segmentation_raw = ann_info['segmentation']
                h, w = segmentation_raw['size']
                rle = maskUtils.frPyObjects(segmentation_raw, h, w)
                mask = maskUtils.decode(rle) > 0.5
                img[mask] = 0
            elif (ann_info['image_id'] == img_info['id']) and ann_info['category_id'] != 1:
                nb_ann += 1
                ann_list.append(ann_info['category_id'])
        
        if nb_ann > 0:
            # print(nb_ann, file_name, ann_list)
            img_id_list_exist_ann.append(img_info['id'])
            cv2.imwrite(f"{new_img_dir}/{file_name}", img)



    if mode=='coco80':
        coco_name_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    elif mode=='xhs9':
        coco_name_list = ['person', 'bicycle', 'motorcycle', 'frisbee',
                'snowboard', 'sports ball', 'baseball bat',
                'skateboard', 'tennis racket',
        ]
    new_json['categories'] = []
    mapping_old_to_new = {}
    idx_new = 0
    for idx in range(len(raw_json['categories'])):
        supercategory = raw_json['categories'][idx]['supercategory']
        id_raw = raw_json['categories'][idx]['id']
        name = raw_json['categories'][idx]['name']
        if name in coco_name_list:
            idx_new += 1
            cate_info = {
                'supercategory': supercategory,
                'id': idx_new,
                'name': name
            }
            mapping_old_to_new[id_raw] = idx_new
            new_json['categories'].append(cate_info)


    new_json['annotations'] = []
    for idx, ann_info in enumerate(raw_json['annotations']):
        if idx == 0:
            print(ann_info)
        segmentation = ann_info['segmentation']
        area = ann_info['area']
        iscrowd = ann_info['iscrowd']
        image_id = ann_info['image_id']
        bbox = ann_info['bbox']
        category_id_old = ann_info['category_id']
        id_raw = ann_info['id']
        if category_id_old not in mapping_old_to_new:
            continue
        if category_id_old == 1:
            continue
        if image_id not in img_id_list_exist_ann:
            continue
        category_id_new = mapping_old_to_new[category_id_old]

        if not isinstance(segmentation['counts'], list):
            segmentation['counts'] = str(segmentation['counts'])
        ann_info_new = {
            'segmentation': segmentation,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id_new,
            'id': id_raw
        }
        new_json['annotations'].append(ann_info_new)
    # print(new_json)
    write_json(new_json, new_json_file)





def f2(raw_json_file=None, new_json_file=None):
    if new_json_file is None:
        new_json_file = raw_json_file.replace('.json', '_polygon.json')

    raw_json = load_json(raw_json_file)
    print(raw_json.keys())
    print(raw_json['categories'])

    temp_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/test/imgs/'
    os.makedirs(temp_dir, exist_ok=True)

    new_json = {}
    for k in ['info', 'licenses', 'images', 'categories']:
        if k not in raw_json:
            continue
        new_json[k] = raw_json[k]


    new_json['annotations'] = []
    for idx, ann_info in enumerate(raw_json['annotations']):
        if idx == 0:
            pass
            # print(ann_info)
        segmentation_raw = ann_info['segmentation']
        area = ann_info['area']
        iscrowd = ann_info['iscrowd']
        image_id = ann_info['image_id']
        bbox = ann_info['bbox']
        category_id = ann_info['category_id']
        id = ann_info['id']

        segmentation = []
        h, w = segmentation_raw['size']
        rle = maskUtils.frPyObjects(segmentation_raw, h, w)
        mask = maskUtils.decode(rle)
        contours, hierarchy = cv2.findContours(mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) <= 2:
                continue
            contour = contour.reshape([-1, ]).tolist()
            segmentation.append(contour)
        if len(segmentation) == 0:
            continue
        
        segmentation_all = []
        for s in segmentation:
            segmentation_all += s
        segmentation_all = np.array(segmentation_all).reshape([-1, 2])
        x1 = segmentation_all[:, 0].min()
        y1 = segmentation_all[:, 1].min()
        x2 = segmentation_all[:, 0].max()
        y2 = segmentation_all[:, 1].max()
        if x1 <0 or y1<0 or x2 >= w or y2>= h:
            print('error', idx)

        bbox = [
            float(np.clip(x1, 0, w-1)),
            float(np.clip(y1, 0, h-1)),
            float(np.clip(x2-x1, 0, w-1)),
            float(np.clip(y2-y1, 0, h-1)),
        ]

        if idx <= 100:
            mask *= 255
            mask = np.ascontiguousarray(mask)            
            left_top = (int(bbox[0]), int(bbox[1]))
            right_bottom = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
            mask = cv2.rectangle(mask, left_top, right_bottom, (128, 128, 128), 20)
            cv2.imwrite(f'{temp_dir}/{idx}.jpg', mask)

            # print(mask.shape)
            # aaaaaa

        ann_info_new = {
            'segmentation': segmentation,
            # 'segmentation': segmentation_raw,
            'area': area,
            'iscrowd': iscrowd,
            'image_id': image_id,
            'bbox': bbox,
            'category_id': category_id,
            'id': id
        }

        new_json['annotations'].append(ann_info_new)
    write_json(new_json, new_json_file)


def check_and_vis_json(json_file, img_dir, mode='instance'):
    raw_json = load_json(json_file)
    print(raw_json.keys())
    print(raw_json['categories'])

    temp_dir = '/share/wangqixun/workspace/github_project/CBNetV2_train/wqx/test/imgs'
    os.makedirs(temp_dir, exist_ok=True)

    if mode=='instance':
        nb = 0
        for idx, ann_info in enumerate(raw_json['annotations']):
            if idx == 0:
                pass
                # print(ann_info)
            segmentation = ann_info['segmentation']
            area = ann_info['area']
            iscrowd = ann_info['iscrowd']
            image_id = ann_info['image_id']
            bbox = ann_info['bbox']
            category_id = ann_info['category_id']
            id = ann_info['id']

            for idx_img, img_info in enumerate(raw_json['images']):
                if img_info['id'] == image_id:
                    img_file = os.path.join(img_dir, img_info['file_name'])
                    img = cv2.imread(img_file)
                    break

            h, w = segmentation['size']
            rle = maskUtils.frPyObjects(segmentation, h, w)
            mask = maskUtils.decode(rle) > 0
            # mask = (mask * 255).astype(np.uint8)
            # mask = np.concatenate([mask[..., None]]*3, axis=-1)

            color = np.array([256, 64, 128]).reshape([1, 1, 3])

            left_top = (int(bbox[0]), int(bbox[1]))
            right_bottom = (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3]))
            img = cv2.rectangle(img.copy(), left_top, right_bottom, (255, 64, 128), 2)
            img[mask] = 0.5 * img[mask] + 0.5*color

            cv2.imwrite(f'{temp_dir}/{idx}.jpg', img)
            nb += 1
            if nb > 1000:
                return
    elif mode=='img':
        nb = 0
        for idx_img, img_info in enumerate(raw_json['images']):
            img_id = img_info['id']
            img_file = os.path.join(img_dir, img_info['file_name'])
            img = cv2.imread(img_file)





if __name__ == '__main__':
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/businessDataset/testSet/annotations/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_test_xhs9.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0425/0512/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0425/0512/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0512.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0517/annotations/split1/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0517/annotations/split1/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0517/annotations/split2/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0517/annotations/split2/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp2.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0517/annotations/split1_full/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0517/annotations/split1_full/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1_full.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0517/annotations/split3/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0517/annotations/split3/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp3.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0608/annotations/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0608/annotations/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0608sp1.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0625/annotations/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0625/annotations/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0625sp1.json',
    #     mode='xhs9',
    # )
    # f1(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0628/annotations/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0628/annotations/dataset/new',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0628sp3.json',
    #     mode='xhs9',
    # )







    # f1_del_person(
    #     raw_json_file='/share/yanzhen/tools/internal-data/0517/annotations/split1_full/annotationInCOCOReformat.json',
    #     img_dir='/share/yanzhen/tools/internal-data/0517/annotations/split1_full/dataset/new',
    #     new_img_dir='/share/wangqixun/workspace/github_project/CBNetV2_train/data/images/train_xhs9_0517sp1_full_wo_person',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs9_0517sp1_full_wo_person.json',
    #     mode='xhs9',
    # )
    # f1(
    #     '/share/yanzhen/tools/internal-data/0425/0512/annotationInCOCOReformat_val.json',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_val_xhs11.json'
    # )

    # f2(
    #     '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs11.json',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_train_xhs11_polygon.json'
    # )
    # f2(
    #     '/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_val_xhs11.json',
    #     new_json_file='/share/wangqixun/workspace/github_project/CBNetV2_train/data/annotationInCOCOReformat_val_xhs11_polygon.json'
    # )


    check_and_vis_json(
        # json_file='/share/yanzhen/tools/internal-data/0425/0512/annotationInCOCOReformat.json',
        # img_dir='/share/yanzhen/tools/internal-data/0425/0512/dataset/new',
        json_file='/share/yanzhen/tools/internal-data/0628/annotations/annotationInCOCOReformat.json',
        img_dir='/share/yanzhen/tools/internal-data/0628/annotations/dataset/new',
    )
















