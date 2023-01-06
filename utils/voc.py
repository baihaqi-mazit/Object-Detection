import sys

sys.path.append(".")
import os
import shutil
import xml.etree.ElementTree as ET

import config as cfg
import requests
from tqdm import tqdm

img_ext = ['.jpeg', '.jpg', '.png']


def main(class_names, URL=False, URLs=[]):
    len_train = 0
    len_test = 0
    
    train_data_path = cfg.TRAIN_DATA_PATH if not URLs else URLs[0]
    train_annotation_path = os.path.join('data', 'train_annotation.txt')
    if os.path.exists(train_annotation_path):
        os.remove(train_annotation_path)

    valid_data_path = cfg.VALID_DATA_PATH if not URLs else URLs[1]
    valid_annotation_path = os.path.join('data', 'valid_annotation.txt')
    if os.path.exists(valid_annotation_path):
        os.remove(valid_annotation_path)

    test_data_path = cfg.TEST_DATA_PATH if not URLs else URLs[2]
    test_annotation_path = os.path.join('data', 'test_annotation.txt')
    if os.path.exists(test_annotation_path):
        os.remove(test_annotation_path)

    

    if URL:
        len_train = parse_voc_annotation_URL(train_data_path, train_annotation_path, class_names, use_difficult_bbox=False, URLs=URLs)
        len_test  = parse_voc_annotation_URL(test_data_path, test_annotation_path, class_names, use_difficult_bbox=False, URLs=URLs)
        len_valid = parse_voc_annotation_URL(valid_data_path, valid_annotation_path, class_names, use_difficult_bbox=False, URLs=URLs)
    else:
        len_train = parse_voc_annotation(train_data_path, train_annotation_path, class_names, use_difficult_bbox=False)
        len_test  = parse_voc_annotation(test_data_path, test_annotation_path, class_names, use_difficult_bbox=False)
        len_valid = parse_voc_annotation(valid_data_path, valid_annotation_path, class_names, use_difficult_bbox=False)

    print("The number of images for train and test are :train : {0} | valid : {1} | test : {2}".format(len_train, len_valid, len_test))

def parse_voc_annotation(data_path, anno_path, class_names, use_difficult_bbox=False):
    # classes = cfg.DATA["CLASSES"]
    classes = class_names if class_names else cfg.DATA["CLASSES"]
    classes = list(map(str.lower, classes))

    image_ids = [f for f in os.listdir(data_path) if os.path.splitext(f)[-1].lower() in img_ext]

    annotation = ''
    with open(anno_path, 'a') as f:
        print()
        print('Parsing XML files...')
        for image_id in tqdm(image_ids):
            image_path = os.path.join(data_path, image_id)
            annotation = image_path
            label_path = os.path.join(data_path, os.path.splitext(image_id)[0] +'.xml')
            try:
                root = ET.parse(label_path).getroot()
            except:
                print(label_path)
                continue
            objects = root.findall('object')

            if not objects: continue            # skip file if no label

            has_bbox = False
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                try:
                    class_id = classes.index(obj.find("name").text.lower().strip())
                except ValueError:
                    class_id = classes.index('others')
                    # print(obj.find("name").text.lower().strip(), image_path)
                    # try:
                    #     shutil.move(image_path, os.path.join('data', 'image', 'removed', image_id))
                    #     shutil.move(image_path.replace('.jpg', '.xml'), os.path.join('data', 'image', 'removed', image_id.replace('.jpg', '.xml')))
                    # except FileNotFoundError:
                    #     continue
                    # continue
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()

                if int(xmin) < 0:
                    xmin = '0'
                    has_bbox = False
                    continue
                if int(ymin) < 0:
                    ymin = '0'
                    has_bbox = False
                    continue
                if int(xmax) < 0:
                    xmax = '0'
                    has_bbox = False
                    continue
                if int(ymax) < 0:
                    ymax = '0'
                    has_bbox = False
                    continue

                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
                
                has_bbox = True
            
            if has_bbox:
                annotation += '\n'
                f.write(annotation)
    return len(image_ids)

def parse_voc_annotation_URL(data_path, anno_path, class_names, use_difficult_bbox=False, URLs=[]):
    # classes = cfg.DATA["CLASSES"]
    classes = class_names if class_names else cfg.DATA["CLASSES"]
    classes = list(map(str.lower, classes))

    if URLs:
        image_ids = [f[0] for f in data_path]
        label_ids = [f[1] for f in data_path]
        
        data_path = ''
    else:
        image_ids = [f for f in os.listdir(data_path) if os.path.splitext(f)[-1].lower() in img_ext]

    annotation = ''
    with open(anno_path, 'a') as f:
        print()
        print('Parsing XML files...')
        for i, image_id in tqdm(enumerate(image_ids)):
            image_path = os.path.join(data_path, image_id) if not URLs else image_id
            annotation = image_path
            label_path = os.path.join(data_path, os.path.splitext(image_id)[0] +'.xml') if not URLs else label_ids[i]
            
            if 'http' in label_path:
                req = requests.get(label_path, stream=True)
                req.raw.decode_content = True  # ensure transfer encoding is honoured
                label_path = req.raw

            try:
                root = ET.parse(label_path).getroot()
            except:
                print(label_path)
                continue
            objects = root.findall('object')

            if not objects: continue            # skip file if no label

            has_bbox = False
            for obj in objects:
                difficult = obj.find("difficult").text.strip()
                if (not use_difficult_bbox) and (int(difficult) == 1):
                    continue
                bbox = obj.find('bndbox')
                try:
                    class_id = classes.index(obj.find("name").text.lower().strip())
                except ValueError:
                    class_id = classes.index('others')
                    # print(obj.find("name").text.lower().strip(), image_path)
                    # try:
                    #     shutil.move(image_path, os.path.join('data', 'image', 'removed', image_id))
                    #     shutil.move(image_path.replace('.jpg', '.xml'), os.path.join('data', 'image', 'removed', image_id.replace('.jpg', '.xml')))
                    # except FileNotFoundError:
                    #     continue
                    # continue
                xmin = bbox.find('xmin').text.strip()
                ymin = bbox.find('ymin').text.strip()
                xmax = bbox.find('xmax').text.strip()
                ymax = bbox.find('ymax').text.strip()

                if int(xmin) < 0:
                    xmin = '0'
                    has_bbox = False
                    continue
                if int(ymin) < 0:
                    ymin = '0'
                    has_bbox = False
                    continue
                if int(xmax) < 0:
                    xmax = '0'
                    has_bbox = False
                    continue
                if int(ymax) < 0:
                    ymax = '0'
                    has_bbox = False
                    continue

                annotation += ' ' + ','.join([xmin, ymin, xmax, ymax, str(class_id)])
                
                has_bbox = True
            
            if has_bbox:
                annotation += '\n'
                f.write(annotation)
    return len(image_ids)


if __name__ =="__main__":
    main()
