import fnmatch
import glob
import os

import cv2
import xml.etree.ElementTree as ET

files = glob.glob('test/bbox/*.jpg')
for f in files:
    os.remove(f)

for root, dirs, files in os.walk('test/sample'):
    image_file = [[os.path.join(root, name), name] for name in files if fnmatch.fnmatch(name, '*.jpg')]

for root, dirs, files in os.walk('test/sample'):
    xml_file = [[os.path.join(root, name), name] for name in files if fnmatch.fnmatch(name, '*.xml')]

for image_path, image_name in image_file:
    image_name = image_name.replace('.jpg', '')
    img = cv2.imread(image_path)
    for xml_path, xml_name in xml_file:
        xml_name = xml_name.replace('.xml', '')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        sample_annotations = []
        name_annotation = []
        if xml_name == image_name:
            for neighbor in root.iter('object'):
                name = neighbor.find('name').text
                name_annotation.append(name)
            for neighbor in root.iter('bndbox'):
                xmin = int(neighbor.find('xmin').text)
                ymin = int(neighbor.find('ymin').text)
                xmax = int(neighbor.find('xmax').text)
                ymax = int(neighbor.find('ymax').text)
                sample_annotations.append([xmin, ymin, xmax, ymax])

            for x, bbox in enumerate(sample_annotations):
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(img,name_annotation[x],(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            path = "test/bbox/{}.jpg".format(image_name)
            cv2.imwrite(path, img)
