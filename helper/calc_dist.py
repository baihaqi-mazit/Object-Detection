import xml.etree.ElementTree as ET
import os
from tqdm import tqdm
import pprint
import json
import csv
import matplotlib.pyplot as plt

def get_class_name(xml_file, class_names_19_class):
    try:
        root = ET.parse(xml_file).getroot()
    except Exception as e:
        print(e)
        return None

    for elem in root:
        if elem.tag == 'object':
            for subelem in elem:
                if subelem.tag == 'name':
                    if subelem.text.lower() in class_names_19_class:
                        class_name = subelem.text
                    else:
                        class_name = 'Others'
                    return class_name
    return None

if __name__ == '__main__':
    folder_path = 'data/image'

    class_names_19_class_txtfile = 'data/logo_labels_19class.txt'
    with open(class_names_19_class_txtfile) as f:
        content = f.readlines()
        class_names_19_class = [x.strip().lower() for x in content]

    for dataset in ['train', 'valid', 'test']:
        class_dict = {}
        class_count_dict = {}
        ctr = 0

        for filename in tqdm([os.path.join(folder_path, dataset, f) for f in os.listdir(os.path.join(folder_path, dataset))]):
            if filename.endswith('.xml'):
                ctr+=1

                class_name = get_class_name(filename, class_names_19_class)

                try:
                    # class_dict[class_name].append(filename)
                    class_count_dict[class_name] += 1
                except:
                    # class_dict[class_name] = filename
                    class_count_dict[class_name] = 1

        pprint.pprint(class_count_dict, width = 4)
        print('\nTotal:', ctr)
        print()

        with open('data/dataset.json', 'w') as fp:
            json.dump(class_count_dict, fp)

        # with open('data/dataset.csv', 'w') as f:
        #     for key in class_count_dict.keys():
        #         f.write("%s,%s\n"%(key,class_count_dict[key]))

        fig, axes = plt.subplots(figsize=(7,5), dpi=100)
        plt.bar(class_count_dict.keys(), height=class_count_dict.values())
        plt.xticks(rotation=90)
        plt.title('Barplot of Class Distribution')
        plt.show()