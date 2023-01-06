import argparse
import os
import shutil
import sys

import requests

sys.path.append(".")

import re
import xml.etree.ElementTree as ET

import numpy as np
import utils.voc as voc
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from helper.rename_file import remove_underscore

image_ext = ['.png', '.jpeg', '.jpg', '.tif']


def rename_xml_files(path_to_file, filename):
    for ext in image_ext:
        if ext.replace('.', '_') in filename:
            renamed_filename = re.sub(ext.replace('.', '_'), '', filename)
            break
        else:
            renamed_filename = filename

    try:
        os.rename(os.path.join(path_to_file, filename), os.path.join(path_to_file, renamed_filename))
    except Exception as e:
        print(e)

    return renamed_filename


def merge_files_in_folders(folder_path):
    classes = []
    image_files = {}
    annot_files = {}
    total_image = 0
    for folder in [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]:
        txt_files = []
        xml_files = []
        png_files = []
        for filename in os.listdir(os.path.join(folder_path, folder)):
            if (filename.endswith('.txt')) and ('_label' in filename):
                with open(os.path.join(folder_path, folder, filename)) as txtfile:
                    content = [("'" + line.strip() + "'") for line in txtfile.readlines()]
                    classes = classes + content

            elif filename.endswith('.xml'):
                if '_png' in filename:
                    filename = rename_xml_files(os.path.join(folder_path, folder), filename)
                xml_files.append(os.path.join(folder_path, folder, filename))
                png_files.append(os.path.join(folder_path, folder, filename.replace('.xml', '.png')))

                image_files[folder] = png_files
                annot_files[folder] = xml_files
            
            elif filename.endswith('.txt'):
                with open(os.path.join(folder_path, folder, filename)) as txtfile:
                    content = [line.split()[0].strip() for line in txtfile.readlines()]
                    classes = classes + content

                txt_files.append(os.path.join(folder_path, folder, filename))
                png_files.append(os.path.join(folder_path, folder, filename.replace('.txt', '.png')))

                image_files[folder] = png_files
                annot_files[folder] = txt_files
        
        total_image = total_image + len(image_files[folder])
        print('{:30s}: {} images'.format(folder, len(png_files)))
    
    classes = sorted(list(set(classes)))

    print()
    print('Number of classes:', len(classes))
    print('Number of files  :', total_image)
    print()

    with open('data/label.txt', 'w') as f:
        f.write(',\n'.join(classes))

    return image_files


def get_urls(req):
    return {item['fileName']:item['url'] for item in req['trainingFile']}


def get_files(label_format, folder_path, classes, req):
    label_files = []
    image_files = []
    pair_urls = {}
    
    if req is not None:
        folder_path = ''

        pair_urls = get_urls(req)
        urls = list(pair_urls.keys())
        image_urls = [f for f in urls if os.path.splitext(f)[-1].lower() in image_ext]
        label_urls = [f for f in urls if os.path.splitext(f)[-1].lower() == ('.' + label_format.lower())]

        filenames = label_urls
    else:
        filenames = os.listdir(folder_path)
    
    for filename in filenames:
        if ' ' in filename:
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, filename.replace(' ', '_')))
            filename = filename.replace(' ', '_')

        if (filename.endswith('.txt')) and ('_label' in filename) and len(classes) < 1:
            with open(os.path.join(folder_path, filename)) as txtfile:
                content = [("'" + line.strip() + "'") for line in txtfile.readlines()]
                classes = classes + content

        elif filename.lower().endswith('.xml') and label_format == 'XML':
            file_ext = None
            for ext in ['_png', '_jpg', '_jpeg']:
                if ext in filename:
                    filename = rename_xml_files(folder_path, filename)
                    file_ext = ext
                    break
            
            if file_ext is None:
                for ext in image_ext:
                    if req is not None:
                        if filename.replace('.xml', ext) in image_urls:
                            file_ext = ext
                            break
                    else:
                        if os.path.exists(os.path.join(folder_path, filename.replace('.xml', ext))):
                            file_ext = ext
                            break
            
            if req is not None:
                has_label = os.path.join(folder_path, filename) in urls
                has_image = os.path.join(folder_path, filename.replace('.xml', file_ext.replace('_', '.'))) in urls if file_ext is not None else False
            else:
                has_label = os.path.exists(os.path.join(folder_path, filename))
                has_image = os.path.exists(os.path.join(folder_path, filename.replace('.xml', file_ext.replace('_', '.')))) if file_ext is not None else False
            
            if has_label and has_image:
                classes = classes + get_class_name(os.path.join(folder_path, filename), classes, pair_urls)

                if pair_urls:
                    label_files.append(pair_urls[filename])
                    image_files.append(pair_urls[filename.replace('.xml', file_ext.replace('_', '.'))])
                else:
                    label_files.append(os.path.join(folder_path, filename))
                    image_files.append(os.path.join(folder_path, filename.replace('.xml', file_ext.replace('_', '.'))))

        elif filename.lower().endswith('.txt') and label_format == 'TXT':
            file_ext = None
            for ext in ['_png', '_jpg', '_jpeg']:
                if ext in filename:
                    filename = rename_xml_files(os.path.join(folder_path), filename)
                    file_ext = ext
                    break
            
            if file_ext is None:
                for ext in image_ext:
                    if os.path.exists(os.path.join(folder_path, filename.replace('.txt', ext))):
                        file_ext = ext
                        break
            
            has_label = os.path.exists(os.path.join(folder_path, filename))
            has_image = os.path.exists(os.path.join(folder_path, filename.replace('.txt', file_ext.replace('_', '.')))) if file_ext is not None else False
            
            if has_label and has_image:
                with open(os.path.join(folder_path, filename)) as txtfile:
                    content = [line.split()[0].strip() for line in txtfile.readlines()]
                    classes = classes + content
                    # print(content)

                label_files.append(os.path.join(folder_path, filename))
                image_files.append(os.path.join(folder_path, filename.replace('.txt', file_ext.replace('_', '.'))))

        # print('Total files: {} images'.format(len(jpg_files)))
    
    classes = sorted(list(set(classes)))

    print()
    print('Number of classes:', len(classes))
    print()
    # print('Image files:', image_files)
    # print('XML files  :', label_files)

    with open('data/output_label.txt', 'w') as f:
        f.write('\n'.join(classes))

    image_files = {0:image_files}

    return image_files, label_files


def get_class_name(xml_file, classes, pair_urls={}):
    class_names = []

    if pair_urls:
        xml_url = pair_urls[xml_file]
        req = requests.get(xml_url, stream=True)
        req.raw.decode_content = True  # ensure transfer encoding is honoured

        try:
            root = ET.parse(req.raw).getroot()
        except:
            print('Unable to extract from XML: {}({})'.format(xml_file, xml_url))
        objects = root.findall('object')
        for obj in objects:
            class_name = obj.find("name").text.lower().strip()
            if class_name not in classes:
                class_name = 'others'
            class_names.append(class_name)
    else:
        with open(xml_file, 'a') as f:
            try:
                root = ET.parse(xml_file).getroot()
            except:
                print('Unable to extract from XML: {}'.format(xml_file))
            objects = root.findall('object')
            for obj in objects:
                class_name = obj.find("name").text.lower().strip()
                if class_name not in classes:
                    class_name = 'others'
                class_names.append(class_name)

    return class_names

def transfer_split_files(folders, folder_name, images, labels):
    print('\n{} : {} ({} samples)'.format(folder_name.capitalize(), folders[folder_name], len(images)))

    for filename in tqdm(list(zip(images, labels))):
        for i in range(2):
            source = filename[i]
            destination = os.path.join(folders[folder_name], os.path.split(filename[i])[-1])
            try:
                shutil.copyfile(source, destination) 
            except FileNotFoundError:
                print('Source: ', source, ',Destination: ', destination)


def split_dataset(
    path_to_dataset='', 
    train_ratio=0.7,
    valid_ratio=0.2, 
    label_format='', 
    overwrite=False,
    stratify=False,
    files_in_folder=False,
    classes=[],
    req=[]):
    
    """
    Split the dataset in the given path into three subsets(test, validation, train)
    """

    assert path_to_dataset != '', 'Please define path to dataset'
    assert label_format != '', 'Please define type of format for label [XML, TXT]'

    folders = {}

    # create directory
    if not req:
        for folder in ['train', 'valid', 'test']:
            folders[folder] = os.path.join(os.path.split(path_to_dataset)[0], folder)
            if overwrite:
                if os.path.exists(folders[folder]):
                    shutil.rmtree(folders[folder])
                os.makedirs(folders[folder])
            else:
                if not os.path.exists(folders[folder]):
                    os.makedirs(folders[folder])

    # get image filenames
    if files_in_folder:
        image_dict = merge_files_in_folders(path_to_dataset)
    else:
        # image_dict = get_files(label_format, path_to_dataset, classes, req)
        image_dict, label_files = get_files(label_format, path_to_dataset, classes, req)
        # print(image_dict)

    image_files = []
    for f in image_dict: image_files += image_dict[f]
    # print('image_files', image_files)

    # get label filenames
    # if label_format.lower() == 'xml':
    #     label_files = [os.path.splitext(f)[0] + '.xml' for f in image_files if os.path.exists(os.path.splitext(f)[0] + '.xml') or os.path.exists('_'.join(f.rsplit('.', 1))+'.xml')]
    # elif label_format.lower() == 'txt':
    #     label_files = [os.path.splitext(f)[0] + '.txt' for f in image_files if os.path.exists(os.path.splitext(f)[0] + '.txt') or os.path.exists('_'.join(f.rsplit('.', 1))+'.txt')]

    # split filenames
    if stratify:
        # WIP - not working yet
        labels = [get_class_name(f, classes) for f in label_files]
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(image_files), np.array(label_files), train_size=train_ratio, random_state=20, shuffle=True, stratify=np.array(label_files))
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=(valid_ratio/(1-train_ratio)), random_state=20, shuffle=True, stratify=np.array(label_files))
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(np.array(image_files), np.array(label_files), train_size=train_ratio, random_state=20, shuffle=True, stratify=None)
        X_valid, X_test, y_valid, y_test = train_test_split(X_valid, y_valid, train_size=(valid_ratio/(1-train_ratio)), random_state=20, shuffle=True, stratify=None)
    
    # transfer split filenames to respective folders
    if req:
        # print(list(zip(X_train, y_train)))
        # print(list(zip(X_valid, y_valid)))
        # print(list(zip(X_test, y_test)))

        voc.main(
            class_names=classes,
            URL=True,
            URLs=[
                list(zip(X_train, y_train)),
                list(zip(X_valid, y_valid)),
                list(zip(X_test, y_test))])
    else:
        transfer_split_files(folders, 'train', X_train, y_train)
        transfer_split_files(folders, 'valid', X_valid, y_valid)
        transfer_split_files(folders, 'test', X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=float, default=0.7, help='Ratio for train set. Default: 0.7')
    parser.add_argument('--valid', type=float, default=0.2, help='Ratio for validation set. Default: 0.2')
    parser.add_argument('--dataset', type=str, default='data/image/all', help='Path to dataset directory')
    parser.add_argument('--format', type=str, default='XML', help='Format of labels: [XML, TXT]')
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing train, valid, test folder')
    parser.add_argument('--stratify', action='store_true', help='use stratify to split train, valid, test folder')
    parser.add_argument('--files_in_folder', action='store_true', help='get image files from folders')
    parser.add_argument('--classes', type=str, default='data\logo_labels_19class.txt', help='Textfile list of classes')
    opt = parser.parse_args()

    class_names_19_class_txtfile = 'data\logo_labels_19class.txt'

    with open(class_names_19_class_txtfile) as f:
        content = f.readlines()
        class_names_19_class = [x.strip().lower() for x in content]

    split_dataset(
        path_to_dataset = opt.dataset,
        train_ratio = opt.train,
        valid_ratio = opt.valid,
        label_format = opt.format,
        overwrite = opt.overwrite,
        stratify = opt.stratify,
        files_in_folder = opt.files_in_folder,
        classes = class_names_19_class)

    voc.main()
