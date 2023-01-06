import argparse
import os
import re
import xml.etree.ElementTree as ET

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


def get_files(label_format, folder_path, whitelist, blacklist, others, output):
    image_files = {}

    txt_files = []
    xml_files = []
    image_files = []
    classes = []
    
    for filename in os.listdir(os.path.join(folder_path)):
        if ' ' in filename:
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, filename.replace(' ', '_')))
            filename = filename.replace(' ', '_')

        if filename.lower().endswith('.xml') and label_format == 'XML':
            file_ext = None
            for ext in ['_png', '_jpg', '_jpeg']:
                if ext in filename:
                    filename = rename_xml_files(os.path.join(folder_path), filename)
                    file_ext = ext
                    break
            
            if file_ext is None:
                for ext in image_ext:
                    if os.path.exists(os.path.join(folder_path, filename.replace('.xml', ext))):
                        file_ext = ext
                        break
            
            has_label = os.path.exists(os.path.join(folder_path, filename))
            has_image = os.path.exists(os.path.join(folder_path, filename.replace('.xml', file_ext.replace('_', '.')))) if file_ext is not None else False
            if has_label and has_image:
                classes = classes + get_class_name(os.path.join(folder_path, filename), whitelist=whitelist, blacklist=blacklist, others=others)

                xml_files.append(os.path.join(folder_path, filename))
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

                txt_files.append(os.path.join(folder_path, filename))
                image_files.append(os.path.join(folder_path, filename.replace('.txt', file_ext.replace('_', '.'))))

        # print('Total files: {} images'.format(len(jpg_files)))
    
    classes = sorted(list(set(classes)))

    print()
    print('Number of classes:', len(classes))
    print('List of classes:', classes)
    print()

    with open(output, 'w') as f:
        f.write('\n'.join(classes))


def get_class_name(xml_file, whitelist=[], blacklist=[], others=False):
    class_names = []
    with open(xml_file, 'a') as f:
        try:
            root = ET.parse(xml_file).getroot()
        except:
            print('Unable to extract from XML: {}'.format(xml_file))
        objects = root.findall('object')
        for obj in objects:
            class_name = obj.find("name").text.lower().strip()
            if whitelist:
                if class_name not in whitelist:
                    if others:
                        class_name = 'others'
                    else:
                        continue
            
            if blacklist:
                if class_name in blacklist:
                    if others:
                        class_name = 'others'
                    else:
                        continue

            class_names.append(class_name)

    return class_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/image/all', help='Path to dataset directory')
    parser.add_argument('--format', type=str, default='XML', help='Format of labels: [XML, TXT]')
    parser.add_argument('--output', type=str, default='data/label.txt', help='Path to output label')
    parser.add_argument('--whitelist', nargs='+', default=[], help='List of classes to include')
    parser.add_argument('--blacklist', nargs='+', default=[], help='List of classes to exclude')
    parser.add_argument('--others', action='store_true', help='If not in list, create class "Others"')
    opt = parser.parse_args()

    if opt.whitelist:
        if '.txt' in opt.whitelist[0].lower():
            with open(opt.whitelist[0]) as f:
                content = f.readlines()
                opt.whitelist = [x.strip().lower() for x in content]
        else:
            opt.whitelist = list(map(lambda x:x.lower(), opt.whitelist)) if opt.whitelist else []
            
    if opt.blacklist:
        if '.txt' in opt.blacklist[0].lower():
            with open(opt.blacklist[0]) as f:
                content = f.readlines()
                opt.blacklist = [x.strip().lower() for x in content]
        else:
            opt.blacklist = list(map(lambda x:x.lower(), opt.blacklist)) if opt.blacklist else []

    get_files(
        label_format=opt.format, 
        folder_path=opt.dataset, 
        whitelist=opt.whitelist, 
        blacklist=opt.blacklist, 
        others=opt.others, 
        output=opt.output)
