import os
import argparse
import re

img_ext = ['.png', '.jpeg', '.jpg']

def remove_underscore(path_to_folder, label_format):
    for filename in [f for f in os.listdir(path_to_folder) if f.endswith('.'+ label_format.lower())]:
        for ext in img_ext:
            if ext.replace('.', '_') in filename:
                renamed_filename = re.sub(ext.replace('.', '_'), '', filename)
                break
            else:
                renamed_filename = filename

        try:
            os.rename(os.path.join(path_to_folder, filename), os.path.join(path_to_folder, renamed_filename))
        except Exception as e:
            print(e)
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='', help='Path to folder')
    parser.add_argument('--label_format', type=str, default='XML', help='Format of labels: [XML, TXT]')
    opt = parser.parse_args()

    remove_underscore(opt.label_path, opt.label_format)

