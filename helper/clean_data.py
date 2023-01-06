import os
import zipfile
import shutil

dataset_path = 'data/image/hospitals'

for folders in os.listdir(dataset_path):
    for files in os.listdir(os.path.join(dataset_path, folders)):
        files_dir = os.path.join(dataset_path, folders, files)

        if os.path.splitext(files)[-1].lower() == '.zip':
            if '_pvoc_' in files:
                with zipfile.ZipFile(files_dir, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(dataset_path, folders))
                # shutil.rmtree(files_dir)
                os.remove(files_dir)
            else:
                os.remove(files_dir)
                # shutil.rmtree(files_dir)

        elif os.path.splitext(files)[-1].lower() == '.tif':
            os.remove(files_dir)
