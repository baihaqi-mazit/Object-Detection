# credit: https://medium.com/joelthchao/programmatically-detect-corrupted-image-8c1b2006c3d3
# credit: https://opensource.com/article/17/2/python-tricks-artists

from skimage import io
import os
from tqdm import tqdm
from PIL import Image
import cv2
from scipy import misc

def verify_image(img_file):
    try:
        img = io.imread(img_file)
    except:
        return False
    return True


def verify_image_pillow(img_file):
    # if img_file.endswith('.png'):
    #     try:
    #         img = Image.open(img_file) # open the image file
    #         img.verify() # verify that it is, in fact an image
    #     except (IOError, SyntaxError) as e:
    #         print('Bad file:', img_file) # print out the names of corrupt files
    
    try:
        im = Image.open(img_file)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        return True
    except OSError: 
        print('Bad file:', img_file) # print out the names of corrupt files
        return False


def verify_image_cv(img_file):
    with open(r'invailfile.txt', 'w+') as txt:
        try:
            img_raw = cv2.imread(img_file, flags=-1)
            cv2.imwrite(img_file, img_raw)
        except:
            print("fail load: " + img_file + '\n')


def verify_image_scikit(img_file):
    # with open('invalidfile.txt', 'a') as txt:
    #     print(img_file + '\n')
    #     try:
    #         img_raw = misc.imread(img_file)
    #         misc.imsave(img_file, img_raw)
    #     except:
    #         txt.write(img_file+'\n')
    #         print("fail load: " + img_file + '\n')

    try:
        _ = io.imread(img_file)
        img = cv2.imread(img_file)
        # Do stuff with img
    except Exception as e:
        print(img_file, e)
        return False


def verify_image_simple(filename):
    if os.path.splitext(filename)[-1] in image_ext:
        print(filename)
        cv2.imread(filename)


if __name__ == '__main__':
    folder_path = 'data/image/all'

    image_ext = ['.png', '.jpeg', '.jpg']

    for filename in tqdm([os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in image_ext]):
        # if not verify_image(filename):
        #     print('Corrupted image: {}'.format(filename))
        
        # verify_image_pillow(filename)
        # verify_image_cv(filename)
        # verify_image_scikit(filename)
        verify_image_simple(filename)
