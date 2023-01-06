import argparse
import json
import os
import pprint
import shutil
import string
import sys
import urllib.request
# from nltk import word_tokenize
# from nltk.corpus import stopwords
import cv2
import numpy as np

import torch

# stop_words = stopwords.words('english')
sys.path.append(".")

import config as cfg
from model.yolov3 import Yolov3
from utils.evaluator import Evaluator
from utils.visualize import *

img_ext = ['.jpg', '.jpeg', '.png']


class Tester(object):
    def __init__(self,
                 weight_path=None,
                 img_size=544,
                 visual=None,
                 class_names=[]
                 ):
        self.img_size = img_size
        self.__conf_threshold = 0.5
        self.__device = torch.device('cpu')

        self.__visual = visual
        # self.__classes = cfg.DATA["CLASSES"]
        self.__classes = class_names if class_names else cfg.DATA["CLASSES"]
        self.__model = Yolov3().to(self.__device)

        self.__load_model_weights(weight_path)

        self.__evalter = Evaluator(self.__model, visual=False)

    def __load_model_weights(self, weight_path):
        print("loading weight file from : {}".format(weight_path))

        weight = os.path.join(weight_path)
        chkpt = torch.load(weight, map_location=self.__device)
        self.__model.load_state_dict(chkpt)
        print("loading weight file is done")
        del chkpt

    def test(self):
        if self.__visual:
            imgs = []
            imgs_path = None
            img = None

            if os.path.isdir(self.__visual):
                imgs = [f for f in os.listdir(self.__visual) if os.path.splitext(f)[-1] in img_ext]
                imgs_path = self.__visual
            elif os.path.isfile(self.__visual):
                imgs = [os.path.split(self.__visual)[-1]]
                imgs_path = os.path.split(self.__visual)[0]
            elif 'http' in self.__visual:
                img = self.url_to_image(self.__visual)
                imgs = [os.path.split(self.__visual)[-1]]
                imgs_path = os.path.split(self.__visual)[0]
            else:
                print('--img_path: defined invalid path - {}'.format(self.__visual))
                return

            assert len(imgs) > 0, 'No images in Test folder'

            json_file = open("sample.json", "w")
            results = []
            for v in imgs:
                path = os.path.join(imgs_path, v)
                print("\ntest images : {}".format(path))

                if img is None:
                    img = cv2.imread(path)
                else:
                    img = cv2.imread(path)
                ori_img = img.copy()
                assert img is not None

                bboxes_prd = self.__evalter.get_bbox(img)
                # print(bboxes_prd)

                if bboxes_prd.shape[0] != 0:
                    print(bboxes_prd.shape[0])
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]

                    visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores,
                                    class_labels=self.__classes)
                    path = "results/{}".format(v)

                    txt = 'Fish Count : {}'.format(str(bboxes_prd.shape[0]))
                    x, y = (10, 600)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_color_bg = (0,0,0)
                    text_color = (0, 255, 0)
                    font_scale = 1
                    font_thickness = 2
                    text_size, _ = cv2.getTextSize(txt, font, font_scale, font_thickness)
                    text_w, text_h = text_size
                    cv2.rectangle(img, (x,y), (x + text_w, y + text_h), text_color_bg, -1)
                    cv2.putText(img, txt, (x, y + text_h + font_scale - 1), font, font_scale, text_color,
                                font_thickness)

                    cv2.imwrite(path, img)
                    print("saved images : {}".format(path))
                    v = v.replace('.jpg', '')

                    count = 0
                    for class_ind, score, box in zip(class_inds, scores, boxes):
                        if score > self.__conf_threshold:
                            output = {}
                            # output['class_name'] = self.__classes[class_ind]
                            output["score"] = score
                            output["bbox"] = box.tolist()
                            output["image_id"] = v
                            output["category_id"] = int(class_ind) + 1
                            x1, y1, x2, y2 = list(map(int, output['bbox']))

                            results.append(output)
                            # output['text'] = pytesseract.image_to_string(ori_img[y1:y2, x1:x2]).strip().replace('\n', '')
                            # text = pytesseract.image_to_string(ori_img[y1:y2, x1:x2])
                            # array = []
                            #
                            # for char in text:
                            #     if char not in string.punctuation:
                            #         array.append(char)
                            #     else:
                            #         char = " "
                            #         array.append(char)
                            #
                            # text_d = "".join(array)
                            # words = word_tokenize(text_d)
                            # filtered_words = [word for word in words if word not in stop_words]
                            # filtered_sentence = " ".join(filtered_words)
                            # output['test'] = filtered_sentence
                            # results.append(output)
                            # # print('Class: {}, Conf: {}, Bbox: {}'.format(self.__classes[class_ind], score, box.tolist()))
                            # count = count+1
                            # write_text = open('texts/{}{}.txt'.format(v,count),'w')
                            # write_text.write(filtered_sentence)
                            # write_text.close()

            pprint.pprint(results)
            json.dump(results, json_file)

    def url_to_image(self, image_url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        with urllib.request.urlopen(image_url) as url:
            image = np.asarray(bytearray(url.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='weight file path')
    parser.add_argument('--img_path', type=str, default='sample', help='test image path or None')
    parser.add_argument('--classes', type=str, default='data/label.txt', help='label classes textfile path')
    opt = parser.parse_args()

    class_names = []

    print('Reading class names from: {}'.format(opt.classes))

    with open(opt.classes) as f:
        content = f.readlines()
        class_names = [x.strip().lower() for x in content]
    print('Class names:\n{}'.format(class_names))

    Tester(weight_path=opt.weights,
           visual=opt.img_path,
           class_names=class_names).test()
