import os
import pickle
import shutil
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from tqdm import tqdm

import config as cfg
from utils.visualize import *


img_ext = ['.jpg', '.jpeg', '.png']


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        try:
            image_resized = cv2.resize(img, (resize_w, resize_h))
        except:
            print(resize_w, resize_h, resize_ratio, w_org, h_org)

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Evaluator(object):
    def __init__(self, model, class_names=[], valid=True, visual=True):
        # self.classes = cfg.DATA["CLASSES"]
        self.classes = class_names if class_names else cfg.DATA["CLASSES"]
        self.pred_result_path = cfg.RESULT_PATH
        self.val_data_path = cfg.VALID_DATA_PATH if valid else cfg.TEST_DATA_PATH
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape =  cfg.TEST["TEST_IMG_SIZE"]

        self.__visual = visual
        self.__visual_imgs = 0

        self.model = model
        self.device = next(model.parameters()).device

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds = [f for f in os.listdir(self.val_data_path) if os.path.splitext(f)[-1] in img_ext]

        if os.path.exists(self.pred_result_path):
            shutil.rmtree(self.pred_result_path)
        os.mkdir(self.pred_result_path)

        for img_ind in tqdm(img_inds):
            img_path = os.path.join(self.val_data_path, img_ind)
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            if bboxes_prd.shape[0]!=0 and self.__visual and self.__visual_imgs < 100:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]

                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = "data/results/{}.jpg".format(self.__visual_imgs)
                cv2.imwrite(path, img)

                self.__visual_imgs += 1

            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])

                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'

                with open(os.path.join(self.pred_result_path, 'comp4_det_test_' + class_name + '.txt'), 'a') as f:
                    f.write(s)

        return self.__calc_APs()

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(320, 640, 96)
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        bboxes = self.nms(bboxes, self.conf_thresh, self.nms_thresh)
        # boxes_list, scores_list, labels_list = bboxes
        # boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5, skip_box_thr=0.0001)

        return bboxes


    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape

        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, p_d = self.model(img)
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        return bboxes


    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        pred_coor = self.xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)

        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        filename = os.path.join(self.pred_result_path, 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations', '{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path)
        APs = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = self.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)
            APs[cls] = AP
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)

        return APs

    def nms(self, bboxes, score_threshold, iou_threshold, sigma=0.3, method='nms'):
        classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
            cls_bboxes = bboxes[cls_mask]
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = self.iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                assert method in ['nms', 'soft-nms']
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > score_threshold
                cls_bboxes = cls_bboxes[score_mask]
        return np.array(best_bboxes)


    def xywh2xyxy(self, x):
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    
    def voc_eval(self,
                detpath,
                annopath,
                imagesetfile,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=False):
        # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')

        imagenames = [os.path.join(imagesetfile,f) for f in os.listdir(imagesetfile) if os.path.splitext(f)[-1] in img_ext]


        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                try:
                    recs[os.path.split(imagename)[-1]] = self.parse_rec(os.path.splitext(imagename)[0] + '.xml')
                    if i % 100 == 0:
                        print ('Reading annotation for {:d}/{:d}'.format(
                            i + 1, len(imagenames)))
                except Exception as e:
                    pop_recs = recs.pop(imagename, None)
                    print('pop_recs:', pop_recs)
                    print(e)
                    continue

            # save
            print ('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        imagenames = [os.path.split(f)[-1] for f in imagenames]
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'].lower() == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        try:
            with open(detfile, 'r') as f:
                lines = f.readlines()
        except:
            return 0,0,0

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                    (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                    (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap


    def iou_xyxy_numpy(self, boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        IOU = 1.0 * inter_area / union_area
        return IOU

    
    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text.lower()
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            
            # added to classify object with classes other than in list as 'others'
            if obj_struct['name'] not in self.classes:
                obj_struct['name'] = 'others'

            objects.append(obj_struct)

        return objects

    
    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
