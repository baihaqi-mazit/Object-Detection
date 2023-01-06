import os
import sys

sys.path.append(".")

import argparse
import json
import random
import time
import zipfile
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import config as cfg
import utils.gpu as gpu
from model.loss.yolo_loss import YoloV3Loss
from model.yolov3 import Yolov3
from utils import cosine_lr_scheduler
from utils import datasets as data
from utils.early_stopping import EarlyStopping
from utils.evaluator import *
from utils.tools import *
from helper.split_dataset import *
import utils.voc as voc


class Trainer(object):
    def __init__(self, opt, class_names):
        init_seeds(0)
        self.device = gpu.select_device(opt.gpu)
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = opt.epoch
        self.weight_path = opt.weight
        self.multi_scale_train = opt.multiscale
        self.train_dataset = data.VocDataset(anno_file_type="train", img_size=opt.img_size, class_names=class_names)
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=opt.batch,
                                           num_workers=opt.worker,
                                           shuffle=True)
        self.yolov3 = Yolov3(class_names=class_names).to(self.device)
        self.opt = opt
        self.class_names = class_names

        if self.opt.optim == 'sgd':
            self.optimizer = optim.SGD(self.yolov3.parameters(), lr=self.opt.lr_init, momentum=self.opt.momentum, weight_decay=self.opt.weight_decay)
        elif self.opt.optim == 'adam':
            self.optimizer = optim.Adam(self.yolov3.parameters(), lr=self.opt.lr_init, weight_decay=self.opt.weight_decay)
        elif self.opt.optim == 'adamw':
            self.optimizer = optim.AdamW(self.yolov3.parameters(), lr=self.opt.lr_init, weight_decay=self.opt.weight_decay)

        self.criterion = YoloV3Loss(anchors=cfg.MODEL["ANCHORS"], strides=cfg.MODEL["STRIDES"],
                                    iou_threshold_loss=cfg.TRAIN["IOU_THRESHOLD_LOSS"])

        self.__load_model_weights(opt.weight, opt.resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                          T_max=self.epochs*len(self.train_dataloader),
                                                          lr_init=self.opt.lr_init,
                                                          lr_min=self.opt.lr_end,
                                                          warmup=self.opt.warmup*len(self.train_dataloader))

        # self.results = os.path.join(os.path.split(self.opt.weight)[0], 'results.txt')
        self.results = os.path.join(self.opt.exp, 'results.txt')

        # for chart plot
        self.fig, self.ax = plt.subplots(2, 3, figsize=(12, 6))
        self.ax = self.ax.ravel()
        self.title = ['GIoU', 'Objectness', 'Classification', 'Loss', 'LR', 'mAP']

        if not self.opt.resume:
            with open(self.results, 'w') as f:
                pass
        else:
            self.plot_chart(self.epochs)


    def __load_model_weights(self, weight_path, resume):
        if resume:
            # last_weight = os.path.join(self.opt.exp, 'last.pt')
            last_weight = self.opt.weight
            assert last_weight != '', 'Please define path to "last.pt" weights'
            chkpt = torch.load(last_weight, map_location=self.device)
            self.yolov3.load_state_dict(chkpt['model'])

            self.start_epoch = chkpt['epoch'] + 1
            if chkpt['optimizer'] is not None:
                self.optimizer.load_state_dict(chkpt['optimizer'])
                self.best_mAP = chkpt['best_mAP']
            del chkpt
        else:
            if 'darknet' in weight_path or not weight_path:
                weight_path = 'weights/darknet53_448.weights'
                self.yolov3.load_darknet_weights(weight_path)
            else:
                # WIP
                checkpoint = torch.load(weight_path, map_location=self.device)  # load checkpoint
                mod_weights = self.removekey(checkpoint['model'],[
                    '_Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv.weight',
                    '_Yolov3__fpn._FPN_YOLOV3__conv0_1._Convolutional__conv.bias',
                    '_Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv.weight',
                    '_Yolov3__fpn._FPN_YOLOV3__conv1_1._Convolutional__conv.bias',
                    '_Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv.weight',
                    '_Yolov3__fpn._FPN_YOLOV3__conv2_1._Convolutional__conv.bias',
                    ])
                self.yolov3.load_state_dict(mod_weights, strict=False)

                #Transfer learning (train only YOLO layers)
                # yolo_output_shape = cfg.DATA["NUM"]
                yolo_output_shape = len(self.class_names) if self.class_names else cfg.DATA["NUM"]
                for i, (name, p) in enumerate(self.yolov3.named_parameters()):
                    p.requires_grad = True if (p.shape[0] == ((5 + yolo_output_shape) * 3)) else False
                    

    def removekey(self, d, listofkeys):
        r = dict(d)
        for key in listofkeys:
            # print('key: {} is removed'.format(key))
            r.pop(key)
        return r


    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(self.opt.exp, "best.pt")
        last_weight = os.path.join(self.opt.exp, "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.yolov3.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)

        if self.opt.save_ckpt and epoch > 0 and epoch % 10 == 0:
            torch.save(chkpt, os.path.join(self.opt.exp, 'chkpt_epoch%g.pt'%epoch))
        del chkpt


    def train(self):
        since = time.time()

        print("Train datasets number is : {}".format(len(self.train_dataset)))

        es = EarlyStopping(patience=self.opt.es)

        eval_dict = {}
        epoch_dict = {}

        losses = []
        running_epoch = []
        eval_dict['last_loss'] = losses
        eval_dict['last_epoch'] = running_epoch
        
        mAP = 0

        assert self.start_epoch < self.epochs, 'Use bigger epoch than last epoch: {}'.format(self.start_epoch)

        for epoch in range(self.start_epoch, self.epochs):
            self.yolov3.train()

            mloss = torch.zeros(4)
            perform_dict = {}
            
            for i, (imgs, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)  in enumerate(self.train_dataloader):
                if imgs is None: continue

                self.scheduler.step(len(self.train_dataloader)*epoch + i)

                imgs = imgs.to(self.device)
                label_sbbox = label_sbbox.to(self.device)
                label_mbbox = label_mbbox.to(self.device)
                label_lbbox = label_lbbox.to(self.device)
                sbboxes = sbboxes.to(self.device)
                mbboxes = mbboxes.to(self.device)
                lbboxes = lbboxes.to(self.device)

                p, p_d = self.yolov3(imgs)

                loss, loss_giou, loss_conf, loss_cls = self.criterion(p, p_d, label_sbbox, label_mbbox,
                                                label_lbbox, sbboxes, mbboxes, lbboxes)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update running mean of tracked metrics
                loss_items = torch.tensor([loss_giou, loss_conf, loss_cls, loss])
                mloss = (mloss * i + loss_items) / (i + 1)

                # multi-scale training (320-608 pixels) every 10 batches
                if self.multi_scale_train and (i+1)%10 == 0:
                    self.train_dataset.img_size = random.choice(range(10,20)) * 32
                    print("multi_scale_img_size : {}".format(self.train_dataset.img_size), end='\r')

            # Log performance for eval.json
            perform_dict['loss'] = mloss[3].item()
            epoch_dict[epoch+1] = perform_dict
            eval_dict['epoch'] = epoch_dict

            running_epoch.append(epoch+1)
            losses.append(mloss[3].item())

            mAP = 0
            # if (epoch+1) > 4 and (epoch+1)/5==0:
            if (epoch+1) > 5:
                print('*'*20+"Validate"+'*'*20)
                with torch.no_grad():
                    APs = Evaluator(self.yolov3, self.class_names).APs_voc()
                    for i in APs:
                        print("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    print('mAP:%g'%(mAP))

            # Print epoch results
            s1 = ('Epoch:[ %d | %d ]    loss_giou: %.4f    loss_conf: %.4f    loss_cls: %.4f    loss: %.4f    '
                    'lr: %g') % (epoch+1, self.epochs, mloss[0],mloss[1], mloss[2], mloss[3],
                                self.optimizer.param_groups[0]['lr'])
            print(s1)

            s2 = ('%d/%d    %.4f     %.4f    %.4f    %.4f    '
                    '%g    %.4f') % (epoch+1, self.epochs, mloss[0],mloss[1], mloss[2], mloss[3],
                                self.optimizer.param_groups[0]['lr'], mAP)

            self.results = os.path.join(self.opt.exp, 'results.txt')
            with open(self.results, 'a') as f:
                f.write('{}\n'.format(s2))           # write results.txt

            # Plot loss chart
            self.plot_chart(self.epochs)

            if self.opt.es and es.step(mloss[3]):
                print("Early stopping at epoch:", epoch+1)
                eval_dict['last_loss'] = mloss[3].item()
                eval_dict['early_stop'] = epoch+1
                break

            self.__save_model_weights(epoch, mAP)
            print('best mAP : %g' % (self.best_mAP))
            

        # log evaluation result
        eval_dict['last_loss'] = losses[-1]
        eval_dict['last_epoch'] = running_epoch[-1]
        if self.opt.exp:
            with open(os.path.join(self.opt.exp, 'eval.json'), 'w') as outfile:
                json.dump(eval_dict, outfile)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


    def plot_chart(self, stop):
        try:
            results = np.loadtxt(self.results, usecols=[1, 2, 3, 4, 5, 6], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(1, min(stop, n) if stop else n)

            for i in range(6):
                y = results[i, x]
                y[y == 0] = np.nan  # dont show zero loss values

                self.ax[i].plot(x, y, marker='.', linewidth=2, markersize=8)
                self.ax[i].set_title(self.title[i])

        except Exception as e:
            print('Warning: Plotting error for %s, skipping file' % self.results)
            print(e)

        self.fig.tight_layout()
        self.fig.savefig(os.path.join(self.opt.exp, 'results.png'), dpi=200)


def zip_folder(exp_path):
    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file), 
                        os.path.relpath(os.path.join(root, file), 
                                        os.path.join(path, '..')))

    print('Zipping file...')
    zipf = zipfile.ZipFile(os.path.join(os.path.split(exp_path)[0], os.path.split(exp_path)[-1]) + '.zip', 'w', zipfile.ZIP_DEFLATED)
    zipdir(exp_path, zipf)
    zipf.close()
    print('DONE')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"), help='experiment name')
    parser.add_argument('--weight', type=str, default='', help='weight file path')
    parser.add_argument('--resume', action='store_true', default=False,  help='if True, resume last training')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--augment', action='store_true', default=True,  help='if True, apply online augment')
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--multiscale', action='store_true', default=True,  help='if True, apply multi-scale')
    parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--worker', type=int, default=4, help='number of workers')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum value')
    parser.add_argument('--warmup', type=int, default=2, help='number of warmup epochs')
    parser.add_argument('--optim', type=str, default='sgd', help='type of optimizer [sgd, adam, adamw]')
    parser.add_argument('--es', type=int, default=0, help='early stopping patience value. if zero, no early stopping')
    parser.add_argument('--img_size', type=int, default=416, help='train image size')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay value')
    parser.add_argument('--lr_init', type=float, default=0.0001, help='initial learning rate value')
    parser.add_argument('--lr_end', type=float, default=0.000001, help='final learning rate value')
    parser.add_argument('--save_ckpt', type=bool, default=False, help='if save checkpoints, set True')
    parser.add_argument('--zip', action='store_true', default=True, help='if True, zip archive experiment folder')
    parser.add_argument('--split_dataset', action='store_true', default=True, help='if True, split dataset into train, valid, test set')
    parser.add_argument('--overwrite_dataset', action='store_true', default=True, help='if True, overwrite previous dataset in train, valid, test set')
    parser.add_argument('--dataset', type=str, default='data/image/all', help='path to overall dataset')
    parser.add_argument('--format', type=str, default='XML', help='Format of labels: [XML, TXT]')
    parser.add_argument('--classes', type=str, default='data/label.txt', help='Textfile list of classes')
    parser.add_argument('--reqs', type=str, default='', help='Sample json request')
    opt = parser.parse_args()

    # test url
    req = None
    if opt.reqs:
        with open(opt.reqs) as f:
            req = json.load(f)

    if opt.exp:
        opt.exp = os.path.join('experiments', opt.exp + '_' + datetime.now().strftime("%Y%m%d%H%M%S"))

        if not os.path.exists(opt.exp):
            os.makedirs(opt.exp)

        # Log hyperparameters and configs
        with open(os.path.join(opt.exp, 'conf.json'), 'w') as outfile:
            json.dump(vars(opt), outfile)
    else:
        opt.exp = 'weights'

    class_names = []

    # Split dataset
    if opt.split_dataset:
        assert opt.classes != '', 'Please define path to textfile with class names using "opt.classes"'

        with open(opt.classes) as f:
            content = f.readlines()
            class_names = [x.strip().lower() for x in content]

        shutil.copy(opt.classes, os.path.join(opt.exp, os.path.split(opt.classes)[-1]))
        
        split_dataset(
            path_to_dataset=opt.dataset,
            train_ratio=0.7,
            valid_ratio=0.2,
            label_format=opt.format,
            overwrite=opt.overwrite_dataset,
            stratify=False,
            files_in_folder=False,
            classes=class_names,
            req=req)
    
    if not opt.reqs:
        voc.main(class_names)

    trainer = Trainer(opt=opt, class_names=class_names)
    trainer.train()

    # Zip archive experiment folder
    zip_folder(opt.exp)
