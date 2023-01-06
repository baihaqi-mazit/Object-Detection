# Fish Detection
---
# Introduction
This is the code for running training and prediction using pre-trained YOLOv3 model for:
* Fish Detection

The models are written in PyTorch. The dataset is provided, cleaned, pre-processed and utilized by Skymind.

---

## Environment
* Windows 10
* Python 3.7

```bash
# Install packages
pip install -r requirements.txt
```

---

## Setup

### 1. Download weight file
* Pre-trained weights :  [Google Drive](https://drive.google.com/drive/folders/1V7f0khfRcWcreX3_QKzasvrZZkInV9lY?usp=sharing) 
* Download latest weights from latest folder
* Put the weight files in `weights` folder

---

## Inference

To predict on image path using argument, run:
```bash
python inference.py
```

```
(OPTIONAL ARGUMENTS):
-h, --help           show this help message and exit
  --weights WEIGHTS    weight file path
  --img_path IMG_PATH  test image path or None
  --classes CLASSES    label classes textfile path
```
where `img_path` can be:
- Path to image file
- Path to folder with image files
- If empty, runs on pre-defined path to test set in `config.py`
```
(DEFAULTS):
  --weights WEIGHTS    'weights/best.pt'
  --img_path IMG_PATH  'sample'
  --classes CLASSES    'data/label.txt'
```

Supported input formats : 
* JPG
* JPEG
* PNG

Expected output:
- Image with bounding box in `results` folder
- Output list of dictionary:
```
[
   {
      'bbox': [520.402099609375,
           437.2177429199219,
           559.5408325195312,
           453.2886657714844],
      'class_name': 'unitPrice',
      'conf': 0.8106774687767029,
      'text': '12.30'
   },
   {
      'bbox': [520.402099609375,
           437.2177429199219,
           559.5408325195312,
           453.2886657714844],
      'class_name': 'unitPrice',
      'conf': 0.8106774687767029
      'text': '45.60'
   },
  ...
]
  ```
---
## Training
### 1. Structure dataset
* Transfer image and XML labels into folder `data/image/all`

### 2. Configurations: 
Change the configurations in `config.py` (IF NECESSARY):
- TRAIN_DATA_PATH: Path to train dataset
- VALID_DATA_PATH: Path to validation dataset
- TEST_DATA_PATH: Path to test dataset
- RESULT_PATH: Path for output result
- DATA: List and number of classes (Optional, if no classnames textfile provided)
- MODEL: Anchors, strides and anchors per scale
- TRAIN: Hyperparameters for training
- TEST: Hyperparameters for training

The defaults should be fine if follow the same default structure.

### 3. Generate label textfile

Generate class name textfile if not available by running following command:
```
python helper/generate_label.py
```
```
(OPTIONAL ARGUMENTS):
-h, --help            show this help message and exit
  --dataset DATASET     Path to dataset directory
  --format FORMAT       Format of labels: [XML, TXT]
  --output OUTPUT       Path to output label
  --whitelist WHITELIST [WHITELIST ...]
                        List of classes to include
  --blacklist BLACKLIST [BLACKLIST ...]
                        List of classes to exclude
  --others              If not in list, create class "Others"
```
```
(DEFAULTS):
  --dataset DATASET     'data/image/all'
  --format FORMAT       'XML'
  --output OUTPUT       'data/label.txt'
  --whitelist WHITELIST []
  --blacklist BLACKLIST []
  --others              False
```
### 3. Start training

Run the following command to start training:
```
python train.py --exp Test
```
Example:

`Local dataset`
```
python train.py --exp Test --epoch 50
```
`URL dataset`
```
python train.py --exp Test --epoch 50 --reqs sample_request.json
```
```
(OPTIONAL ARGUMENTS):
-h, --help            show this help message and exit
  --exp EXP             experiment name
  --weight WEIGHT       weight file path
  --resume              if True, resume last training
  --gpu GPU             gpu id
  --augment             if True, apply online augment
  --batch BATCH         batch size
  --multiscale          if True, apply multi-scale
  --epoch EPOCH         number of epochs
  --worker WORKER       number of workers
  --momentum MOMENTUM   momentum value
  --warmup WARMUP       number of warmup epochs
  --optim OPTIM         type of optimizer [sgd, adam, adamw]
  --es ES               early stopping patience value. if zero, no early
                        stopping
  --img_size IMG_SIZE   train image size
  --weight_decay WEIGHT_DECAY
                        weight decay value
  --lr_init LR_INIT     initial learning rate value
  --lr_end LR_END       final learning rate value
  --save_ckpt SAVE_CKPT
                        if save checkpoints, set True
  --zip                 if True, zip archive experiment folder
  --split_dataset       if True, split dataset into train, valid, test set
  --overwrite_dataset   if True, overwrite previous dataset in train, valid,
                        test set
  --dataset DATASET     path to overall dataset
  --format FORMAT       Format of labels: [XML, TXT]
  --classes CLASSES     Textfile list of classes
  --reqs REQS           Sample json request
```
```
(DEFAULTS):
  --exp EXP             Current datetime
  --weight WEIGHT       Empty (Required if using pretrained weights)
  --resume              False
  --gpu GPU             0
  --augment             True
  --batch BATCH         1
  --multiscale          True
  --epoch EPOCH         50
  --worker WORKER       4
  --momentum MOMENTUM   0.9
  --warmup WARMUP       2
  --optim OPTIM         'sgd'
  --es ES               0
  --img_size IMG_SIZE   416
  --weight_decay WEIGHT_DECAY
                        0.0005
  --lr_init LR_INIT     0.0001
  --lr_end LR_END       0.000001
  --save_ckpt SAVE_CKPT
                        False
  --zip                 True
  --split_dataset       True
  --overwrite_dataset   True
  --dataset DATASET     'data/image/all'
  --format FORMAT       'XML'
  --classes CLASSES     'data/label.txt'
  --reqs REQS           ''
```

Supported input formats : 
* JPG
* JPEG
* PNG

Expected output:
- Zipped folder containing:
   1) `best.pt`: Best weights from previous training
   2) `last.pt`: Last weights from the training
   3) `conf.json`: Configurations for the training
   4) `eval.json`: Evaluation results for the training
   5) `label.txt`: Class names for the training
   6) `results.png`: Performance graph for the training
   7) `results.txt`: Performance log for the training

---
## To-dos
- [x] Read URL dataset (images + labels) from JSON request
- [ ] Read hyperparameters configurations from JSON request

