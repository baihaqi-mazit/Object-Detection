# Modified from: https://github.com/ultralytics/yolov5/blob/develop/models/export.py

import sys

sys.path.append(".")

import argparse

import torch
import utils.gpu as gpu
from model.yolov3 import Yolov3


def load_class_names(class_txtfile):
    with open(class_txtfile) as f:
        content = f.readlines()
        return [x.strip().lower() for x in content]


def attempt_load(weight, map_location, class_names):
    model = Yolov3(class_names=class_names).to(map_location)
    chkpt = torch.load(weight, map_location=map_location)

    if 'best' in weight:
        model.load_state_dict(chkpt)
    else:
        model.load_state_dict(chkpt['model'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='weights path')
    parser.add_argument('--classes', type=str, default='data/label.txt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--include', nargs='+', default=['onnx'], help='include formats')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set YOLOv5 Detect() inplace=True')
    parser.add_argument('--train', action='store_true', help='model.train() mode')
    parser.add_argument('--optimize', action='store_true', help='optimize TorchScript for mobile')  # TorchScript-only
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')  # ONNX-only
    parser.add_argument('--simplify', action='store_true', help='simplify ONNX model')  # ONNX-only
    parser.add_argument('--opset-version', type=int, default=12, help='ONNX opset version')  # ONNX-only
    opt = parser.parse_args()

    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    opt.include = [x.lower() for x in opt.include]
    print(opt)

    class_txtfile = opt.classes

    # Load PyTorch model
    device = gpu.select_device(opt.device)
    class_names = load_class_names(class_txtfile)
    model = attempt_load(opt.weights, map_location=device, class_names=class_names)  # load FP32 model

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)  # image size(1,3,320,192) iDetection

    prefix = 'ONNX:'
    try:
        import onnx

        print(f'{prefix} starting export with onnx {onnx.__version__}...')
        f = opt.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, f, verbose=True, opset_version=opt.opset_version, input_names=['images'],
                            training=torch.onnx.TrainingMode.TRAINING if opt.train else torch.onnx.TrainingMode.EVAL,
                            do_constant_folding=not opt.train,
                            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # size(1,3,640,640)
                                        'output': {0: 'batch', 2: 'y', 3: 'x'}} if opt.dynamic else None)
        # torch.onnx.export(model, img, f, verbose=True, input_names=['images'], output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

    except Exception as e:
        print(f'{prefix} export failure: {e}')
