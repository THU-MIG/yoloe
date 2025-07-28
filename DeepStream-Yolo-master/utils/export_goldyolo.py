import os
import onnx
import torch
import torch.nn as nn

import yolov6.utils.general as _m
from yolov6.layers.common import SiLU
from gold_yolo.switch_tool import switch_to_deploy
from yolov6.utils.checkpoint import load_checkpoint


def _dist2bbox(distance, anchor_points, box_format='xyxy'):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    bbox = torch.cat([x1y1, x2y2], -1)
    return bbox

_m.dist2bbox.__code__ = _dist2bbox.__code__


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        objectness = x[:, :, 4:5]
        scores, labels = torch.max(x[:, :, 5:], dim=-1, keepdim=True)
        scores *= objectness
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def gold_yolo_export(weights, device, inplace=True, fuse=True):
    model = load_checkpoint(weights, map_location=device, inplace=inplace, fuse=fuse)
    model = switch_to_deploy(model)
    for layer in model.modules():
        t = type(layer)
        if t.__name__ == 'RepVGGBlock':
            layer.switch_to_deploy()
    model.eval()
    for k, m in model.named_modules():
        if m.__class__.__name__ == 'Conv':
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif m.__class__.__name__ == 'Detect':
            m.inplace = False
    return model


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=ResourceWarning)


def main(args):
    suppress_warnings()

    print(f'\nStarting: {args.weights}')

    print('Opening Gold-YOLO model')

    device = torch.device('cpu')
    model = gold_yolo_export(args.weights, device)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = f'{args.weights}.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'output': {
            0: 'batch'
        }
    }

    print('Exporting the model to ONNX')
    torch.onnx.export(
        model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset, do_constant_folding=True,
        input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes if args.dynamic else None
    )

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Done: {onnx_output_file}\n')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DeepStream Gold-YOLO conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
