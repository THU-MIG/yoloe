import os
import onnx
import torch
import torch.nn as nn

from yolov6.models.effidehead import Detect
from yolov6.layers.common import RepVGGBlock, SiLU
from yolov6.utils.checkpoint import load_checkpoint

try:
    from yolov6.layers.common import ConvModule
except ImportError:
    from yolov6.layers.common import Conv as ConvModule


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[:, :, :4]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        objectness = x[:, :, 4:5]
        scores, labels = torch.max(x[:, :, 5:], dim=-1, keepdim=True)
        scores *= objectness
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolov6_export(weights, device):
    model = load_checkpoint(weights, map_location=device, inplace=True, fuse=True)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, ConvModule):
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
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

    print('Opening YOLOv6 model')

    device = torch.device('cpu')
    model = yolov6_export(args.weights, device)

    model = nn.Sequential(model, DeepStreamOutput())

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280] * 2

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
    parser = argparse.ArgumentParser(description='DeepStream YOLOv6 conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--p6', action='store_true', help='P6 model')
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
