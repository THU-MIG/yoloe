import os
import onnx
import torch
import torch.nn as nn


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x[0]
        boxes = x[:, :, :4]
        convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype, device=boxes.device
        )
        boxes @= convert_matrix
        objectness = x[:, :, 4:5]
        scores, labels = torch.max(x[:, :, 5:], dim=-1, keepdim=True)
        scores *= objectness
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def yolor_export(weights, cfg, size, device):
    if os.path.isfile('models/experimental.py'):
        import models
        from models.experimental import attempt_load
        from utils.activations import Hardswish
        model = attempt_load(weights, map_location=device)
        for k, m in model.named_modules():
            m._non_persistent_buffers_set = set()
            if isinstance(m, models.common.Conv) and isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None
        model.model[-1].training = False
        model.model[-1].export = False
    else:
        from models.models import Darknet
        model_name = os.path.basename(weights).split('.pt')[0]
        if cfg == '':
            cfg = 'cfg/' + model_name + '.cfg'
            if not os.path.isfile(cfg):
                raise SystemExit('CFG file not found')
        model = Darknet(cfg, img_size=size[::-1]).to(device)
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
        model.float()
        model.fuse()
        model.eval()
        model.module_list[-1].training = False
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

    print('Opening YOLOR model')

    device = torch.device('cpu')
    model = yolor_export(args.weights, args.cfg, args.size, device)

    if hasattr(model, 'names') and len(model.names) > 0:
        print('Creating labels.txt file')
        with open('labels.txt', 'w', encoding='utf-8') as f:
            for name in model.names:
                f.write(f'{name}\n')

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
    parser = argparse.ArgumentParser(description='DeepStream YOLOR conversion')
    parser.add_argument('-w', '--weights', required=True, type=str, help='Input weights (.pt) file path (required)')
    parser.add_argument('-c', '--cfg', default='', help='Input cfg (.cfg) file path')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--p6', action='store_true', help='P6 model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version')
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
