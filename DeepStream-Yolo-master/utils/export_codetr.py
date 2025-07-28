import os
import types
import onnx
import torch
import torch.nn as nn
from copy import deepcopy

from projects import *
from mmengine.registry import MODELS
from mmdeploy.utils import load_config
from mmdet.utils import register_all_modules
from mmengine.model import revert_sync_batchnorm
from mmengine.runner.checkpoint import load_checkpoint


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = []
        scores = []
        labels = []
        for det in x:
            boxes.append(det.bboxes)
            scores.append(det.scores.unsqueeze(-1))
            labels.append(det.labels.unsqueeze(-1))
        boxes = torch.stack(boxes, dim=0)
        scores = torch.stack(scores, dim=0)
        labels = torch.stack(labels, dim=0)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def forward_deepstream(self, batch_inputs, batch_data_samples):
    b, _, h, w = batch_inputs.shape
    batch_data_samples = [{'batch_input_shape': (h, w), 'img_shape': (h, w)} for _ in range(b)]
    img_feats = self.extract_feat(batch_inputs)
    return self.predict_query_head(img_feats, batch_data_samples, rescale=False)


def query_head_predict_deepstream(self, feats, batch_data_samples, rescale=False):
    with torch.no_grad():
        outs = self.forward(feats, batch_data_samples)
        predictions = self.predict_by_feat(
            *outs, batch_img_metas=batch_data_samples, rescale=rescale)
    return predictions


def codetr_export(weights, config, device):
    register_all_modules()
    model_cfg = load_config(config)[0]
    model = deepcopy(model_cfg.model)
    model.pop('pretrained', None)
    for key in model['train_cfg']:
        if 'rpn_proposal' in key:
            key['rpn_proposal'] = {}
    model['test_cfg'] = [{}, {'rpn': {}, 'rcnn': {}}, {}]
    preprocess_cfg = deepcopy(model_cfg.get('preprocess_cfg', {}))
    preprocess_cfg.update(deepcopy(model_cfg.get('data_preprocessor', {})))
    model.setdefault('data_preprocessor', preprocess_cfg)
    model = MODELS.build(model)
    load_checkpoint(model, weights, map_location=device)
    model = revert_sync_batchnorm(model)
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'switch_to_deploy'):
        model.backbone.switch_to_deploy()
    if hasattr(model, 'switch_to_deploy') and callable(model.switch_to_deploy):
        model.switch_to_deploy()
    model = model.to(device)
    model.eval()
    del model.data_preprocessor
    model._forward = types.MethodType(forward_deepstream, model)
    model.query_head.predict = types.MethodType(query_head_predict_deepstream, model.query_head)
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

    print('Opening CO-DETR model')

    device = torch.device('cpu')
    model = codetr_export(args.weights, args.config, device)

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
    parser = argparse.ArgumentParser(description='DeepStream CO-DETR conversion')
    parser.add_argument('-w', '--weights', required=True, type=str, help='Input weights (.pth) file path (required)')
    parser.add_argument('-c', '--config', required=True, help='Input config (.py) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if not os.path.isfile(args.config):
        raise SystemExit('Invalid config file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
