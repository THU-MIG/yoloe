import os
import types
import onnx
import torch
import torch.nn as nn

from mmdet.apis import init_detector
from projects.easydeploy.model import DeployModel, MMYOLOBackend
from projects.easydeploy.bbox_code import rtmdet_bbox_decoder as bbox_decoder


class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores, labels = torch.max(x[1], dim=-1, keepdim=True)
        return torch.cat([boxes, scores, labels.to(boxes.dtype)], dim=-1)


def pred_by_feat_deepstream(self, cls_scores, bbox_preds, objectnesses=None, **kwargs):
    assert len(cls_scores) == len(bbox_preds)
    dtype = cls_scores[0].dtype
    device = cls_scores[0].device

    num_imgs = cls_scores[0].shape[0]
    featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

    mlvl_priors = self.prior_generate(featmap_sizes, dtype=dtype, device=device)

    flatten_priors = torch.cat(mlvl_priors)

    mlvl_strides = [
        flatten_priors.new_full(
            (featmap_size[0] * featmap_size[1] * self.num_base_priors,), stride
        ) for featmap_size, stride in zip(
            featmap_sizes, self.featmap_strides
        )
    ]
    flatten_stride = torch.cat(mlvl_strides)

    flatten_cls_scores = [
        cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.num_classes) for cls_score in cls_scores
    ]
    cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

    flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4) for bbox_pred in bbox_preds]
    flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

    if objectnesses is not None:
        flatten_objectness = [objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1) for objectness in objectnesses]
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

    scores = cls_scores

    bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds, flatten_stride)

    return bboxes, scores


def rtmdet_export(weights, config, device):
    model = init_detector(config, weights, device=device)
    model.eval()
    deploy_model = DeployModel(baseModel=model, backend=MMYOLOBackend.ONNXRUNTIME, postprocess_cfg=None)
    deploy_model.eval()
    deploy_model.with_postprocess = True
    deploy_model.prior_generate = model.bbox_head.prior_generator.grid_priors
    deploy_model.num_base_priors = model.bbox_head.num_base_priors
    deploy_model.featmap_strides = model.bbox_head.featmap_strides
    deploy_model.num_classes = model.bbox_head.num_classes
    deploy_model.pred_by_feat = types.MethodType(pred_by_feat_deepstream, deploy_model)
    return deploy_model


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

    print('Opening RTMDet model')

    device = torch.device('cpu')
    model = rtmdet_export(args.weights, args.config, device)

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
    parser = argparse.ArgumentParser(description='DeepStream RTMDet conversion')
    parser.add_argument('-w', '--weights', required=True, type=str, help='Input weights (.pt) file path (required)')
    parser.add_argument('-c', '--config', required=True, help='Input config (.py) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
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
