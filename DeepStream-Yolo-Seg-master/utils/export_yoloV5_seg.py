import os
import sys
import random
import argparse
import warnings
import onnx
import torch
import torch.nn as nn
from models.experimental import attempt_load
from utils.torch_utils import select_device
from models.yolo import Detect


class RoiAlign(torch.autograd.Function):
    @staticmethod
    def forward(self, X, rois, batch_indices, coordinate_transformation_mode='half_pixel', mode='avg', output_height=160,
                output_width=160, sampling_ratio=0, spatial_scale=0.25):
        N, C, H, W = X.shape
        num_rois = rois.shape[0]
        return torch.randn((num_rois, C, output_height, output_width), device=rois.device, dtype=rois.dtype)

    @staticmethod
    def symbolic(g, X, rois, batch_indices, coordinate_transformation_mode='half_pixel', mode='avg', output_height=160,
                 output_width=160, sampling_ratio=0, spatial_scale=0.25):
        return g.op("RoiAlign", X, rois, batch_indices, coordinate_transformation_mode_s=coordinate_transformation_mode,
                    mode_s=mode, output_height_i=output_height, output_width_i=output_width, sampling_ratio_i=sampling_ratio,
                    spatial_scale_f=spatial_scale)


class NMS(torch.autograd.Function):
    @staticmethod
    def forward(self, boxes, scores, max_output_boxes_per_class=100, iou_threshold=0.45, score_threshold=0.25):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class=100, iou_threshold=0.45, score_threshold=0.25):
        return g.op("NonMaxSuppression", boxes, scores, torch.tensor([max_output_boxes_per_class]),
                    torch.tensor([iou_threshold]), torch.tensor([score_threshold]), center_point_box_i=0)


class DeepStreamOutput(nn.Module):
    def __init__(self, nc, conf_thres=0.25, iou_thres=0.45, max_det=100):
        self.nc = nc
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        super().__init__()

    def forward(self, x):
        preds = x[0]
        boxes = preds[:, :, :4]
        objectness = preds[:, :, 4:5]
        scores, classes = torch.max(preds[:, :, 5:self.nc+5], 2, keepdim=True)
        scores *= objectness
        classes = classes.float()
        masks = preds[:, :, self.nc+5:]
        protos = x[1]

        convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype,
                                      device=boxes.device)

        boxes = boxes @ convert_matrix

        selected_indices = NMS.apply(boxes, scores.transpose(1, 2).contiguous(), self.max_det, self.conf_thres,
                                     self.iou_thres)

        b, c, mh, mw = protos.shape
        n = selected_indices.shape[0]

        batch_index = selected_indices[:, 0]
        box_index = selected_indices[:, 2]

        selected_boxes = boxes[batch_index, box_index, :]
        selected_scores = scores[batch_index, box_index, :]
        selected_classes = classes[batch_index, box_index, :]
        selected_masks = masks[batch_index, box_index, :]

        pooled_proto = RoiAlign.apply(protos, selected_boxes, batch_index, 'half_pixel', 'avg', int(mh), int(mw), 0, 0.25)

        masks_protos = selected_masks.unsqueeze(dim=1) @ pooled_proto.float().view(n, c, mh * mw)
        masks_protos = masks_protos.sigmoid().view(-1, mh * mw)

        dets = torch.cat([selected_boxes, selected_scores, selected_classes, masks_protos], dim=1)

        batched_dets = dets.unsqueeze(0).repeat(b, 1, 1)
        batch_template = torch.arange(0, b, dtype=batch_index.dtype, device=batch_index.device).unsqueeze(1)
        batched_dets = batched_dets.where((batch_index == batch_template).unsqueeze(-1), batched_dets.new_zeros(1))

        y, i = batched_dets.shape[1:]

        final_dets = batched_dets.new_zeros((b, self.max_det, i))
        final_dets[:, :y, :] = batched_dets

        final_boxes = final_dets[:, :, :4]
        final_scores = final_dets[:, :, 4:5]
        final_classes = final_dets[:, :, 5:6]
        final_masks = final_dets[:, :, 6:]

        final_masks = final_masks.view(b, -1, mh, mw)

        return final_boxes, final_scores, final_classes, final_masks


def suppress_warnings():
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)


def yolov5_export(weights, device):
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.dynamic = False
            m.export = True
    return model


def main(args):
    suppress_warnings()

    print('\nStarting: %s' % args.weights)

    print('Opening YOLOv5-Seg model\n')

    device = select_device('cpu')
    model = yolov5_export(args.weights, device)

    if len(model.names.keys()) > 0:
        print('\nCreating labels.txt file')
        f = open('labels.txt', 'w')
        for name in model.names.values():
            f.write(name + '\n')
        f.close()

    model = nn.Sequential(model, DeepStreamOutput(len(model.names), args.conf_thres, args.iou_thres, args.max_det))

    img_size = args.size * 2 if len(args.size) == 1 else args.size

    if img_size == [640, 640] and args.p6:
        img_size = [1280] * 2

    onnx_input_im = torch.zeros(args.batch, 3, *img_size).to(device)
    onnx_output_file = os.path.basename(args.weights).split('.pt')[0] + '.onnx'

    dynamic_axes = {
        'input': {
            0: 'batch'
        },
        'boxes': {
            0: 'batch'
        },
        'scores': {
            0: 'batch'
        },
        'classes': {
            0: 'batch'
        },
        'masks': {
            0: 'batch'
        }
    }

    print('\nExporting the model to ONNX')
    torch.onnx.export(model, onnx_input_im, onnx_output_file, verbose=False, opset_version=args.opset,
                      do_constant_folding=True, input_names=['input'], output_names=['boxes', 'scores', 'classes', 'masks'],
                      dynamic_axes=dynamic_axes if args.dynamic else None)

    if args.simplify:
        print('Simplifying the ONNX model')
        import onnxsim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx, _ = onnxsim.simplify(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print('Done: %s\n' % onnx_output_file)


def parse_args():
    parser = argparse.ArgumentParser(description='DeepStream YOLOv5-Seg conversion')
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pt) file path (required)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [H,W] (default [640])')
    parser.add_argument('--p6', action='store_true', help='P6 model')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Minimum confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=100, help='Maximum detections')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    return args


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
