import os
import onnx
import paddle
import paddle.nn as nn

from ppdet.engine import Trainer
from ppdet.utils.cli import ArgsParser
from ppdet.slim import build_slim_model
from ppdet.data.source.category import get_categories
from ppdet.utils.check import check_version, check_config
from ppdet.core.workspace import load_config, merge_config


class DeepStreamOutput(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x['bbox']
        x['bbox_num'] = x['bbox_num'].transpose([0, 2, 1])
        scores = paddle.max(x['bbox_num'], axis=-1, keepdim=True)
        labels = paddle.argmax(x['bbox_num'], axis=-1, keepdim=True)
        return paddle.concat((boxes, scores, paddle.cast(labels, dtype=boxes.dtype)), axis=-1)


class DeepStreamInput(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        y = {}
        y['image'] = x['image']
        y['scale_factor'] = paddle.to_tensor([1.0, 1.0], dtype=x['image'].dtype)
        return y


def ppyoloe_export(FLAGS):
    cfg = load_config(FLAGS.config)
    FLAGS.opt['weights'] = FLAGS.weights
    FLAGS.opt['exclude_nms'] = True
    merge_config(FLAGS.opt)
    if FLAGS.slim_config:
        cfg = build_slim_model(cfg, FLAGS.slim_config, mode='test')
    merge_config(FLAGS.opt)
    check_config(cfg)
    check_version()
    trainer = Trainer(cfg, mode='test')
    trainer.load_weights(cfg.weights)
    trainer.model.eval()
    if not os.path.exists('.tmp'):
        os.makedirs('.tmp')
    static_model, _ = trainer._get_infer_cfg_and_input_spec('.tmp')
    os.system('rm -r .tmp')
    return trainer.cfg, static_model


def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore')


def main(FLAGS):
    suppress_warnings()

    print(f'\nStarting: {FLAGS.weights}')

    print('Opening PPYOLOE model')

    paddle.set_device('cpu')
    cfg, model = ppyoloe_export(FLAGS)

    anno_file = cfg['TestDataset'].get_anno()
    if os.path.isfile(anno_file):
        _, catid2name = get_categories(cfg['metric'], anno_file, 'detection_arch')
        print('Creating labels.txt file')
        with open('labels.txt', 'w', encoding='utf-8') as f:
            for name in catid2name.values():
                f.write(f'{name}\n')

    model = nn.Sequential(DeepStreamInput(), model, DeepStreamOutput())

    img_size = [cfg.eval_height, cfg.eval_width]

    onnx_input_im = {}
    onnx_input_im['image'] = paddle.static.InputSpec(shape=[FLAGS.batch, 3, *img_size], dtype='float32')
    onnx_output_file = f'{FLAGS.weights}.onnx'

    print('Exporting the model to ONNX')
    paddle.onnx.export(model, FLAGS.weights, input_spec=[onnx_input_im], opset_version=FLAGS.opset)

    if FLAGS.simplify:
        print('Simplifying the ONNX model')
        import onnxslim
        model_onnx = onnx.load(onnx_output_file)
        model_onnx = onnxslim.slim(model_onnx)
        onnx.save(model_onnx, onnx_output_file)

    print(f'Done: {onnx_output_file}\n')


def parse_args():
    parser = ArgsParser()
    parser.add_argument('-w', '--weights', required=True, help='Input weights (.pdparams) file path (required)')
    parser.add_argument('--slim_config', default=None, type=str, help='Slim configuration file of slim method')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='ONNX simplify model')
    parser.add_argument('--dynamic', action='store_true', help='Dynamic batch-size')
    parser.add_argument('--batch', type=int, default=1, help='Static batch-size')
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit('Invalid weights file')
    if args.dynamic and args.batch > 1:
        raise SystemExit('Cannot set dynamic batch-size and static batch-size at same time')
    elif args.dynamic:
        args.batch = None
    return args


if __name__ == '__main__':
    FLAGS = parse_args()
    main(FLAGS)
