import os
import onnx
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.engine import Trainer
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_version, check_config
from ppdet.core.workspace import load_config, merge_config


class DeepStreamOutput(nn.Layer):
    def __init__(self, img_size, use_focal_loss):
        super().__init__()
        self.img_size = img_size
        self.use_focal_loss = use_focal_loss

    def forward(self, x):
        boxes = x['bbox']
        convert_matrix = paddle.to_tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]], dtype=boxes.dtype
        )
        boxes @= convert_matrix
        boxes *= paddle.to_tensor([[*self.img_size]]).flip(1).tile([1, 2]).unsqueeze(1)
        bbox_num = F.sigmoid(x['bbox_num']) if self.use_focal_loss else F.softmax(x['bbox_num'])[:, :, :-1]
        scores = paddle.max(bbox_num, axis=-1, keepdim=True)
        labels = paddle.argmax(bbox_num, axis=-1, keepdim=True)
        return paddle.concat((boxes, scores, paddle.cast(labels, dtype=boxes.dtype)), axis=-1)


def rtdetr_paddle_export(FLAGS):
    cfg = load_config(FLAGS.config)
    FLAGS.opt['weights'] = FLAGS.weights
    FLAGS.opt['exclude_nms'] = True
    FLAGS.opt['exclude_post_process'] = True
    merge_config(FLAGS.opt)
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

    print('Opening RT-DETR Paddle model')

    paddle.set_device('cpu')
    cfg, model = rtdetr_paddle_export(FLAGS)

    img_size = [cfg.eval_size[1], cfg.eval_size[0]]

    model = nn.Sequential(model, DeepStreamOutput(img_size, cfg.use_focal_loss))

    onnx_input_im = {}
    onnx_input_im['image'] = paddle.static.InputSpec(shape=[FLAGS.batch, 3, *img_size], dtype='float32')
    onnx_output_file = f'{FLAGS.weights}.onnx'

    print('Exporting the model to ONNX\n')
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
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')
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
