# Benchmarks

### Config

```
board = NVIDIA Tesla V100 16GB (AWS: p3.2xlarge)
batch-size = 1
eval = val2017 (COCO)
sample = 1920x1080 video
```

**NOTE**: Used maintain-aspect-ratio=1 in config_infer file for Darknet (with letter_box=1) and PyTorch models.

### NMS config

- Eval

```
nms-iou-threshold = 0.6 (Darknet) / 0.65 (YOLOv5, YOLOv6, YOLOv7, YOLOR and YOLOX) / 0.7 (Paddle, YOLO-NAS, DAMO-YOLO, YOLOv8 and YOLOv7-u6)
pre-cluster-threshold = 0.001
topk = 300
```

- Test

```
nms-iou-threshold = 0.45
pre-cluster-threshold = 0.25
topk = 300
```

### Results

**NOTE**: * = PyTorch.

**NOTE**: ** = The YOLOv4 is trained with the trainvalno5k set, so the mAP is high on val2017 test.

**NOTE**: star = DAMO-YOLO model trained with distillation.

**NOTE**: The V100 GPU decoder max out at 625-635 FPS on DeepStream even using lighter models.

**NOTE**: The GPU bbox parser is a bit slower than CPU bbox parser on V100 GPU tests.

| DeepStream         | Precision | Resolution | IoU=0.5:0.95 | IoU=0.5 | IoU=0.75 | FPS<br />(without display) |
|:------------------:|:---------:|:----------:|:------------:|:-------:|:--------:|:--------------------------:|
| YOLO-NAS L         | FP16      | 640        | 0.484        | 0.658   | 0.532    | 235.27                     |
| YOLO-NAS M         | FP16      | 640        | 0.480        | 0.651   | 0.524    | 287.39                     |
| YOLO-NAS S         | FP16      | 640        | 0.442        | 0.614   | 0.485    | 478.52                     |
| PP-YOLOE+_x        | FP16      | 640        | 0.528        | 0.705   | 0.579    | 121.17                     |
| PP-YOLOE+_l        | FP16      | 640        | 0.511        | 0.686   | 0.557    | 191.82                     |
| PP-YOLOE+_m        | FP16      | 640        | 0.483        | 0.658   | 0.528    | 264.39                     |
| PP-YOLOE+_s        | FP16      | 640        | 0.424        | 0.594   | 0.464    | 476.13                     |
| PP-YOLOE-s (400)   | FP16      | 640        | 0.423        | 0.589   | 0.463    | 461.23                     |
| DAMO-YOLO-L star   | FP16      | 640        | 0.502        | 0.674   | 0.551    | 176.93                     |
| DAMO-YOLO-M star   | FP16      | 640        | 0.485        | 0.656   | 0.530    | 242.24                     |
| DAMO-YOLO-S star   | FP16      | 640        | 0.460        | 0.631   | 0.502    | 385.09                     |
| DAMO-YOLO-S        | FP16      | 640        | 0.445        | 0.611   | 0.486    | 378.68                     |
| DAMO-YOLO-T star   | FP16      | 640        | 0.419        | 0.586   | 0.455    | 492.24                     |
| DAMO-YOLO-Nl       | FP16      | 416        | 0.392        | 0.559   | 0.423    | 483.73                     |
| DAMO-YOLO-Nm       | FP16      | 416        | 0.371        | 0.532   | 0.402    | 555.94                     |
| DAMO-YOLO-Ns       | FP16      | 416        | 0.312        | 0.460   | 0.335    | 627.67                     |
| YOLOX-x            | FP16      | 640        | 0.447        | 0.616   | 0.483    | 125.40                     |
| YOLOX-l            | FP16      | 640        | 0.430        | 0.598   | 0.466    | 193.10                     |
| YOLOX-m            | FP16      | 640        | 0.397        | 0.566   | 0.431    | 298.61                     |
| YOLOX-s            | FP16      | 640        | 0.335        | 0.502   | 0.365    | 522.05                     |
| YOLOX-s legacy     | FP16      | 640        | 0.375        | 0.569   | 0.407    | 518.52                     |
| YOLOX-Darknet      | FP16      | 640        | 0.414        | 0.595   | 0.453    | 212.88                     |
| YOLOX-Tiny         | FP16      | 640        | 0.274        | 0.427   | 0.292    | 633.95                     |
| YOLOX-Nano         | FP16      | 640        | 0.212        | 0.342   | 0.222    | 633.04                     |
| YOLOv8x            | FP16      | 640        | 0.499        | 0.669   | 0.545    | 130.49                     |
| YOLOv8l            | FP16      | 640        | 0.491        | 0.660   | 0.535    | 180.75                     |
| YOLOv8m            | FP16      | 640        | 0.468        | 0.637   | 0.510    | 278.08                     |
| YOLOv8s            | FP16      | 640        | 0.415        | 0.578   | 0.453    | 493.45                     |
| YOLOv8n            | FP16      | 640        | 0.343        | 0.492   | 0.373    | 627.43                     |
| YOLOv7-u6          | FP16      | 640        | 0.484        | 0.652   | 0.530    | 193.54                     |
| YOLOv7x*           | FP16      | 640        | 0.496        | 0.679   | 0.536    | 155.07                     |
| YOLOv7*            | FP16      | 640        | 0.476        | 0.660   | 0.518    | 226.01                     |
| YOLOv7-Tiny Leaky* | FP16      | 640        | 0.345        | 0.516   | 0.372    | 626.23                     |
| YOLOv7-Tiny Leaky* | FP16      | 416        | 0.328        | 0.493   | 0.349    | 633.90                     |
| YOLOv6-L 4.0       | FP16      | 640        | 0.490        | 0.671   | 0.535    | 178.41                     |
| YOLOv6-M 4.0       | FP16      | 640        | 0.460        | 0.635   | 0.502    | 293.39                     |
| YOLOv6-S 4.0       | FP16      | 640        | 0.416        | 0.585   | 0.453    | 513.90                     |
| YOLOv6-N 4.0       | FP16      | 640        | 0.349        | 0.503   | 0.378    | 633.37                     |
| YOLOv5x 7.0        | FP16      | 640        | 0.471        | 0.652   | 0.513    | 149.93                     |
| YOLOv5l 7.0        | FP16      | 640        | 0.455        | 0.637   | 0.497    | 235.55                     |
| YOLOv5m 7.0        | FP16      | 640        | 0.421        | 0.604   | 0.459    | 351.69                     |
| YOLOv5s 7.0        | FP16      | 640        | 0.344        | 0.529   | 0.372    | 618.13                     |
| YOLOv5n 7.0        | FP16      | 640        | 0.247        | 0.414   | 0.257    | 629.66                     |
