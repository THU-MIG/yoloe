# YOLOv8-Seg usage

**NOTE**: The yaml file is not required.

* [Convert model](#convert-model)
* [Edit the config_infer_primary_yoloV8_seg file](#edit-the-config_infer_primary_yolov8_seg-file)

##

### Convert model

#### 1. Download the YOLOv8 repo and install the requirements

```
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip3 install -r requirements.txt
python3 setup.py install
pip3 install onnx onnxsim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yoloV8_seg.py` file from `DeepStream-Yolo-Seg/utils` directory to the `ultralytics` folder.

#### 3. Download the model

Download the `pt` file from [YOLOv8](https://github.com/ultralytics/assets/releases/) releases (example for YOLOv8s-Seg)

```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOv8s-Seg)

```
python3 export_yoloV8_seg.py -w yolov8s-seg.pt --dynamic
```

**NOTE**: Confidence threshold (example for conf-thres = 0.25)

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
--conf-thres 0.25
```

**NOTE**: NMS IoU threshold (example for iou-thres = 0.45)

```
--iou-thres 0.45
```

**NOTE**: Maximum detections (example for max-det = 100)

```
--max-det 100
```

**NOTE**: To change the inference size (defaut: 640)

```
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```
-s 1280
```

or

```
-s 1280 1280
```

**NOTE**: To simplify the ONNX model

```
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```
--batch 4
```

#### 5. Copy generated files

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo-Seg` folder.

##

### Edit the config_infer_primary_yoloV8_seg file

Edit the `config_infer_primary_yoloV8_seg.txt` file according to your model (example for YOLOv8s-Seg)

```
[property]
...
onnx-file=yolov8s-seg.onnx
model-engine-file=yolov8s-seg.onnx_b1_gpu0_fp32.engine
...
```

**NOTE**: To output the masks, use

```
[property]
...
output-instance-mask=1
segmentation-threshold=0.5
...
```

**NOTE**: The **YOLOv8-Seg** resizes the input with center padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=1
...
```
