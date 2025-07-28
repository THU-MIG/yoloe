# DeepStream-Yolo-Seg

NVIDIA DeepStream SDK application for YOLO-Seg models

--------------------------------------------------------------------------------------------------
### YOLO objetct detection models and other infos: https://github.com/marcoslucianops/DeepStream-Yolo
--------------------------------------------------------------------------------------------------

### Getting started

* [Supported models](#supported-models)
* [Instructions](#basic-usage)
* [YOLOv5-Seg usage](docs/YOLOv5_Seg.md)
* [YOLOv7-Seg usage](docs/YOLOv7_Seg.md)
* [YOLOv8-Seg usage](docs/YOLOv8_Seg.md)
* [NMS configuration](#nms-configuration)
* [Detection threshold configuration](#detection-threshold-configuration)

##

### Supported models

* [YOLOv8-Seg](https://github.com/ultralytics/ultralytics)
* [YOLOv7-Seg](https://github.com/WongKinYiu/yolov7)
* [YOLOv5-Seg](https://github.com/ultralytics/yolov5)

##

### Instructions

#### 1. Download the DeepStream-Yolo-Seg repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo-Seg.git
cd DeepStream-Yolo-Seg
```

#### 2. Compile the libs

* DeepStream 6.3 on x86 platform

  ```
  CUDA_VER=12.1 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.2 on x86 platform

  ```
  CUDA_VER=11.8 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.1.1 on x86 platform

  ```
  CUDA_VER=11.7 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.1 on x86 platform

  ```
  CUDA_VER=11.6 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.0.1 / 6.0 on x86 platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 on Jetson platform

  ```
  CUDA_VER=11.4 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

* DeepStream 6.0.1 / 6.0 on Jetson platform

  ```
  CUDA_VER=10.2 make -C nvdsinfer_custom_impl_Yolo_seg
  ```

#### 3. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

##

### NMS configuration

For now, the NMS is configured in the ONNX exporter file.

**NOTE**: Make sure to set `cluster-mode=4` in the config_infer file.

##

### Detection threshold configuration

The minimum detection confidence threshold is configured in the ONNX exporter file. The `pre-cluster-threshold` should be >= the value used in the ONNX model.

```
[class-attrs-all]
pre-cluster-threshold=0.25
topk=100
```

##

My projects: https://www.youtube.com/MarcosLucianoTV
