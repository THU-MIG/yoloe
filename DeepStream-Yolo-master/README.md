# DeepStream-Yolo

NVIDIA DeepStream SDK 7.1 / 7.0 / 6.4 / 6.3 / 6.2 / 6.1.1 / 6.1 / 6.0.1 / 6.0 / 5.1  configuration for YOLO models

--------------------------------------------------------------------------------------------------
### For now, I am limited for some updates. Thank you for understanding.
--------------------------------------------------------------------------------------------------
### YOLO-Pose: https://github.com/marcoslucianops/DeepStream-Yolo-Pose
### YOLO-Seg: https://github.com/marcoslucianops/DeepStream-Yolo-Seg
### YOLO-Face: https://github.com/marcoslucianops/DeepStream-Yolo-Face
--------------------------------------------------------------------------------------------------
### Important: please export the ONNX model with the new export file, generate the TensorRT engine again with the updated files, and use the new config_infer_primary file according to your model
--------------------------------------------------------------------------------------------------

### Improvements on this repository

* Support for INT8 calibration
* Support for non square models
* Models benchmarks
* Support for Darknet models (YOLOv4, etc) using cfg and weights conversion with GPU post-processing
* Support for D-FINE, RT-DETR, CO-DETR (MMDetection), YOLO-NAS, PPYOLOE+, PPYOLOE, DAMO-YOLO, Gold-YOLO, RTMDet (MMYOLO), YOLOX, YOLOR, YOLO11, YOLOv10, YOLOv9, YOLOv8, YOLOv7, YOLOv6, YOLOv5u and YOLOv5 using ONNX conversion with GPU post-processing
* GPU bbox parser
* Custom ONNX model parser
* Dynamic batch-size
* INT8 calibration (PTQ) for Darknet and ONNX exported models

##

### Getting started

* [Requirements](#requirements)
* [Supported models](#supported-models)
* [Benchmarks](docs/benchmarks.md)
* [dGPU installation](docs/dGPUInstalation.md)
* [Basic usage](#basic-usage)
* [Docker usage](#docker-usage)
* [NMS configuration](#nms-configuration)
* [Notes](#notes)
* [INT8 calibration](docs/INT8Calibration.md)
* [YOLOv5 usage](docs/YOLOv5.md)
* [YOLOv5u usage](docs/YOLOv5u.md)
* [YOLOv6 usage](docs/YOLOv6.md)
* [YOLOv7 usage](docs/YOLOv7.md)
* [YOLOv8 usage](docs/YOLOv8.md)
* [YOLOv9 usage](docs/YOLOv9.md)
* [YOLOv10 usage](docs/YOLOv10.md)
* [YOLO11 usage](docs/YOLO11.md)
* [YOLOR usage](docs/YOLOR.md)
* [YOLOX usage](docs/YOLOX.md)
* [RTMDet (MMYOLO) usage](docs/RTMDet.md)
* [Gold-YOLO usage](docs/GoldYOLO.md)
* [DAMO-YOLO usage](docs/DAMOYOLO.md)
* [PP-YOLOE / PP-YOLOE+ usage](docs/PPYOLOE.md)
* [YOLO-NAS usage](docs/YOLONAS.md)
* [CO-DETR (MMDetection) usage](docs/CODETR.md)
* [RT-DETR PyTorch usage](docs/RTDETR_PyTorch.md)
* [RT-DETR Paddle usage](docs/RTDETR_Paddle.md)
* [RT-DETR Ultralytics usage](docs/RTDETR_Ultralytics.md)
* [D-FINE usage](docs/DFINE.md)
* [Using your custom model](docs/customModels.md)
* [Multiple YOLO GIEs](docs/multipleGIEs.md)

##

### Requirements

#### DeepStream 7.1 on x86 platform

* [Ubuntu 22.04](https://releases.ubuntu.com/22.04/)
* [CUDA 12.6 Update 2](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)
* [TensorRT 10.3 GA (10.3.0.26)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 535.183.06 (Data center / Tesla series) / 560.35.03 (TITAN, GeForce RTX / GTX series and RTX / Quadro series)](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 7.1](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1)
* [GStreamer 1.20.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 7.0 on x86 platform

* [Ubuntu 22.04](https://releases.ubuntu.com/22.04/)
* [CUDA 12.2 Update 2](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)
* [TensorRT 8.6 GA (8.6.1.6)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 535 (>= 535.161.08)](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 7.0](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.0)
* [GStreamer 1.20.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.4 on x86 platform

* [Ubuntu 22.04](https://releases.ubuntu.com/22.04/)
* [CUDA 12.2 Update 2](https://developer.nvidia.com/cuda-12-2-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local)
* [TensorRT 8.6 GA (8.6.1.6)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 535 (>= 535.104.12)](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.4](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=6.4)
* [GStreamer 1.20.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.3 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 12.1 Update 1](https://developer.nvidia.com/cuda-12-1-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.5 GA Update 2 (8.5.3.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 525 (>= 525.125.06)](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.3](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=6.3)
* [GStreamer 1.16.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.2 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.5 GA Update 1 (8.5.2.2)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 525 (>= 525.85.12)](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.2](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.16.3](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.7 Update 1](https://developer.nvidia.com/cuda-11-7-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.4 GA (8.4.1.5)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 515.65.01](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on x86 platform

* [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
* [CUDA 11.6 Update 1](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local)
* [TensorRT 8.2 GA Update 4 (8.2.5.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 510.47.03](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.16.2](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.4 Update 1](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
* [TensorRT 8.0 GA (8.0.1)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
* [NVIDIA Driver 470.63.01](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 5.1 on x86 platform

* [Ubuntu 18.04](https://releases.ubuntu.com/18.04.6/)
* [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
* [TensorRT 7.2.2](https://developer.nvidia.com/nvidia-tensorrt-7x-download)
* [NVIDIA Driver 460.32.03](https://www.nvidia.com/Download/index.aspx)
* [NVIDIA DeepStream SDK 5.1](https://developer.nvidia.com/deepstream-sdk-download-tesla-archived)
* [GStreamer 1.14.5](https://gstreamer.freedesktop.org/)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 7.1 on Jetson platform

* [JetPack 6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61)
* [NVIDIA DeepStream SDK 7.1](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.1)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 7.0 on Jetson platform

* [JetPack 6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)
* [NVIDIA DeepStream SDK 7.0](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=7.0)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.4 on Jetson platform

* [JetPack 6.0 DP](https://developer.nvidia.com/embedded/jetpack-sdk-60dp)
* [NVIDIA DeepStream SDK 6.4](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=6.4)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.3 on Jetson platform

* JetPack [5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) / [5.1.2](https://developer.nvidia.com/embedded/jetpack-sdk-512)
* [NVIDIA DeepStream SDK 6.3](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files?version=6.3)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.2 on Jetson platform

* JetPack [5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) / [5.1.2](https://developer.nvidia.com/embedded/jetpack-sdk-512) / [5.1.1](https://developer.nvidia.com/embedded/jetpack-sdk-511) / [5.1](https://developer.nvidia.com/embedded/jetpack-sdk-51)
* [NVIDIA DeepStream SDK 6.2](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1.1 on Jetson platform

* [JetPack 5.0.2](https://developer.nvidia.com/embedded/jetpack-sdk-502)
* [NVIDIA DeepStream SDK 6.1.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.1 on Jetson platform

* [JetPack 5.0.1 DP](https://developer.nvidia.com/embedded/jetpack-sdk-501dp)
* [NVIDIA DeepStream SDK 6.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 6.0.1 / 6.0 on Jetson platform

* [JetPack 4.6.4](https://developer.nvidia.com/jetpack-sdk-464)
* [NVIDIA DeepStream SDK 6.0.1 / 6.0](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

#### DeepStream 5.1 on Jetson platform

* [JetPack 4.5.1](https://developer.nvidia.com/embedded/jetpack-sdk-451-archive)
* [NVIDIA DeepStream SDK 5.1](https://developer.nvidia.com/embedded/deepstream-on-jetson-downloads-archived)
* [DeepStream-Yolo](https://github.com/marcoslucianops/DeepStream-Yolo)

##

### Supported models

* [Darknet](https://github.com/AlexeyAB/darknet)
* [MobileNet-YOLO](https://github.com/dog-qiuqiu/MobileNet-Yolo)
* [YOLO-Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest)
* [YOLOv5](https://github.com/ultralytics/yolov5)
* [YOLOv5u](https://github.com/ultralytics/ultralytics)
* [YOLOv6](https://github.com/meituan/YOLOv6)
* [YOLOv7](https://github.com/WongKinYiu/yolov7)
* [YOLOv8](https://github.com/ultralytics/ultralytics)
* [YOLOv9](https://github.com/WongKinYiu/yolov9)
* [YOLOv10](https://github.com/THU-MIG/yolov10)
* [YOLO11](https://github.com/ultralytics/ultralytics)
* [YOLOR](https://github.com/WongKinYiu/yolor)
* [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [RTMDet (MMYOLO)](https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet)
* [Gold-YOLO](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO)
* [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)
* [PP-YOLOE / PP-YOLOE+](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8/configs/ppyoloe)
* [YOLO-NAS](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS.md)
* [CO-DETR (MMDetection)](https://github.com/open-mmlab/mmdetection/tree/main/projects/CO-DETR)
* [RT-DETR](https://github.com/lyuwenyu/RT-DETR)
* [D-FINE](https://github.com/Peterande/D-FINE)

##

### Basic usage

#### 1. Download the repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
```

#### 2. Download the `cfg` and `weights` files from [Darknet](https://github.com/AlexeyAB/darknet) repo to the DeepStream-Yolo folder

#### 3. Compile the lib

3.1. Set the `CUDA_VER` according to your DeepStream version

```
export CUDA_VER=XY.Z
```

* x86 platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 = 12.1
  DeepStream 6.2 = 11.8
  DeepStream 6.1.1 = 11.7
  DeepStream 6.1 = 11.6
  DeepStream 6.0.1 / 6.0 = 11.4
  DeepStream 5.1 = 11.1
  ```

* Jetson platform

  ```
  DeepStream 7.1 = 12.6
  DeepStream 7.0 / 6.4 = 12.2
  DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
  DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
  ```

3.2. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

#### 4. Edit the `config_infer_primary.txt` file according to your model (example for YOLOv4)

```
[property]
...
custom-network-config=yolov4.cfg
model-file=yolov4.weights
...
```

**NOTE**: For **Darknet** models, by default, the dynamic batch-size is set. To use static batch-size, uncomment the line

```
...
force-implicit-batch-dim=1
...
```

#### 5. Run

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: If you want to use YOLOv2 or YOLOv2-Tiny models, change the `deepstream_app_config.txt` file before run it

```
...
[primary-gie]
...
config-file=config_infer_primary_yoloV2.txt
...
```

##

### Docker usage

* x86 platform

  ```
  nvcr.io/nvidia/deepstream:7.1-gc-triton-devel
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch
  ```

* Jetson platform

  ```
  nvcr.io/nvidia/deepstream:7.1-triton-multiarch
  ```

**NOTE**: To compile the `nvdsinfer_custom_impl_Yolo`, you need to install the g++ inside the container

```
apt-get install build-essential
```

**NOTE**: With DeepStream 7.1, the docker containers do not package libraries necessary for certain multimedia operations like audio data parsing, CPU decode, and CPU encode. This change could affect processing certain video streams/files like mp4 that include audio track. Please run the below script inside the docker images to install additional packages that might be necessary to use all of the DeepStreamSDK features:

```
/opt/nvidia/deepstream/deepstream/user_additional_install.sh
```

##

### NMS Configuration

To change the `nms-iou-threshold`, `pre-cluster-threshold` and `topk` values, modify the config_infer file

```
[class-attrs-all]
nms-iou-threshold=0.45
pre-cluster-threshold=0.25
topk=300
```

**NOTE**: Make sure to set `cluster-mode=2` in the config_infer file.

##

### Notes

1. Sometimes while running gstreamer pipeline or sample apps, user can encounter error: `GLib (gthread-posix.c): Unexpected error from C library during 'pthread_setspecific': Invalid argument.  Aborting.`. The issue is caused because of a bug in `glib 2.0-2.72` version which comes with Ubuntu 22.04 by default. The issue is addressed in `glib 2.76` and its installation is required to fix the issue (https://github.com/GNOME/glib/tree/2.76.6).

    - Migrate `glib` to newer version

      ```
      pip3 install meson
      pip3 install ninja
      ```

      **NOTE**: It is recommended to use Python virtualenv.

      ```
      git clone https://github.com/GNOME/glib.git
      cd glib
      git checkout 2.76.6
      meson build --prefix=/usr
      ninja -C build/
      cd build/
      ninja install
      ```

    - Check and confirm the newly installed glib version:

      ```
      pkg-config --modversion glib-2.0
      ```

2. Sometimes with RTSP streams the application gets stuck on reaching EOS. This is because of an issue in rtpjitterbuffer component. To fix this issue, a script has been provided with required details to update gstrtpmanager library.

    ```
    /opt/nvidia/deepstream/deepstream/update_rtpmanager.sh
    ```

##

### Extract metadata

You can get metadata from DeepStream using Python and C/C++. For C/C++, you can edit the `deepstream-app` or `deepstream-test` codes. For Python, your can install and edit [deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps).

Basically, you need manipulate the `NvDsObjectMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsObjectMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/struct__NvDsObjectMeta.html)) `and NvDsFrameMeta` ([Python](https://docs.nvidia.com/metropolis/deepstream/dev-guide/python-api/PYTHON_API/NvDsMeta/NvDsFrameMeta.html) / [C/C++](https://docs.nvidia.com/metropolis/deepstream/dev-guide/sdk-api/struct__NvDsFrameMeta.html)) to get the label, position, etc. of bboxes.

##

My projects: https://www.youtube.com/MarcosLucianoTV
