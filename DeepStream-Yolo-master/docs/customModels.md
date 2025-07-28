# How to use custom models on deepstream-app

* [Directory tree](#directory-tree)
* [Compile the lib](#compile-the-lib)
* [Understanding and editing deepstream_app_config file](#understanding-and-editing-deepstream_app_config-file)
* [Understanding and editing config_infer_primary file](#understanding-and-editing-config_infer_primary-file)
* [Testing the model](#testing-the-model)

##

### Directory tree

#### 1. Download the repo

```
git clone https://github.com/marcoslucianops/DeepStream-Yolo.git
cd DeepStream-Yolo
```

#### 2. Copy the class names file to DeepStream-Yolo folder and remane it to `labels.txt`

#### 3. Copy the `onnx` or `cfg` and `weights` files to DeepStream-Yolo folder

##

### Compile the lib

1. Set the `CUDA_VER` according to your DeepStream version

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

2. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##

### Understanding and editing deepstream_app_config file

To understand and edit `deepstream_app_config.txt` file, read the [DeepStream Reference Application - Configuration Groups](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html#configuration-groups)


* tiled-display

  ```
  [tiled-display]
  enable=1
  # If you have 1 stream use 1/1 (rows/columns), if you have 4 streams use 2/2 or 4/1 or 1/4 (rows/columns)
  rows=1
  columns=1
  # Resolution of tiled display
  width=1280
  height=720
  gpu-id=0
  nvbuf-memory-type=0
  ```

* source

  * Example for 1 source:

    ```
    [source0]
    enable=1
    # 1=Camera (V4L2), 2=URI, 3=MultiURI, 4=RTSP, 5=Camera (CSI; Jetson only)
    type=3
    # Stream URL
    uri=rtsp://192.168.1.2/Streaming/Channels/101/httppreview
    # Number of sources copy (if > 1, edit rows/columns in tiled-display section; use type=3 for more than 1 source)
    num-sources=1
    gpu-id=0
    cudadec-memtype=0
    ```

  * Example for 1 duplcated source:

    ```
    [source0]
    enable=1
    type=3
    uri=rtsp://192.168.1.2/Streaming/Channels/101/
    num-sources=2
    gpu-id=0
    cudadec-memtype=0
    ```

  * Example for 2 sources:

    ```
    [source0]
    enable=1
    type=3
    uri=rtsp://192.168.1.2/Streaming/Channels/101/
    num-sources=1
    gpu-id=0
    cudadec-memtype=0

    [source1]
    enable=1
    type=3
    uri=rtsp://192.168.1.3/Streaming/Channels/101/
    num-sources=1
    gpu-id=0
    cudadec-memtype=0
    ```

* sink

  ```
  [sink0]
  enable=1
  # 1=Fakesink, 2=EGL (nveglglessink), 3=Filesink, 4=RTSP, 5=Overlay (Jetson only)
  type=2
  # Indicates how fast the stream is to be rendered (0=As fast as possible, 1=Synchronously)
  sync=0
  gpu-id=0
  nvbuf-memory-type=0
  ```

* streammux

  ```
  [streammux]
  gpu-id=0
  # Boolean property to inform muxer that sources are live
  live-source=1
  batch-size=1
  batched-push-timeout=40000
  # Resolution of streammux
  width=1920
  height=1080
  enable-padding=0
  nvbuf-memory-type=0
  ```

* primary-gie

  ```
  [primary-gie]
  enable=1
  gpu-id=0
  gie-unique-id=1
  nvbuf-memory-type=0
  config-file=config_infer_primary.txt
  ```

  **NOTE**: Edit the `config-file` according to your YOLO model.

##

### Understanding and editing config_infer_primary file

To understand and edit `config_infer_primary.txt` file, read the [DeepStream Plugin Guide - Gst-nvinfer File Configuration Specifications](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvinfer.html#gst-nvinfer-file-configuration-specifications)

* model-color-format

  ```
  # 0=RGB, 1=BGR, 2=GRAYSCALE
  model-color-format=0
  ```

  **NOTE**: Set it according to the number of channels in the `cfg` file (1=GRAYSCALE, 3=RGB for Darknet YOLO) or your model configuration (ONNX).

* custom-network-config and model-file (Darknet YOLO)

  * Example for custom YOLOv4 model

    ```
    custom-network-config=yolov4_custom.cfg
    model-file=yolov4_custom.weights
    ```

* onnx-file (ONNX)

  * Example for custom YOLOv8 model

    ```
    onnx-file=yolov8s_custom.onnx
    ```

* model-engine-file 

  * Example for `batch-size=1` and `network-mode=2`

    ```
    model-engine-file=model_b1_gpu0_fp16.engine
    ```

  * Example for `batch-size=1` and `network-mode=1`

    ```
    model-engine-file=model_b1_gpu0_int8.engine
    ```

  * Example for `batch-size=1` and `network-mode=0`

    ```
    model-engine-file=model_b1_gpu0_fp32.engine
    ```

  * Example for `batch-size=2` and `network-mode=0`

    ```
    model-engine-file=model_b2_gpu0_fp32.engine
    ```

  **NOTE**: To change the generated engine filename (Darknet YOLO), you need to edit and rebuild the `nvdsinfer_model_builder.cpp` file (`/opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer/nvdsinfer_model_builder.cpp`, lines 825-827)

  ```
  suggestedPathName =
      modelPath + "_b" + std::to_string(initParams.maxBatchSize) + "_" +
      devId + "_" + networkMode2Str(networkMode) + ".engine";
  ```

* batch-size

  ```
  batch-size=1
  ```

* network-mode

  ```
  # 0=FP32, 1=INT8, 2=FP16
  network-mode=0
  ```

* num-detected-classes 

  ```
  num-detected-classes=80
  ```

  **NOTE**: Set it according to number of classes in `cfg` file (Darknet YOLO) or your model configuration (ONNX).

* interval

  ```
  # Number of consecutive batches to be skipped
  interval=0
  ```

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```
