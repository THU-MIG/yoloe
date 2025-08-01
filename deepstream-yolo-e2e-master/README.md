
<div align="center">
  <p>
    <a href="https://github.com/levipereira/deepstream-yolo-e2e" target="_blank">
      <img width="100%" src="https://github.com/levipereira/deepstream-yolo-e2e/releases/download/v1.0/banner_yolo_e2e.jpg" alt="DeepStream YOLO  banner"></a>
  </p>
</div>  

<div align="center">

<h3><span style="color:#76B900;">DEEPSTREAM</span> & <span style="color:blueviolet;">YOLO E2E</span></h3>

<p>Implementation of End-to-End YOLO Detection and Segmentation Models for DeepStream</p>

<p>This repository offers an optimized implementation of End-to-End YOLO models for DeepStream, enhancing inference efficiency by integrating Non-Maximum Suppression (NMS) directly into the YOLO models. This approach supports dynamic batch sizes and input sizes, providing seamless adaptability.</p>

<h3>🎥 Special Feature: Direct <span style="color:red;">YouTube</span> Video Integration</h3>

</div>
<div align="justify"> 
Now, you can integrate <strong>videos directly from YouTube</strong> into your pipeline. 📹 This feature enables seamless streaming and processing of YouTube videos, providing an expanded range of input sources for real-time analytics and AI-driven insights. 🌐 It’s perfect for scenarios where accessing online video data is essential, opening up new possibilities for multimedia applications. 

> **Note:** This feature is available only in the Python application.
</div>
<div align="center"> 

### DeepStream Version Supported
| DeepStream Version | dGPU/X86 | Jetson |
|--------------------|----------|--------|
| 7.1                | ✅        | ✅     |
| 7.0                | ✅        | ✅      |
| 6.4                | ✅        | ✅      |
| 6.3                | ✅        | ✅      |
| 6.2                | ✅        | ✅      |
| 6.1                | ❌        | ❌      |

<div align="justify"> 


</div> 

<div align="justify"> 

# Prerequisites

## 1. NVIDIA GPU CARD
Make sure you have an Nvidia GPU installed on your system and that the latest drivers are properly configured. 
Download and install the GPU drivers from the official Nvidia website:
[Nvidia Drivers Download](https://www.nvidia.com/en-us/drivers/)

## 2. Supported OS / Platform
- **Windows 11**
  * Enable WSL2 in Windows and install Linux(Ubuntu) from the Microsoft Store.
  * [DeepStream On WSL](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_on_WSL2.html)
- **Linux (x86_64)**
- **NVIDIA Jetson ARM**
  
## 3. Docker 
Docker is required for creating and managing containers, simplifying development and deployment. 
To install Docker on Ubuntu, use the convenience script:<br>
[Docker Installation Guide for Ubuntu](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

After the installation, add your user to the docker group to run Docker commands without sudo:
```bash
sudo usermod -aG docker $USER
```
 ## 4. NVIDIA Container Toolkit
The NVIDIA Container Toolkit allows Docker containers to utilize the Nvidia GPU to accelerate your applications. 
To install the toolkit, follow the official guide:

[NVIDIA Container Toolkit Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

After installation, verify that the setup is correct by running a GPU-enabled container:
```bash
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04  nvidia-smi
```

</div>



<div align="center"> 
 
#### Models Available

| Dataset | Model    | Feature      | Dynamic Shape | Dynamic Batch Size | NMS-Free | Efficient NMS / RoiAlign |
|---------|----------|--------------|:-------------:|:-----------------:|:--------:|:-------------------------:|
| COCO    | YOLO11   | Detection    |       ✅       |         ✅         |     <span style="color:red">🚫</span>    |     ✅                   |
| COCO    | YOLOv10  | Detection    |       ✅       |         ✅         |     ✅    |   <span style="color:red">🚫</span>   |
| COCO    | YOLOv9   | Detection    |       ✅       |         ✅         |     ✅    |       ✅                 |
| COCO    | YOLOv8   | Detection    |       ✅       |         ✅         | <span style="color:red">🚫</span> |       ✅                 |
| COCO    | YOLOv7   | Detection    |       ✅       |         ✅         | <span style="color:red">🚫</span> |       ✅                 |
| COCO    | YOLO11   | Segmentation |       ✅       |         ✅         |     <span style="color:red">🚫</span>    |     ✅                   |
| COCO    | YOLOv9   | Segmentation |       ✅       |         ✅         | <span style="color:red">🚫</span> |            ✅            |
| COCO    | YOLOv8   | Segmentation |       ✅       |         ✅         | <span style="color:red">🚫</span> |            ✅            |
| COCO    | YOLOv7   | Segmentation |       ✅       |         ✅         | <span style="color:red">🚫</span> |            ✅            |
| WIDER FACE    | YOLO11   | Detection    |       ✅       |         ✅         |     <span style="color:red">🚫</span>    |     ✅                   |
| WIDER FACE    | YOLOv10  | Detection    |       ✅       |         ✅         |     ✅    |   <span style="color:red">🚫</span>   |
| WIDER FACE    | YOLOv8   | Detection    |       ✅       |         ✅         |     ✅    |       ✅                 |

</div>



<table align="center" border="1" cellpadding="10">
    <tr>
        <th colspan="2">Features</th>
    </tr>
    <tr>
        <td><strong>Dynamic Shapes</strong></td>
        <td>TensorRT enables the creation of network resolutions different from the original exported ONNX.</td>
    </tr>
    <tr>
        <td><strong>Dynamic Batch Size</strong></td>
        <td>Dynamically adjusts the batch size to maximize model performance according to the number of deepstream sources.</td>
    </tr>
    <tr>
        <td><strong>NMS-Free</strong></td>
        <td>Models natively implement NMS-Free, available for some YOLOv9 models and all YOLOv10 detection models.</td>
    </tr>
    <tr>
        <td><strong>TensorRT Plugins</strong></td>
        <td>TensorRT <code>EfficientNMS</code> plugin for detection models, and <code>EfficientNMSX / ROIAlign</code> plugins for segmentation models.</td>
    </tr>
</table>

<br>
 
<div align="justify"> 

# Project Workflow 

This project involves important steps as outlined below:

### 1. Clone Repo and Submodules
```bash
git clone https://github.com/levipereira/deepstream-yolo-e2e.git
cd deepstream-yolo-e2e
git submodule update --init --recursive
```

### 2. **Run Deepstream Container**
In this example, we will use **DeepStream 7.1**.

Start the docker container from `deepstream-yolo-e2e` dir:
>Note: If you are not using the computer's display output, especially on Jetson devices, remove the `-e DISPLAY=$DISPLAY` and  `-v /tmp/.X11-unix/:/tmp/.X11-unix` options, as they may cause errors.
#### 2.1 Windows WSL
```bash
xhost +
docker run \
        -it \
        --privileged \
        --rm \
        --net=host \
        --ipc=host \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -e CUDA_CACHE_DISABLE=0 \
        --device /dev/snd \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v `pwd`:/apps/deepstream-yolo-e2e \
        -w /apps/deepstream-yolo-e2e \
        nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

#### 2.2 Linux
```bash
xhost +
docker run \
        -it \
        --privileged \
        --rm \
        --net=host \
        --ipc=host \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v `pwd`:/apps/deepstream-yolo-e2e \
        -w /apps/deepstream-yolo-e2e \
        nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```

#### 2.3 Jetson
```bash
xhost +
docker run \
        -it \
        --privileged \
        --rm \
        --net=host \
        --ipc=host \
        --runtime nvidia \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix \
        -v `pwd`:/apps/deepstream-yolo-e2e \
        -w /apps/deepstream-yolo-e2e \
        nvcr.io/nvidia/deepstream:7.1-triton-multiarch
```


### 3. **Install packages**
```bash
cd /apps/deepstream-yolo-e2e
bash /apps/deepstream-yolo-e2e/one_hit_install.sh
```
 
### 4. Run DeepStream Application
```bash
cd /apps/deepstream-yolo-e2e
./deepstream.py
```

### 🚀 Important Tip 🚀

The model with the highest performance and accuracy is **YOLOv9-QAT (ReLU)**. This quantized model delivers exceptional results and supports multiple sources, depending on your GPU capabilities.

You can find it in the model selection menu:
- **Coco > Detection > Balanced > YOLOv9 QAT (ReLU)**

 
## General Configuration Settings
The application can be configured using the [`config/python_app/config.ini`](config/python_app/config.ini) file. Below are the key settings you can modify:


### Configuration Parameters

- **MUXER_OUTPUT_WIDTH and MUXER_OUTPUT_HEIGHT**: These parameters define the dimensions (width and height) of the output video stream produced by the muxer.

- **TILED_OUTPUT_WIDTH and TILED_OUTPUT_HEIGHT**: These parameters specify the dimensions of the tiled output format.  

- **OUTPUT_DIRECTORY**: This is the directory where output files will be saved.
 
- **OUTPUT_PREFIX**: This parameter specifies the prefix for the output file names.

- **RTSP_PORT**: This is the port used for the RTSP stream. The default value is `8554`.

- **RTSP_FACTORY**: This represents the path for the RTSP stream. For example, setting it to `/live` allows for streaming under this path.

- **RTSP_UDPSYNC**: This is the internal port used by DeepStream to connect to the RTSP server. The default value is 8255.

**RTSP URL Format** <br>When constructing the RTSP URL, it will always follow this format:
```
rtsp://<server_ip>:<RTSP_PORT><RTSP_FACTORY>
```
For example, if you use the default settings, the URL would be:

```bash
rtsp://<server_ip>:8554/live
```
 
<h3>References</h3>
<ul>
    <li>
        <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLO11</a>: Official repository for the YOLO11 model.
    </li>
    <li>
        <a href="https://github.com/THU-MIG/yolov10" target="_blank">YOLOv10</a>: The official repository for YOLOv10.
    </li>
    <li>
        <a href="https://github.com/WongKinYiu/yolov9" target="_blank">YOLOv9</a>: Access the official YOLOv9 repository.
    </li>
    <li>
        <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8</a>: Official repository for the YOLOv8 model.
    </li>
    <li>
        <a href="https://github.com/WongKinYiu/yolov7" target="_blank">YOLOv7</a>: Explore the YOLOv7 repository.
    </li>
    <li>
        <a href="https://github.com/akanametov/yolo-face" target="_blank">YOLO-FACE</a>: Dedicated repository for the YOLO-FACE model.
    </li>
    <li>
        <a href="https://github.com/levipereira/ultralytics" target="_blank">Export YOLO 11/v10/v8</a>: This repository for exporting YOLO11, v10, and v8 models with End2End.
    </li>
    <li>
        <a href="https://github.com/levipereira/yolov9-qat" target="_blank">YOLOv9 QAT</a>: Official repository for quantization-aware training (QAT) of the YOLOv9 model,.
    </li>
    <li>
        <a href="https://github.com/levipereira/nvdsinfer_yolo" target="_blank">NvDsInferYolo</a>: Explore the official repository for the NvDsInferYolo parsing function.
    </li>
    <li>
        <a href="https://github.com/levipereira/yolo_e2e" target="_blank">YOLO E2E</a>: This repository focuses on exporting YOLO models with End2End capabilities.
    </li>
    <li>
        <a href="https://github.com/levipereira/TensorRT" target="_blank">TensorRT Plugin EfficientNMSX</a>: Official repository for the EfficientNMSX plugin implemented in TensorRT.
    </li>
</ul>
