# YOLOX usage

**NOTE**: You can use the main branch of the YOLOX repo to convert all model versions.

* [Convert model](#convert-model)
* [Compile the lib](#compile-the-lib)
* [Edit the config_infer_primary_yolox file](#edit-the-config_infer_primary_yolox-file)
* [Edit the deepstream_app_config file](#edit-the-deepstream_app_config-file)
* [Testing the model](#testing-the-model)

##

### Convert model

#### 1. Download the YOLOX repo and install the requirements

```
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -r requirements.txt
python3 setup.py develop
pip3 install onnx onnxslim onnxruntime
```

**NOTE**: It is recommended to use Python virtualenv.

#### 2. Copy conversor

Copy the `export_yolox.py` file from `DeepStream-Yolo/utils` directory to the `YOLOX` folder.

#### 3. Download the model

Download the `pth` file from [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX/releases/) releases (example for YOLOX-s)

```
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
```

**NOTE**: You can use your custom model.

#### 4. Convert model

Generate the ONNX model file (example for YOLOX-s)

```
python3 export_yolox.py -w yolox_s.pth -c exps/default/yolox_s.py --dynamic
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

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

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 11.

```
--opset 12
```

#### 5. Copy generated file

Copy the generated ONNX model file to the `DeepStream-Yolo` folder.

##

### Compile the lib

1. Open the `DeepStream-Yolo` folder and compile the lib

2. Set the `CUDA_VER` according to your DeepStream version

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

3. Make the lib

```
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

##

### Edit the config_infer_primary_yolox file

Edit the `config_infer_primary_yolox.txt` file according to your model (example for YOLOX-s with 80 classes)

```
[property]
...
onnx-file=yolox_s.pth.onnx
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: If you are using the **legacy** model, you should edit the `config_infer_primary_yolox_legacy.txt` file.

**NOTE**: The **YOLOX and YOLOX legacy** resize the input with left/top padding. To get better accuracy, use

```
[property]
...
maintain-aspect-ratio=1
symmetric-padding=0
...
```

**NOTE**: The **YOLOX** uses BGR color format for the image input. It is important to change the `model-color-format` according to the trained values.

```
[property]
...
model-color-format=1
...
```

**NOTE**: The **YOLOX legacy** uses RGB color format for the image input. It is important to change the `model-color-format` according to the trained values.

```
[property]
...
model-color-format=0
...
```

**NOTE**: The **YOLOX** uses no normalization on the image preprocess. It is important to change the `net-scale-factor` according to the trained values.

```
[property]
...
net-scale-factor=1
...
```

**NOTE**: The **YOLOX legacy** uses normalization on the image preprocess. It is important to change the `net-scale-factor` and `offsets` according to the trained values.

Default: `mean = 0.485, 0.456, 0.406` and `std = 0.229, 0.224, 0.225`

```
[property]
...
net-scale-factor=0.0173520735727919486
offsets=123.675;116.28;103.53
...
```

##

### Edit the deepstream_app_config file

```
...
[primary-gie]
...
config-file=config_infer_primary_yolox.txt
```

**NOTE**: If you are using the **legacy** model, you should edit it to `config_infer_primary_yolox_legacy.txt`.

##

### Testing the model

```
deepstream-app -c deepstream_app_config.txt
```

**NOTE**: The TensorRT engine file may take a very long time to generate (sometimes more than 10 minutes).

**NOTE**: For more information about custom models configuration (`batch-size`, `network-mode`, etc), please check the [`docs/customModels.md`](customModels.md) file.
