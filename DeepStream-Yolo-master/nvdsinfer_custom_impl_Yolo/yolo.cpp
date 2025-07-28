/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Edited by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "NvOnnxParser.h"

#include "yolo.h"
#include "yoloPlugins.h"

#ifdef OPENCV
#include "calibrator.h"
#endif

Yolo::Yolo(const NetworkInfo& networkInfo) : m_InputBlobName(networkInfo.inputBlobName),
    m_NetworkType(networkInfo.networkType), m_ModelName(networkInfo.modelName),
    m_OnnxFilePath(networkInfo.onnxFilePath), m_WtsFilePath(networkInfo.wtsFilePath),
    m_CfgFilePath(networkInfo.cfgFilePath), m_BatchSize(networkInfo.batchSize),
    m_ImplicitBatch(networkInfo.implicitBatch), m_Int8CalibPath(networkInfo.int8CalibPath),
    m_DeviceType(networkInfo.deviceType), m_NumDetectedClasses(networkInfo.numDetectedClasses),
    m_ClusterMode(networkInfo.clusterMode), m_NetworkMode(networkInfo.networkMode),
    m_ScaleFactor(networkInfo.scaleFactor), m_Offsets(networkInfo.offsets), m_WorkspaceSize(networkInfo.workspaceSize),
    m_InputFormat(networkInfo.inputFormat), m_InputC(0), m_InputH(0), m_InputW(0), m_InputSize(0), m_NumClasses(0),
    m_LetterBox(0), m_NewCoords(0), m_YoloCount(0)
{
}

Yolo::~Yolo()
{
  destroyNetworkUtils();
}

nvinfer1::ICudaEngine* 
#if NV_TENSORRT_MAJOR >= 8
Yolo::createEngine(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config)
#else
Yolo::createEngine(nvinfer1::IBuilder* builder)
#endif

{
  assert(builder);

#if NV_TENSORRT_MAJOR < 8
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  if (m_WorkspaceSize > 0) {
    config->setMaxWorkspaceSize((size_t) m_WorkspaceSize * 1024 * 1024);
  }
#endif

  nvinfer1::NetworkDefinitionCreationFlags flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flags);
  assert(network);

  nvonnxparser::IParser* parser;

  if (m_NetworkType == "onnx") {

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 0)
    parser = nvonnxparser::createParser(*network, *builder->getLogger());
#else
    parser = nvonnxparser::createParser(*network, logger);
#endif

    if (!parser->parseFromFile(m_OnnxFilePath.c_str(), static_cast<INT>(nvinfer1::ILogger::Severity::kWARNING))) {
      std::cerr << "\nCould not parse the ONNX file\n" << std::endl;

#if NV_TENSORRT_MAJOR >= 8
      delete parser;
      delete network;
#else
      parser->destroy();
      config->destroy();
      network->destroy();
#endif

      return nullptr;
    }
    m_InputC = network->getInput(0)->getDimensions().d[1];
    m_InputH = network->getInput(0)->getDimensions().d[2];
    m_InputW = network->getInput(0)->getDimensions().d[3];
  }
  else {
    m_ConfigBlocks = parseConfigFile(m_CfgFilePath);
    parseConfigBlocks();
    if (parseModel(*network) != NVDSINFER_SUCCESS) {

#if NV_TENSORRT_MAJOR >= 8
      delete network;
#else
      config->destroy();
      network->destroy();
#endif

      return nullptr;
    }
  }

  if ((m_NetworkType == "darknet" && !m_ImplicitBatch) || network->getInput(0)->getDimensions().d[0] == -1) {
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    assert(profile);
    for (INT i = 0; i < network->getNbInputs(); ++i) {
      nvinfer1::ITensor* input = network->getInput(i);
      nvinfer1::Dims inputDims = input->getDimensions();
      nvinfer1::Dims dims = inputDims;
      dims.d[0] = 1;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, dims);
      dims.d[0] = m_BatchSize;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, dims);
      dims.d[0] = m_BatchSize;
      profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
    config->addOptimizationProfile(profile);
  }

  std::cout << "\nBuilding the TensorRT Engine\n" << std::endl;

  if (m_NetworkType == "darknet") {
    if (m_NumClasses != m_NumDetectedClasses) {
      std::cout << "NOTE: Number of classes mismatch, make sure to set num-detected-classes=" << m_NumClasses
          << " on the config_infer file\n" << std::endl;
    }
    if (m_LetterBox == 1) {
        std::cout << "NOTE: letter_box is set in cfg file, make sure to set maintain-aspect-ratio=1 on the " <<
            "config_infer file to get better accuracy\n" << std::endl;
    }
  }
  if (m_ClusterMode != 2 && m_ClusterMode != 4) {
      std::cout << "NOTE: Wrong cluster-mode is set, make sure to set cluster-mode=4 (YOLOv10, RT-DETR or custom " <<
          "NMS) or cluster-mode=2 on the config_infer file\n" << std::endl;
  }

  if (m_NetworkMode == "FP16") {
    assert(builder->platformHasFastFp16());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  else if (m_NetworkMode == "INT8") {
    assert(builder->platformHasFastInt8());
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (m_Int8CalibPath != "") {

#ifdef OPENCV
      fileExists(m_Int8CalibPath);

      std::string calib_image_list;
      int calib_batch_size;
      if (getenv("INT8_CALIB_IMG_PATH")) {
        calib_image_list = getenv("INT8_CALIB_IMG_PATH");
      }
      else {
        std::cerr << "INT8_CALIB_IMG_PATH not set" << std::endl;
        assert(0);
      }
      if (getenv("INT8_CALIB_BATCH_SIZE")) {
        calib_batch_size = std::stoi(getenv("INT8_CALIB_BATCH_SIZE"));
      }
      else {
        std::cerr << "INT8_CALIB_BATCH_SIZE not set" << std::endl;
        assert(0);
      }
      nvinfer1::IInt8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(calib_batch_size, m_InputC, m_InputH,
          m_InputW, m_ScaleFactor, m_Offsets, m_InputFormat, calib_image_list, m_Int8CalibPath);
      config->setInt8Calibrator(calibrator);
#else
      assert(0 && "OpenCV is required to run INT8 calibrator\n");
#endif

    }
  }

#ifdef GRAPH
  config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
#endif

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR > 0)
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(*builder->getLogger());
#else
  nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
#endif

  assert(runtime);

  nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);

  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
  if (engine) {
    std::cout << "Building complete\n" << std::endl;
  }
  else {
    std::cerr << "Building engine failed\n" << std::endl;
  }

#if NV_TENSORRT_MAJOR >= 8
  delete serializedEngine;
#else
  serializedEngine->destroy();
#endif

#ifdef GRAPH
  nvinfer1::IExecutionContext *context = engine->createExecutionContext();
  nvinfer1::IEngineInspector *inpector = engine->createEngineInspector();
  inpector->setExecutionContext(context);
  std::ofstream graph;
  graph.open("graph.json");
  graph << inpector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
  graph.close();
  std::cout << "Network graph saved to graph.json\n" << std::endl;

#if NV_TENSORRT_MAJOR >= 8
  delete inpector;
  delete context;
#else
  inpector->destroy();
  context->destroy();
#endif

#endif

#if NV_TENSORRT_MAJOR >= 8
  if (m_NetworkType == "onnx") {
    delete parser;
  }
  delete network;
#else
  if (m_NetworkType == "onnx") {
    parser->destroy();
  }
  config->destroy();
  network->destroy();
#endif

  return engine;
}

NvDsInferStatus
Yolo::parseModel(nvinfer1::INetworkDefinition& network) {
  destroyNetworkUtils();

  std::vector<float> weights = loadWeights(m_WtsFilePath);
  std::cout << "Building YOLO network\n" << std::endl;
  NvDsInferStatus status = buildYoloNetwork(weights, network);

  if (status == NVDSINFER_SUCCESS) {
    std::cout << "Building YOLO network complete" << std::endl;
  }
  else {
    std::cerr << "Building YOLO network failed" << std::endl;
  }

  return status;
}

NvDsInferStatus
Yolo::buildYoloNetwork(std::vector<float>& weights, nvinfer1::INetworkDefinition& network)
{
  int weightPtr = 0;

  uint batchSize = m_ImplicitBatch ? m_BatchSize : -1;

  nvinfer1::ITensor* data = network.addInput(m_InputBlobName.c_str(), nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims{4, {static_cast<int>(batchSize), static_cast<int>(m_InputC), static_cast<int>(m_InputH),
      static_cast<int>(m_InputW)}});
  assert(data != nullptr && data->getDimensions().nbDims > 0);

  nvinfer1::ITensor* previous = data;
  std::vector<nvinfer1::ITensor*> tensorOutputs;

  nvinfer1::ITensor* yoloTensorInputs[m_YoloCount];
  uint yoloCountInputs = 0;

  for (uint i = 0; i < m_ConfigBlocks.size(); ++i) {
    std::string layerIndex = "(" + std::to_string(tensorOutputs.size()) + ")";

    if (m_ConfigBlocks.at(i).at("type") == "net")
        printLayerInfo("", "Layer", "Input Shape", "Output Shape", "WeightPtr");
    else if (m_ConfigBlocks.at(i).at("type") == "conv" || m_ConfigBlocks.at(i).at("type") == "convolutional") {
      int channels = getNumChannels(previous);
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = convolutionalLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, channels, previous,
          &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "conv_" + m_ConfigBlocks.at(i).at("activation");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "deconv" || m_ConfigBlocks.at(i).at("type") == "deconvolutional") {
      int channels = getNumChannels(previous);
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = deconvolutionalLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, channels, previous,
          &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "deconv";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "batchnorm") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = batchnormLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "batchnorm_" + m_ConfigBlocks.at(i).at("activation");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "implicit" || m_ConfigBlocks.at(i).at("type") == "implicit_add" ||
        m_ConfigBlocks.at(i).at("type") == "implicit_mul") {
      previous = implicitLayer(i, m_ConfigBlocks.at(i), weights, m_TrtWeights, weightPtr, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "implicit";
      printLayerInfo(layerIndex, layerName, "-", outputVol, std::to_string(weightPtr));
    }
    else if (m_ConfigBlocks.at(i).at("type") == "shift_channels" ||
        m_ConfigBlocks.at(i).at("type") == "control_channels") {
      assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
      int from = stoi(m_ConfigBlocks.at(i).at("from"));
      if (from > 0) {
        from = from - i + 1;
      }
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = channelsLayer(i, m_ConfigBlocks.at(i), previous, tensorOutputs[i + from - 1], &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = m_ConfigBlocks.at(i).at("type") + ": " + std::to_string(i + from - 1);
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "shortcut") {
      assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
      int from = stoi(m_ConfigBlocks.at(i).at("from"));
      if (from > 0) {
        from = from - i + 1;
      }
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);

      std::string activation = "linear";
      if (m_ConfigBlocks.at(i).find("activation") != m_ConfigBlocks.at(i).end()) {
        activation = m_ConfigBlocks.at(i).at("activation");
      }

      std::string inputVol = dimsToString(previous->getDimensions());
      std::string shortcutVol = dimsToString(tensorOutputs[i + from - 1]->getDimensions());
      previous = shortcutLayer(i, activation, inputVol, shortcutVol, m_ConfigBlocks.at(i), previous,
          tensorOutputs[i + from - 1], &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "shortcut_" + activation + ": " + std::to_string(i + from - 1);
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");

      if (inputVol != shortcutVol) {
        std::cout << inputVol << " +" << shortcutVol << std::endl;
      }
    }
    else if (m_ConfigBlocks.at(i).at("type") == "sam") {
      assert(m_ConfigBlocks.at(i).find("from") != m_ConfigBlocks.at(i).end());
      int from = stoi(m_ConfigBlocks.at(i).at("from"));
      if (from > 0) {
        from = from - i + 1;
      }
      assert((i - 2 >= 0) && (i - 2 < tensorOutputs.size()));
      assert((i + from - 1 >= 0) && (i + from - 1 < tensorOutputs.size()));
      assert(i + from - 1 < i - 2);

      std::string activation = "linear";
      if (m_ConfigBlocks.at(i).find("activation") != m_ConfigBlocks.at(i).end()) {
        activation = m_ConfigBlocks.at(i).at("activation");
      }

      std::string inputVol = dimsToString(previous->getDimensions());
      previous = samLayer(i, activation, m_ConfigBlocks.at(i), previous, tensorOutputs[i + from - 1], &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "sam_" + activation + ": " + std::to_string(i + from - 1);
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "route") {
      std::string layers;
      previous = routeLayer(i, layers, m_ConfigBlocks.at(i), tensorOutputs, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "route: " + layers;
      printLayerInfo(layerIndex, layerName, "-", outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "upsample") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = upsampleLayer(i, m_ConfigBlocks[i], previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = "upsample";
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "max" || m_ConfigBlocks.at(i).at("type") == "maxpool" ||
        m_ConfigBlocks.at(i).at("type") == "avg" || m_ConfigBlocks.at(i).at("type") == "avgpool") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = poolingLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = m_ConfigBlocks.at(i).at("type");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "reorg" || m_ConfigBlocks.at(i).at("type") == "reorg3d") {
      std::string inputVol = dimsToString(previous->getDimensions());
      previous = reorgLayer(i, m_ConfigBlocks.at(i), previous, &network);
      assert(previous != nullptr);
      std::string outputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      std::string layerName = m_ConfigBlocks.at(i).at("type");
      printLayerInfo(layerIndex, layerName, inputVol, outputVol, "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "yolo" || m_ConfigBlocks.at(i).at("type") == "region") {
      std::string blobName = m_ConfigBlocks.at(i).at("type") == "yolo" ? "yolo_" + std::to_string(i) :
          "region_" + std::to_string(i);
      nvinfer1::Dims prevTensorDims = previous->getDimensions();
      TensorInfo& curYoloTensor = m_YoloTensors.at(yoloCountInputs);
      curYoloTensor.blobName = blobName;
      curYoloTensor.gridSizeY = prevTensorDims.d[2];
      curYoloTensor.gridSizeX = prevTensorDims.d[3];
      std::string inputVol = dimsToString(previous->getDimensions());
      tensorOutputs.push_back(previous);
      yoloTensorInputs[yoloCountInputs] = previous;
      ++yoloCountInputs;
      std::string layerName = m_ConfigBlocks.at(i).at("type");
      printLayerInfo(layerIndex, layerName, inputVol, "-", "-");
    }
    else if (m_ConfigBlocks.at(i).at("type") == "dropout") {
      // pass
    }
    else {
      std::cerr << "\nUnsupported layer type --> \"" << m_ConfigBlocks.at(i).at("type") << "\"" << std::endl;
      assert(0);
    }
  }

  if ((int) weights.size() != weightPtr) {
    std::cerr << "\nNumber of unused weights left: " << weights.size() - weightPtr << std::endl;
    assert(0);
  }

  if (m_YoloCount == yoloCountInputs) {
    uint64_t outputSize = 0;
    for (uint j = 0; j < yoloCountInputs; ++j) {
      TensorInfo& curYoloTensor = m_YoloTensors.at(j);
      outputSize += curYoloTensor.numBBoxes * curYoloTensor.gridSizeY * curYoloTensor.gridSizeX;
    }

    nvinfer1::IPluginV2DynamicExt* yoloPlugin = new YoloLayer(m_InputW, m_InputH, m_NumClasses, m_NewCoords,
        m_YoloTensors, outputSize);
    assert(yoloPlugin != nullptr);
    nvinfer1::IPluginV2Layer* yolo = network.addPluginV2(yoloTensorInputs, m_YoloCount, *yoloPlugin);
    assert(yolo != nullptr);
    std::string yoloLayerName = m_WtsFilePath;
    yolo->setName(yoloLayerName.c_str());

    std::string outputlayerName;
    nvinfer1::ITensor* detection_output = yolo->getOutput(0);
    outputlayerName = "output";
    detection_output->setName(outputlayerName.c_str());
    network.markOutput(*detection_output);
  }
  else {
    std::cerr << "\nError in yolo cfg file" << std::endl;
    assert(0);
  }

  std::cout << "\nOutput YOLO blob names: " << std::endl;
  for (auto& tensor : m_YoloTensors) {
    std::cout << tensor.blobName << std::endl;
  }

  int nbLayers = network.getNbLayers();
  std::cout << "\nTotal number of YOLO layers: " << nbLayers << "\n" << std::endl;

  return NVDSINFER_SUCCESS;
}

std::vector<std::map<std::string, std::string>>
Yolo::parseConfigFile(const std::string cfgFilePath)
{
  assert(fileExists(cfgFilePath));
  std::ifstream file(cfgFilePath);
  assert(file.good());
  std::string line;
  std::vector<std::map<std::string, std::string>> blocks;
  std::map<std::string, std::string> block;

  while (getline(file, line)) {
    if (line.size() == 0 || line.front() == ' ' || line.front() == '#') {
      continue;
    }

    line = trim(line);
    if (line.front() == '[') {
      if (block.size() > 0) {
        blocks.push_back(block);
        block.clear();
      }
      std::string key = "type";
      std::string value = trim(line.substr(1, line.size() - 2));
      block.insert(std::pair<std::string, std::string>(key, value));
    }
    else {
      int cpos = line.find('=');
      std::string key = trim(line.substr(0, cpos));
      std::string value = trim(line.substr(cpos + 1));
      block.insert(std::pair<std::string, std::string>(key, value));
    }
  }

  blocks.push_back(block);
  return blocks;
}

void
Yolo::parseConfigBlocks()
{
  for (auto block : m_ConfigBlocks) {
    if (block.at("type") == "net") {
      assert((block.find("channels") != block.end()) && "Missing 'channels' param in network cfg");
      assert((block.find("height") != block.end()) && "Missing 'height' param in network cfg");
      assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");

      m_InputC = std::stoul(block.at("channels"));
      m_InputH = std::stoul(block.at("height"));
      m_InputW = std::stoul(block.at("width"));

      m_InputSize = m_InputC * m_InputH * m_InputW;

      if (block.find("letter_box") != block.end()) {
        m_LetterBox = std::stoul(block.at("letter_box"));
      }
    }
    else if ((block.at("type") == "region") || (block.at("type") == "yolo")) {
      assert((block.find("num") != block.end()) &&
          std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
      assert((block.find("classes") != block.end()) &&
          std::string("Missing 'classes' param in " + block.at("type") + " layer").c_str());
      assert((block.find("anchors") != block.end()) &&
          std::string("Missing 'anchors' param in " + block.at("type") + " layer").c_str());

      ++m_YoloCount;

      m_NumClasses = std::stoul(block.at("classes"));

      if (block.find("new_coords") != block.end()) {
        m_NewCoords = std::stoul(block.at("new_coords"));
      }

      TensorInfo outputTensor;

      std::string anchorString = block.at("anchors");
      while (!anchorString.empty()) {
        int npos = anchorString.find_first_of(',');
        if (npos != -1) {
          float anchor = std::stof(trim(anchorString.substr(0, npos)));
          outputTensor.anchors.push_back(anchor);
          anchorString.erase(0, npos + 1);
        }
        else {
          float anchor = std::stof(trim(anchorString));
          outputTensor.anchors.push_back(anchor);
          break;
        }
      }

      if (block.find("mask") != block.end()) {
        std::string maskString = block.at("mask");
        while (!maskString.empty()) {
          int npos = maskString.find_first_of(',');
          if (npos != -1) {
            int mask = std::stoul(trim(maskString.substr(0, npos)));
            outputTensor.mask.push_back(mask);
            maskString.erase(0, npos + 1);
          }
          else {
            int mask = std::stoul(trim(maskString));
            outputTensor.mask.push_back(mask);
            break;
          }
        }
      }

      if (block.find("scale_x_y") != block.end()) {
        outputTensor.scaleXY = std::stof(block.at("scale_x_y"));
      }
      else {
        outputTensor.scaleXY = 1.0;
      }

      outputTensor.numBBoxes = outputTensor.mask.size() > 0 ? outputTensor.mask.size() :
          std::stoul(trim(block.at("num")));
      
      m_YoloTensors.push_back(outputTensor);
    }
  }
}

void
Yolo::destroyNetworkUtils()
{
  for (uint i = 0; i < m_TrtWeights.size(); ++i) {
    if (m_TrtWeights[i].count > 0) {
      free(const_cast<void*>(m_TrtWeights[i].values));
    }
  }
  m_TrtWeights.clear();
}
