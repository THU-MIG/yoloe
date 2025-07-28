/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "deconvolutional_layer.h"

#include <cassert>
#include <math.h>

nvinfer1::ITensor*
deconvolutionalLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, int& inputChannels, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network, std::string layerName)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "deconv" || block.at("type") == "deconvolutional");
  assert(block.find("filters") != block.end());
  assert(block.find("pad") != block.end());
  assert(block.find("size") != block.end());
  assert(block.find("stride") != block.end());

  int filters = std::stoi(block.at("filters"));
  int padding = std::stoi(block.at("pad"));
  int kernelSize = std::stoi(block.at("size"));
  int stride = std::stoi(block.at("stride"));
  std::string activation = block.at("activation");
  int bias = filters;

  int batchNormalize = 0;
  float eps = 1.0e-5;
  if (block.find("batch_normalize") != block.end()) {
    bias = 0;
    batchNormalize = (block.at("batch_normalize") == "1");
    if (block.find("eps") != block.end()) {
      eps = std::stof(block.at("eps"));
    }
  }

  if (block.find("bias") != block.end()) {
    bias = std::stoi(block.at("bias"));
    if (bias == 1) {
      bias = filters;
    }
  }

  int groups = 1;
  if (block.find("groups") != block.end()) {
    groups = std::stoi(block.at("groups"));
  }

  int pad;
  if (padding) {
    pad = (kernelSize - 1) / 2;
  }
  else {
    pad = 0;
  }

  int size = filters * inputChannels * kernelSize * kernelSize / groups;
  std::vector<float> bnBiases;
  std::vector<float> bnWeights;
  std::vector<float> bnRunningMean;
  std::vector<float> bnRunningVar;
  nvinfer1::Weights convWt {nvinfer1::DataType::kFLOAT, nullptr, size};
  nvinfer1::Weights convBias {nvinfer1::DataType::kFLOAT, nullptr, bias};

  if (batchNormalize == 0) {
    float* val;
    if (bias != 0) {
      val = new float[filters];
      for (int i = 0; i < filters; ++i) {
          val[i] = weights[weightPtr];
          ++weightPtr;
      }
      convBias.values = val;
      trtWeights.push_back(convBias);
    }
    val = new float[size];
    for (int i = 0; i < size; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
  }
  else {
    for (int i = 0; i < filters; ++i) {
      bnBiases.push_back(weights[weightPtr]);
      ++weightPtr;
    }
    for (int i = 0; i < filters; ++i) {
      bnWeights.push_back(weights[weightPtr]);
      ++weightPtr;
    }
    for (int i = 0; i < filters; ++i) {
      bnRunningMean.push_back(weights[weightPtr]);
      ++weightPtr;
    }
    for (int i = 0; i < filters; ++i) {
      bnRunningVar.push_back(sqrt(weights[weightPtr] + eps));
      ++weightPtr;
    }
    float* val;
    if (bias != 0) {
      val = new float[filters];
      for (int i = 0; i < filters; ++i) {
        val[i] = weights[weightPtr];
        ++weightPtr;
      }
      convBias.values = val;
    }
    val = new float[size];
    for (int i = 0; i < size; ++i) {
      val[i] = weights[weightPtr];
      ++weightPtr;
    }
    convWt.values = val;
    trtWeights.push_back(convWt);
    if (bias != 0) {
      trtWeights.push_back(convBias);
    }
  }

  nvinfer1::IDeconvolutionLayer* conv = network->addDeconvolutionNd(*input, filters,
      nvinfer1::Dims{2, {kernelSize, kernelSize}}, convWt, convBias);
  assert(conv != nullptr);
  std::string convLayerName = "deconv_" + layerName + std::to_string(layerIdx);
  conv->setName(convLayerName.c_str());
  conv->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
  conv->setPaddingNd(nvinfer1::Dims{2, {pad, pad}});

  if (block.find("groups") != block.end()) {
    conv->setNbGroups(groups);
  }

  output = conv->getOutput(0);

  if (batchNormalize == 1) {
    size = filters;
    nvinfer1::Weights shift {nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights scale {nvinfer1::DataType::kFLOAT, nullptr, size};
    nvinfer1::Weights power {nvinfer1::DataType::kFLOAT, nullptr, size};

    float* shiftWt = new float[size];
    for (int i = 0; i < size; ++i) {
      shiftWt[i] = bnBiases.at(i) - ((bnRunningMean.at(i) * bnWeights.at(i)) / bnRunningVar.at(i));
    }
    shift.values = shiftWt;

    float* scaleWt = new float[size];
    for (int i = 0; i < size; ++i) {
      scaleWt[i] = bnWeights.at(i) / bnRunningVar[i];
    }
    scale.values = scaleWt;

    float* powerWt = new float[size];
    for (int i = 0; i < size; ++i) {
      powerWt[i] = 1.0;
    }
    power.values = powerWt;

    trtWeights.push_back(shift);
    trtWeights.push_back(scale);
    trtWeights.push_back(power);

    nvinfer1::IScaleLayer* batchnorm = network->addScale(*output, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
    assert(batchnorm != nullptr);
    std::string batchnormLayerName = "batchnorm_" + layerName + std::to_string(layerIdx);
    batchnorm->setName(batchnormLayerName.c_str());
    output = batchnorm->getOutput(0);
  }

  output = activationLayer(layerIdx, activation, output, network, layerName);
  assert(output != nullptr);

  return output;
}
