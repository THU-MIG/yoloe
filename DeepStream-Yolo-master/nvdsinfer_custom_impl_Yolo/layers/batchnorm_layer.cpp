/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "batchnorm_layer.h"

#include <cassert>
#include <math.h>

nvinfer1::ITensor*
batchnormLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "batchnorm");
  assert(block.find("filters") != block.end());

  int filters = std::stoi(block.at("filters"));
  std::string activation = block.at("activation");

  float eps = 1.0e-5;
  if (block.find("eps") != block.end()) {
    eps = std::stof(block.at("eps"));
  }

  std::vector<float> bnBiases;
  std::vector<float> bnWeights;
  std::vector<float> bnRunningMean;
  std::vector<float> bnRunningVar;

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

  int size = filters;
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

  nvinfer1::IScaleLayer* batchnorm = network->addScale(*input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
  assert(batchnorm != nullptr);
  std::string batchnormLayerName = "batchnorm_" + std::to_string(layerIdx);
  batchnorm->setName(batchnormLayerName.c_str());
  output = batchnorm->getOutput(0);

  output = activationLayer(layerIdx, activation, output, network);
  assert(output != nullptr);

  return output;
}
