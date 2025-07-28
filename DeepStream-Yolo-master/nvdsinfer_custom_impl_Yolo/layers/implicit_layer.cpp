/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "implicit_layer.h"

#include <cassert>

nvinfer1::ITensor*
implicitLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "implicit" || block.at("type") == "implicit_add" || block.at("type") == "implicit_mul");
  assert(block.find("filters") != block.end());

  int filters = std::stoi(block.at("filters"));

  nvinfer1::Weights convWt {nvinfer1::DataType::kFLOAT, nullptr, filters};

  float* val = new float[filters];
  for (int i = 0; i < filters; ++i) {
    val[i] = weights[weightPtr];
    ++weightPtr;
  }
  convWt.values = val;
  trtWeights.push_back(convWt);

  nvinfer1::IConstantLayer* implicit = network->addConstant(nvinfer1::Dims{4, {1, filters, 1, 1}}, convWt);
  assert(implicit != nullptr);
  std::string implicitLayerName = block.at("type") + "_" + std::to_string(layerIdx);
  implicit->setName(implicitLayerName.c_str());
  output = implicit->getOutput(0);

  return output;
}
