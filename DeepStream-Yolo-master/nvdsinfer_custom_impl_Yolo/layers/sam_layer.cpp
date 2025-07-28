/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "sam_layer.h"

#include <cassert>

nvinfer1::ITensor*
samLayer(int layerIdx, std::string activation, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::ITensor* samInput, nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "sam");

  nvinfer1::IElementWiseLayer* sam = network->addElementWise(*input, *samInput, nvinfer1::ElementWiseOperation::kPROD);
  assert(sam != nullptr);
  std::string samLayerName = "sam_" + std::to_string(layerIdx);
  sam->setName(samLayerName.c_str());
  output = sam->getOutput(0);

  output = activationLayer(layerIdx, activation, output, network);
  assert(output != nullptr);

  return output;
}
