/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "channels_layer.h"

#include <cassert>

nvinfer1::ITensor*
channelsLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::ITensor* implicitTensor, nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "shift_channels" || block.at("type") == "control_channels");

  if (block.at("type") == "shift_channels") {
    nvinfer1::IElementWiseLayer* shift = network->addElementWise(*input, *implicitTensor,
        nvinfer1::ElementWiseOperation::kSUM);
    assert(shift != nullptr);
    std::string shiftLayerName = "shift_channels_" + std::to_string(layerIdx);
    shift->setName(shiftLayerName.c_str());
    output = shift->getOutput(0);
  }
  else if (block.at("type") == "control_channels") {
    nvinfer1::IElementWiseLayer* control = network->addElementWise(*input, *implicitTensor,
        nvinfer1::ElementWiseOperation::kPROD);
    assert(control != nullptr);
    std::string controlLayerName = "control_channels_" + std::to_string(layerIdx);
    control->setName(controlLayerName.c_str());
    output = control->getOutput(0);
  }

  return output;
}
