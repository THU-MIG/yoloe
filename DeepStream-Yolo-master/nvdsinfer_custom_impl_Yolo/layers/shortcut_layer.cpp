/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "shortcut_layer.h"

#include <cassert>

nvinfer1::ITensor*
shortcutLayer(int layerIdx, std::string activation, std::string inputVol, std::string shortcutVol,
    std::map<std::string, std::string>& block, nvinfer1::ITensor* input, nvinfer1::ITensor* shortcutInput,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "shortcut");

  if (inputVol != shortcutVol) {
    std::string name = "slice";
    nvinfer1::Dims start = {4, {0, 0, 0, 0}};
    nvinfer1::Dims size = input->getDimensions();
    nvinfer1::Dims stride = {4, {1, 1, 1, 1}};

    output = sliceLayer(layerIdx, name, shortcutInput, start, size, stride, network);
    assert(output != nullptr);
  }
  else {
    output = shortcutInput;
  }

  nvinfer1::IElementWiseLayer* shortcut = network->addElementWise(*input, *output,
      nvinfer1::ElementWiseOperation::kSUM);
  assert(shortcut != nullptr);
  std::string shortcutLayerName = "shortcut_" + std::to_string(layerIdx);
  shortcut->setName(shortcutLayerName.c_str());
  output = shortcut->getOutput(0);

  output = activationLayer(layerIdx, activation, output, network);
  assert(output != nullptr);

  return output;
}
