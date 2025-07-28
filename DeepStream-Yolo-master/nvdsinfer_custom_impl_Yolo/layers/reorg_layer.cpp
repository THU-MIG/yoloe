/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "reorg_layer.h"

#include <vector>
#include <cassert>

nvinfer1::ITensor*
reorgLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "reorg" || block.at("type") == "reorg3d");

  int stride = 1;
  if(block.find("stride") != block.end()) {
    stride = std::stoi(block.at("stride"));
  }

  nvinfer1::Dims inputDims = input->getDimensions();

  if (block.at("type") == "reorg3d") {
    std::string name1 = "slice1";
    std::string name2 = "slice2";
    std::string name3 = "slice3";
    std::string name4 = "slice4";
    nvinfer1::Dims start1 = {4, {0, 0, 0, 0}};
    nvinfer1::Dims start2 = {4, {0, 0, 0, 1}};
    nvinfer1::Dims start3 = {4, {0, 0, 1, 0}};
    nvinfer1::Dims start4 = {4, {0, 0, 1, 1}};
    nvinfer1::Dims sizeAll = {4, {inputDims.d[0], inputDims.d[1], inputDims.d[2] / stride, inputDims.d[3] / stride}};
    nvinfer1::Dims strideAll = {4, {1, 1, stride, stride}};

    nvinfer1::ITensor* slice1 = sliceLayer(layerIdx, name1, input, start1, sizeAll, strideAll, network);
    assert(slice1 != nullptr);

    nvinfer1::ITensor* slice2 = sliceLayer(layerIdx, name2, input, start2, sizeAll, strideAll, network);
    assert(slice2 != nullptr);

    nvinfer1::ITensor* slice3 = sliceLayer(layerIdx, name3, input, start3, sizeAll, strideAll, network);
    assert(slice3 != nullptr);

    nvinfer1::ITensor* slice4 = sliceLayer(layerIdx, name4, input, start4, sizeAll, strideAll, network);
    assert(slice4 != nullptr);

    std::vector<nvinfer1::ITensor*> concatInputs;
    concatInputs.push_back(slice1);
    concatInputs.push_back(slice2);
    concatInputs.push_back(slice3);
    concatInputs.push_back(slice4);

    nvinfer1::IConcatenationLayer* concat = network->addConcatenation(concatInputs.data(), concatInputs.size());
    assert(concat != nullptr);
    std::string concatLayerName = "concat_" + std::to_string(layerIdx);
    concat->setName(concatLayerName.c_str());
    concat->setAxis(0);
    output = concat->getOutput(0);
  }
  else {
    nvinfer1::IShuffleLayer* shuffle1 = network->addShuffle(*input);
    assert(shuffle1 != nullptr);
    std::string shuffle1LayerName = "shuffle1_" + std::to_string(layerIdx);
    shuffle1->setName(shuffle1LayerName.c_str());
    nvinfer1::Dims reshapeDims1{6, {inputDims.d[0], inputDims.d[1] / (stride * stride), inputDims.d[2], stride,
        inputDims.d[3], stride}};
    shuffle1->setReshapeDimensions(reshapeDims1);
    nvinfer1::Permutation permutation1{{0, 1, 2, 4, 3, 5}};
    shuffle1->setSecondTranspose(permutation1);
    output = shuffle1->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle2 = network->addShuffle(*output);
    assert(shuffle2 != nullptr);
    std::string shuffle2LayerName = "shuffle2_" + std::to_string(layerIdx);
    shuffle2->setName(shuffle2LayerName.c_str());
    nvinfer1::Dims reshapeDims2{4, {inputDims.d[0], inputDims.d[1] / (stride * stride), inputDims.d[2] * inputDims.d[3],
        stride * stride}};
    shuffle2->setReshapeDimensions(reshapeDims2);
    nvinfer1::Permutation permutation2{{0, 1, 3, 2}};
    shuffle2->setSecondTranspose(permutation2);
    output = shuffle2->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle3 = network->addShuffle(*output);
    assert(shuffle3 != nullptr);
    std::string shuffle3LayerName = "shuffle3_" + std::to_string(layerIdx);
    shuffle3->setName(shuffle3LayerName.c_str());
    nvinfer1::Dims reshapeDims3{4, {inputDims.d[0], inputDims.d[1] / (stride * stride), stride * stride,
        inputDims.d[2] * inputDims.d[3]}};
    shuffle3->setReshapeDimensions(reshapeDims3);
    nvinfer1::Permutation permutation3{{0, 2, 1, 3}};
    shuffle3->setSecondTranspose(permutation3);
    output = shuffle3->getOutput(0);

    nvinfer1::IShuffleLayer* shuffle4 = network->addShuffle(*output);
    assert(shuffle4 != nullptr);
    std::string shuffle4LayerName = "shuffle4_" + std::to_string(layerIdx);
    shuffle4->setName(shuffle4LayerName.c_str());
    nvinfer1::Dims reshapeDims4{4, {inputDims.d[0], inputDims.d[1] * stride * stride, inputDims.d[2] / stride,
        inputDims.d[3] / stride}};
    shuffle4->setReshapeDimensions(reshapeDims4);
    output = shuffle4->getOutput(0);
  }

  return output;
}
