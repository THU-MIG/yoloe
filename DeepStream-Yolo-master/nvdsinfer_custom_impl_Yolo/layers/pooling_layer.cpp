/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "pooling_layer.h"

#include <cassert>
#include <iostream>

nvinfer1::ITensor*
poolingLayer(int layerIdx, std::map<std::string, std::string>& block, nvinfer1::ITensor* input,
    nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  assert(block.at("type") == "max" || block.at("type") == "maxpool" || block.at("type") == "avg" ||
      block.at("type") == "avgpool");

  if (block.at("type") == "max" || block.at("type") == "maxpool") {
    assert(block.find("size") != block.end());
    assert(block.find("stride") != block.end());

    int size = std::stoi(block.at("size"));
    int stride = std::stoi(block.at("stride"));

    nvinfer1::IPoolingLayer* maxpool = network->addPoolingNd(*input, nvinfer1::PoolingType::kMAX,
        nvinfer1::Dims{2, {size, size}});
    assert(maxpool != nullptr);
    std::string maxpoolLayerName = "maxpool_" + std::to_string(layerIdx);
    maxpool->setName(maxpoolLayerName.c_str());
    maxpool->setStrideNd(nvinfer1::Dims{2, {stride, stride}});
    maxpool->setPaddingNd(nvinfer1::Dims{2, {(size - 1) / 2, (size - 1) / 2}});
    if (size == 2 && stride == 1) {
      maxpool->setPrePadding(nvinfer1::Dims{2, {0, 0}});
      maxpool->setPostPadding(nvinfer1::Dims{2, {1, 1}});
    }
    output = maxpool->getOutput(0);
  }
  else if (block.at("type") == "avg" || block.at("type") == "avgpool") {
    nvinfer1::Dims inputDims = input->getDimensions();
    nvinfer1::IPoolingLayer* avgpool = network->addPoolingNd(*input, nvinfer1::PoolingType::kAVERAGE,
        nvinfer1::Dims{2, {inputDims.d[1], inputDims.d[2]}});
    assert(avgpool != nullptr);
    std::string avgpoolLayerName = "avgpool_" + std::to_string(layerIdx);
    avgpool->setName(avgpoolLayerName.c_str());
    output = avgpool->getOutput(0);
  }
  else {
    std::cerr << "Pooling not supported: " << block.at("type") << std::endl;
    assert(0);
  }

  return output;
}
