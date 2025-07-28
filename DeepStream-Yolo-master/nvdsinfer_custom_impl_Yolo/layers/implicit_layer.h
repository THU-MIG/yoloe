/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef __IMPLICIT_LAYER_H__
#define __IMPLICIT_LAYER_H__

#include <map>
#include <vector>
#include <string>

#include "NvInfer.h"

nvinfer1::ITensor* implicitLayer(int layerIdx, std::map<std::string, std::string>& block, std::vector<float>& weights,
    std::vector<nvinfer1::Weights>& trtWeights, int& weightPtr, nvinfer1::INetworkDefinition* network);

#endif
