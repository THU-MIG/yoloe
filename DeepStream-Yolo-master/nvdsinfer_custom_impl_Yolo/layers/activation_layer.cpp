/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "activation_layer.h"

#include <cassert>
#include <iostream>

nvinfer1::ITensor*
activationLayer(int layerIdx, std::string activation, nvinfer1::ITensor* input, nvinfer1::INetworkDefinition* network,
    std::string layerName)
{
  nvinfer1::ITensor* output;

  if (activation == "linear") {
    output = input;
  }
  else if (activation == "relu") {
    nvinfer1::IActivationLayer* relu = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
    assert(relu != nullptr);
    std::string reluLayerName = "relu_" + layerName + std::to_string(layerIdx);
    relu->setName(reluLayerName.c_str());
    output = relu->getOutput(0);
  }
  else if (activation == "sigmoid" || activation == "logistic") {
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*input, nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid != nullptr);
    std::string sigmoidLayerName = "sigmoid_" + layerName + std::to_string(layerIdx);
    sigmoid->setName(sigmoidLayerName.c_str());
    output = sigmoid->getOutput(0);
  }
  else if (activation == "tanh") {
    nvinfer1::IActivationLayer* tanh = network->addActivation(*input, nvinfer1::ActivationType::kTANH);
    assert(tanh != nullptr);
    std::string tanhLayerName = "tanh_" + layerName + std::to_string(layerIdx);
    tanh->setName(tanhLayerName.c_str());
    output = tanh->getOutput(0);
  }
  else if (activation == "leaky") {
    nvinfer1::IActivationLayer* leaky = network->addActivation(*input, nvinfer1::ActivationType::kLEAKY_RELU);
    assert(leaky != nullptr);
    std::string leakyLayerName = "leaky_" + layerName + std::to_string(layerIdx);
    leaky->setName(leakyLayerName.c_str());
    leaky->setAlpha(0.1);
    output = leaky->getOutput(0);
  }
  else if (activation == "softplus") {
    nvinfer1::IActivationLayer* softplus = network->addActivation(*input, nvinfer1::ActivationType::kSOFTPLUS);
    assert(softplus != nullptr);
    std::string softplusLayerName = "softplus_" + layerName + std::to_string(layerIdx);
    softplus->setName(softplusLayerName.c_str());
    output = softplus->getOutput(0);
  }
  else if (activation == "mish") {
    nvinfer1::IActivationLayer* softplus = network->addActivation(*input, nvinfer1::ActivationType::kSOFTPLUS);
    assert(softplus != nullptr);
    std::string softplusLayerName = "softplus_" + layerName + std::to_string(layerIdx);
    softplus->setName(softplusLayerName.c_str());
    nvinfer1::IActivationLayer* tanh = network->addActivation(*softplus->getOutput(0), nvinfer1::ActivationType::kTANH);
    assert(tanh != nullptr);
    std::string tanhLayerName = "tanh_" + layerName + std::to_string(layerIdx);
    tanh->setName(tanhLayerName.c_str());
    nvinfer1::IElementWiseLayer* mish = network->addElementWise(*input, *tanh->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD);
    assert(mish != nullptr);
    std::string mishLayerName = "mish_" + layerName + std::to_string(layerIdx);
    mish->setName(mishLayerName.c_str());
    output = mish->getOutput(0);
  }
  else if (activation == "silu" || activation == "swish") {
    nvinfer1::IActivationLayer* sigmoid = network->addActivation(*input, nvinfer1::ActivationType::kSIGMOID);
    assert(sigmoid != nullptr);
    std::string sigmoidLayerName = "sigmoid_" + layerName + std::to_string(layerIdx);
    sigmoid->setName(sigmoidLayerName.c_str());
    nvinfer1::IElementWiseLayer* silu = network->addElementWise(*input, *sigmoid->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD);
    assert(silu != nullptr);
    std::string siluLayerName = "silu_" + layerName + std::to_string(layerIdx);
    silu->setName(siluLayerName.c_str());
    output = silu->getOutput(0);
  }
  else if (activation == "hardsigmoid") {
    nvinfer1::IActivationLayer* hardsigmoid = network->addActivation(*input, nvinfer1::ActivationType::kHARD_SIGMOID);
    assert(hardsigmoid != nullptr);
    std::string hardsigmoidLayerName = "hardsigmoid_" + layerName + std::to_string(layerIdx);
    hardsigmoid->setName(hardsigmoidLayerName.c_str());
    hardsigmoid->setAlpha(1.0 / 6.0);
    hardsigmoid->setBeta(0.5);
    output = hardsigmoid->getOutput(0);
  }
  else if (activation == "hardswish") {
    nvinfer1::IActivationLayer* hardsigmoid = network->addActivation(*input, nvinfer1::ActivationType::kHARD_SIGMOID);
    assert(hardsigmoid != nullptr);
    std::string hardsigmoidLayerName = "hardsigmoid_" + layerName + std::to_string(layerIdx);
    hardsigmoid->setName(hardsigmoidLayerName.c_str());
    hardsigmoid->setAlpha(1.0 / 6.0);
    hardsigmoid->setBeta(0.5);
    nvinfer1::IElementWiseLayer* hardswish = network->addElementWise(*input, *hardsigmoid->getOutput(0),
        nvinfer1::ElementWiseOperation::kPROD);
    assert(hardswish != nullptr);
    std::string hardswishLayerName = "hardswish_" + layerName + std::to_string(layerIdx);
    hardswish->setName(hardswishLayerName.c_str());
    output = hardswish->getOutput(0);
  }
  else {
    std::cerr << "Activation not supported: " << activation << std::endl;
    assert(0);
  }
  return output;
}
