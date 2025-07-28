/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include "slice_layer.h"

#include <cassert>

nvinfer1::ITensor*
sliceLayer(int layerIdx, std::string& name, nvinfer1::ITensor* input, nvinfer1::Dims start, nvinfer1::Dims size,
    nvinfer1::Dims stride, nvinfer1::INetworkDefinition* network)
{
  nvinfer1::ITensor* output;

  nvinfer1::ISliceLayer* slice;

  nvinfer1::Dims inputDims = input->getDimensions();

  if (inputDims.d[0] == -1) {
    slice = network->addSlice(*input, start, nvinfer1::Dims{}, stride);
    assert(slice != nullptr);

    int nbDims = size.nbDims;

    nvinfer1::IShapeLayer* shape = network->addShape(*input);
    assert(shape != nullptr);
    std::string shapeLayerName = "shape_" + name + "_" + std::to_string(layerIdx);
    shape->setName(shapeLayerName.c_str());
    nvinfer1::ITensor* shapeTensor = shape->getOutput(0);
    assert(shapeTensor != nullptr);

#if NV_TENSORRT_MAJOR >= 10
    nvinfer1::ICastLayer* castShape = network->addCast(*shapeTensor, nvinfer1::DataType::kINT32);
    assert(castShape != nullptr);
    std::string castShapeLayerName = "cast_shape_" + name + "_" + std::to_string(layerIdx);
    castShape->setName(castShapeLayerName.c_str());
    nvinfer1::ITensor* castShapeTensor = castShape->getOutput(0);
    assert(castShapeTensor != nullptr);
    shapeTensor = castShapeTensor;
#endif

    nvinfer1::Weights constantWt {nvinfer1::DataType::kINT32, nullptr, nbDims};

    int* val = new int[nbDims];
    for (int i = 0; i < nbDims; ++i) {
      if (inputDims.d[i] == size.d[i]) {
        val[i] = 0;
      }
      else {
        val[i] = inputDims.d[i] - size.d[i];
      }
    }
    constantWt.values = val;

    nvinfer1::IConstantLayer* constant = network->addConstant(nvinfer1::Dims{1, {nbDims}}, constantWt);
    assert(constant != nullptr);
    std::string constantLayerName = "constant_" + name + "_" + std::to_string(layerIdx);
    constant->setName(constantLayerName.c_str());
    nvinfer1::ITensor* constantTensor = constant->getOutput(0);
    assert(constantTensor != nullptr);

    nvinfer1::IElementWiseLayer* divide = network->addElementWise(*shapeTensor, *constantTensor,
        nvinfer1::ElementWiseOperation::kSUB);
    assert(divide != nullptr);
    std::string divideLayerName = "divide_" + name + "_" + std::to_string(layerIdx);
    divide->setName(divideLayerName.c_str());
    nvinfer1::ITensor* divideTensor = divide->getOutput(0);
    assert(divideTensor != nullptr);

    slice->setInput(2, *divideTensor);
  }
  else {
    slice = network->addSlice(*input, start, size, stride);
    assert(slice != nullptr);
  }

  std::string sliceLayerName = name + "_" + std::to_string(layerIdx);
  slice->setName(sliceLayerName.c_str());
  output = slice->getOutput(0);

  return output;
}
