/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#include <stdint.h>

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ void softmaxGPU(const float* input, const int bbindex, const int numGridCells, uint z_id,
    const uint numOutputClasses, float temp, float* output)
{
  int i;
  float sum = 0;
  float largest = -INFINITY;
  for (i = 0; i < numOutputClasses; ++i) {
    int val = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))];
    largest = (val>largest) ? val : largest;
  }
  for (i = 0; i < numOutputClasses; ++i) {
    float e = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] / temp - largest / temp);
    sum += e;
    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] = e;
  }
  for (i = 0; i < numOutputClasses; ++i) {
    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] /= sum;
  }
}

__global__ void gpuRegionLayer(const float* input, float* softmax, float* output, const uint netWidth,
    const uint netHeight, const uint gridSizeX, const uint gridSizeY, const uint numOutputClasses, const uint numBBoxes,
    const uint64_t lastInputSize, const float* anchors)
{
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if (x_id >= gridSizeX || y_id >= gridSizeY || z_id >= numBBoxes) {
    return;
  }

  const int numGridCells = gridSizeX * gridSizeY;
  const int bbindex = y_id * gridSizeX + x_id;

  float xc = (sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)]) + x_id) * netWidth /
      gridSizeX;

  float yc = (sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)]) + y_id) * netHeight /
      gridSizeY;

  float w = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)]) * anchors[z_id * 2] * netWidth /
      gridSizeX;

  float h = __expf(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)]) * anchors[z_id * 2 + 1] *
      netHeight / gridSizeY;

  const float objectness = sigmoidGPU(input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)]);

  softmaxGPU(input, bbindex, numGridCells, z_id, numOutputClasses, 1.0, softmax);

  float maxProb = 0.0f;
  int maxIndex = -1;

  for (uint i = 0; i < numOutputClasses; ++i) {
    float prob = softmax[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))];
    if (prob > maxProb) {
      maxProb = prob;
      maxIndex = i;
    }
  }

  int count = numGridCells * z_id + bbindex + lastInputSize;

  output[count * 6 + 0] = xc - w * 0.5;
  output[count * 6 + 1] = yc - h * 0.5;
  output[count * 6 + 2] = xc + w * 0.5;
  output[count * 6 + 3] = yc + h * 0.5;
  output[count * 6 + 4] = maxProb * objectness;
  output[count * 6 + 5] = (float) maxIndex;
}

cudaError_t cudaRegionLayer(const void* input, void* softmax, void* output, const uint& batchSize,
    const uint64_t& inputSize, const uint64_t& outputSize, const uint64_t& lastInputSize, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, const void* anchors, cudaStream_t stream);

cudaError_t cudaRegionLayer(const void* input, void* softmax, void* output, const uint& batchSize,
    const uint64_t& inputSize, const uint64_t& outputSize, const uint64_t& lastInputSize, const uint& netWidth,
    const uint& netHeight, const uint& gridSizeX, const uint& gridSizeY, const uint& numOutputClasses,
    const uint& numBBoxes, const void* anchors, cudaStream_t stream)
{
  dim3 threads_per_block(16, 16, 4);
  dim3 number_of_blocks((gridSizeX / threads_per_block.x) + 1, (gridSizeY / threads_per_block.y) + 1,
      (numBBoxes / threads_per_block.z) + 1);

  for (unsigned int batch = 0; batch < batchSize; ++batch) {
    gpuRegionLayer<<<number_of_blocks, threads_per_block, 0, stream>>>(
        reinterpret_cast<const float*> (input) + (batch * inputSize),
        reinterpret_cast<float*> (softmax) + (batch * inputSize),
        reinterpret_cast<float*> (output) + (batch * 6 * outputSize),
        netWidth, netHeight, gridSizeX, gridSizeY, numOutputClasses, numBBoxes, lastInputSize,
        reinterpret_cast<const float*> (anchors));
  }
  return cudaGetLastError();
}
