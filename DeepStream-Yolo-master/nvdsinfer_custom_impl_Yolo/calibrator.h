/*
 * Created by Marcos Luciano
 * https://www.github.com/marcoslucianops
 */

#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <vector>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "opencv2/opencv.hpp"

#define CUDA_CHECK(status) {                                                                                           \
  if (status != 0) {                                                                                                   \
    std::cout << "CUDA failure: " << cudaGetErrorString(status) << " in file " << __FILE__  << " at line "  <<         \
        __LINE__ << std::endl;                                                                                         \
    abort();                                                                                                           \
  }                                                                                                                    \
}

class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
  public:
    Int8EntropyCalibrator2(const int& batchSize, const int& channels, const int& height, const int& width,
        const float& scaleFactor, const float* offsets, const int& inputFormat, const std::string& imgPath,
        const std::string& calibTablePath);

    virtual ~Int8EntropyCalibrator2();

    int getBatchSize() const noexcept override;

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(std::size_t& length) noexcept override;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

  private:
    int batchSize;
    int inputC;
    int inputH;
    int inputW;
    int letterBox;
    float scaleFactor;
    const float* offsets;
    int inputFormat;
    std::string calibTablePath;
    size_t imageIndex;
    size_t inputCount;
    std::vector<std::string> imgPaths;
    float* batchData {nullptr};
    void* deviceInput {nullptr};
    bool readCache;
    std::vector<char> calibrationCache;
};

std::vector<float> prepareImage(cv::Mat& img, int inputC, int inputH, int inputW, float scaleFactor,
    const float* offsets, int inputFormat);

#endif //CALIBRATOR_H
