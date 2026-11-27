#include "runner.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
constexpr int candidates = 8400, channels = 37, maskChannels = 32, maskSize = 160;
class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << message << '\n';
  }
} logger;

void check(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
}  // namespace

#include "segmentation.hpp"

Runner::Runner(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(file)), {});
  if (bytes.empty()) throw std::runtime_error("Cannot read engine");
  runtime_ = nvinfer1::createInferRuntime(logger);
  if (!runtime_) throw std::runtime_error("Cannot create TensorRT runtime");
  engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
  if (!engine_) throw std::runtime_error("Cannot deserialize engine");
  context_ = engine_->createExecutionContext();
  if (!context_) throw std::runtime_error("Cannot create execution context");
  const auto shape = engine_->getTensorShape("images");
  const auto boxes = engine_->getTensorShape("output0");
  const auto masks = engine_->getTensorShape("output1");
  if (shape.nbDims != 4 || shape.d[0] != 1 || shape.d[1] != 3 ||
      shape.d[2] != 640 || shape.d[3] != 640 ||
      boxes.nbDims != 3 || boxes.d[0] != 1 || boxes.d[1] != channels || boxes.d[2] != candidates ||
      masks.nbDims != 4 || masks.d[0] != 1 || masks.d[1] != maskChannels ||
      masks.d[2] != maskSize || masks.d[3] != maskSize)
    throw std::runtime_error("Unexpected engine tensor shapes");
  for (const char* name : {"images", "output0", "output1"})
    if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT)
      throw std::runtime_error("Expected float32 engine I/O");
  width_ = shape.d[3];
  height_ = shape.d[2];
  check(cudaStreamCreate(&stream_));
  check(cudaMalloc(&input_, 3 * width_ * height_ * sizeof(float)));
  check(cudaMalloc(&output_, channels * candidates * sizeof(float)));
  check(cudaMalloc(&proto_, maskChannels * maskSize * maskSize * sizeof(float)));
  if (!context_->setTensorAddress("images", input_) ||
      !context_->setTensorAddress("output0", output_) ||
      !context_->setTensorAddress("output1", proto_))
    throw std::runtime_error("Cannot bind engine tensors");
}

Runner::~Runner() {
  delete context_;
  if (stream_) cudaStreamDestroy(stream_);
  cudaFree(input_);
  cudaFree(output_);
  cudaFree(proto_);
  delete engine_;
  delete runtime_;
}

std::string Runner::infer(const std::vector<unsigned char>& bytes) {
  cv::Mat original = cv::imdecode(bytes, cv::IMREAD_COLOR);
  if (original.empty()) return R"({"predictions":[],"error":"invalid_image"})";
  const double gain = std::min(double(width_) / original.cols, double(height_) / original.rows);
  const int resizedWidth = std::max(1, int(std::nearbyint(original.cols * gain)));
  const int resizedHeight = std::max(1, int(std::nearbyint(original.rows * gain)));
  const cv::Rect content((width_ - resizedWidth) / 2, (height_ - resizedHeight) / 2,
                         resizedWidth, resizedHeight);
  cv::Mat input(height_, width_, CV_8UC3, cv::Scalar(114, 114, 114));
  cv::resize(original, input(content), content.size(), 0, 0, cv::INTER_LINEAR);
  cv::Mat blob = cv::dnn::blobFromImage(input, 1.0 / 255, {}, {}, true);
  std::vector<float> detections(channels * candidates), prototypes(maskChannels * maskSize * maskSize);
  try {
    check(cudaMemcpyAsync(input_, blob.ptr<float>(), blob.total() * sizeof(float),
                         cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("enqueueV3 failed");
    check(cudaMemcpyAsync(detections.data(), output_, detections.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream_));
    check(cudaMemcpyAsync(prototypes.data(), proto_, prototypes.size() * sizeof(float),
                         cudaMemcpyDeviceToHost, stream_));
    check(cudaStreamSynchronize(stream_));
  } catch (const std::exception& error) {
    cudaStreamSynchronize(stream_);
    std::cerr << error.what() << '\n';
    return R"({"predictions":[],"error":"inference_failed"})";
  }
  return formatPredictions(detections, prototypes, content, input.size(), original.size(),
                           candidates, maskChannels, maskSize);
}
