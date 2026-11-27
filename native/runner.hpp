#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
class Runner { public: explicit Runner(const std::string&); ~Runner(); std::string infer(const std::vector<unsigned char>&); private: nvinfer1::IRuntime* runtime_{}; nvinfer1::ICudaEngine* engine_{}; nvinfer1::IExecutionContext* context_{}; cudaStream_t stream_{}; void* input_{}; void* output_{}; void* proto_{}; int width_{}; int height_{}; };
