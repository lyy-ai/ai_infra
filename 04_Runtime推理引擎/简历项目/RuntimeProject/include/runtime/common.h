#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace runtime {

enum class DataType { kFloat16, kFloat32, kInt8 };

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  DataType dtype = DataType::kFloat16;
  size_t alignment = 256;
};

struct Node {
  std::string name;
  double compute_us = 0.0;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
};

struct Request {
  int id = 0;
  int arrival_ms = 0;
  int tokens = 0;
  int start_ms = 0;
  int finish_ms = 0;
};

}  // namespace runtime
