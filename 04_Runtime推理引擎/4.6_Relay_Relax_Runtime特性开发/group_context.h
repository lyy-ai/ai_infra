#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace runtime {

enum class DataType { kFloat16, kFloat32, kInt8, kInt4 };
enum class DeviceType { kCPU, kCUDA, kNPU };

struct TensorInfo {
  std::string name;
  std::vector<int64_t> shape;
  DataType dtype;
  DeviceType device;
  size_t alignment = 256;
};

struct IOInfo {
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

struct GroupQuota {
  size_t max_memory_bytes = 0;
  int max_num_seqs = 0;
  int priority = 1;
};

class GroupContext {
 public:
  explicit GroupContext(std::string group_id) : group_id_(std::move(group_id)) {}

  void SetQuota(const GroupQuota& quota) { quota_ = quota; }
  const GroupQuota& quota() const { return quota_; }
  const std::string& group_id() const { return group_id_; }

  void RegisterIO(const std::string& graph_name, const IOInfo& io) { io_registry_[graph_name] = io; }
  const IOInfo* FindIO(const std::string& graph_name) const {
    auto it = io_registry_.find(graph_name);
    return it == io_registry_.end() ? nullptr : &it->second;
  }

 private:
  std::string group_id_;
  GroupQuota quota_;
  std::unordered_map<std::string, IOInfo> io_registry_;
};

}  // namespace runtime
