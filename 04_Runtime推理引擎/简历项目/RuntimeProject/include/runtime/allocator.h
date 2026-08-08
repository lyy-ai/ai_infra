#pragma once

#include <algorithm>
#include <cstdlib>
#include <unordered_map>
#include <vector>

#include "runtime/common.h"

namespace runtime {

class ArenaAllocator {
 public:
  explicit ArenaAllocator(size_t capacity) : buffer_(capacity) {}

  void* Alloc(size_t bytes, size_t alignment = 256) {
    size_t current = offset_;
    size_t aligned = (current + alignment - 1) / alignment * alignment;
    if (aligned + bytes > buffer_.size()) {
      return nullptr;
    }
    offset_ = aligned + bytes;
    return buffer_.data() + aligned;
  }

  void Reset() { offset_ = 0; }
  size_t used() const { return offset_; }
  size_t capacity() const { return buffer_.size(); }

 private:
  std::vector<char> buffer_;
  size_t offset_ = 0;
};

class SizeClassPool {
 public:
  ~SizeClassPool() {
    for (auto& kv : free_) {
      for (void* p : kv.second) {
        std::free(p);
      }
    }
  }

  void* Alloc(size_t bytes) {
    size_t cls = RoundUp(bytes);
    auto& list = free_[cls];
    if (!list.empty()) {
      void* p = list.back();
      list.pop_back();
      return p;
    }
    ++malloc_calls_;
    return std::malloc(cls);
  }

  void Free(void* p, size_t bytes) {
    if (!p) return;
    size_t cls = RoundUp(bytes);
    free_[cls].push_back(p);
  }

  size_t malloc_calls() const { return malloc_calls_; }
  size_t cached_blocks() const {
    size_t n = 0;
    for (const auto& kv : free_) n += kv.second.size();
    return n;
  }

  static size_t RoundUp(size_t bytes) {
    size_t cls = 64;
    while (cls < bytes) cls <<= 1;
    return cls;
  }

 private:
  std::unordered_map<size_t, std::vector<void*>> free_;
  size_t malloc_calls_ = 0;
};

}  // namespace runtime
