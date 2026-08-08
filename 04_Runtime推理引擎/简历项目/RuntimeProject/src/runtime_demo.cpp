#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "runtime/allocator.h"
#include "runtime/graph.h"

namespace {

struct BatchResult {
  int batches = 0;
  int total_tokens = 0;
  double total_time_ms = 0.0;
  double avg_wait_ms = 0.0;
  double throughput = 0.0;
};

BatchResult RunDynamicBatch(std::vector<runtime::Request> requests, int max_batch, int max_tokens) {
  std::sort(requests.begin(), requests.end(), [](const auto& a, const auto& b) {
    return a.arrival_ms < b.arrival_ms;
  });

  int time_ms = 0;
  size_t idx = 0;
  BatchResult result;
  std::vector<int> waits;

  while (idx < requests.size()) {
    if (requests[idx].arrival_ms > time_ms) {
      time_ms = requests[idx].arrival_ms;
    }
    std::vector<runtime::Request*> batch;
    int tokens = 0;
    while (idx < requests.size() && requests[idx].arrival_ms <= time_ms &&
           static_cast<int>(batch.size()) < max_batch && tokens + requests[idx].tokens <= max_tokens) {
      batch.push_back(&requests[idx]);
      tokens += requests[idx].tokens;
      ++idx;
    }
    if (batch.empty()) {
      ++time_ms;
      continue;
    }

    double cost_ms = 8.0 + tokens * 0.02 / 1000.0;
    for (auto* req : batch) {
      req->start_ms = time_ms;
      req->finish_ms = time_ms + static_cast<int>(cost_ms);
      waits.push_back(req->start_ms - req->arrival_ms);
      result.total_tokens += req->tokens;
    }
    time_ms += static_cast<int>(cost_ms);
    ++result.batches;
  }

  result.total_time_ms = time_ms;
  result.avg_wait_ms = waits.empty() ? 0.0 : std::accumulate(waits.begin(), waits.end(), 0.0) / waits.size();
  result.throughput = result.total_time_ms > 0 ? result.total_tokens / (result.total_time_ms / 1000.0) : 0.0;
  return result;
}

}  // namespace

int main() {
  using namespace runtime;

  ArenaAllocator arena(16 * 1024 * 1024);
  void* p1 = arena.Alloc(1024 * 1024);
  void* p2 = arena.Alloc(3 * 1024 * 1024);
  std::cout << "arena used: " << arena.used() / 1024 << " KiB, p1=" << (p1 != nullptr) << ", p2=" << (p2 != nullptr) << "\n";

  SizeClassPool pool;
  void* a = pool.Alloc(100);
  void* b = pool.Alloc(100);
  pool.Free(a, 100);
  void* c = pool.Alloc(100);
  std::cout << "pool malloc_calls: " << pool.malloc_calls() << ", reuse c==a: " << (c == a) << ", b=" << (b != nullptr) << "\n";

  auto graph = DemoGraph();
  double no_graph = EstimateNoGraphUs(graph);
  std::cout << "graph nodes: " << graph.size() << ", no_graph: " << no_graph << " us\n";
  for (int replays : {1, 4, 16, 64}) {
    double graph_per_iter = EstimateGraphPerIterUs(graph, replays);
    std::cout << "cuda_graph replays=" << std::setw(3) << replays << " per_iter=" << std::fixed << std::setprecision(2)
              << graph_per_iter << " us, speedup=" << (no_graph / graph_per_iter) << "x\n";
  }

  std::vector<Request> requests;
  for (int i = 0; i < 24; ++i) {
    requests.push_back(Request{i, (i / 3) * 2, (i % 5 + 1) * 256});
  }
  auto result = RunDynamicBatch(requests, /*max_batch=*/8, /*max_tokens=*/4096);
  std::cout << "dynamic_batch: batches=" << result.batches << ", total_tokens=" << result.total_tokens
            << ", avg_wait=" << result.avg_wait_ms << " ms, throughput=" << result.throughput << " tok/s\n";

  return 0;
}
