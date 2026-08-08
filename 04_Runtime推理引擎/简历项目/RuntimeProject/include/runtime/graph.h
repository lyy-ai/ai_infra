#pragma once

#include <numeric>
#include <vector>

#include "runtime/common.h"

namespace runtime {

inline std::vector<Node> DemoGraph() {
  return {
      {"qkv", 18.0, {"x"}, {"q", "k", "v"}},
      {"attn", 42.0, {"q", "k", "v"}, {"attn_out"}},
      {"proj", 12.0, {"attn_out"}, {"proj_out"}},
      {"norm", 4.0, {"proj_out"}, {"norm_out"}},
      {"mlp", 36.0, {"norm_out"}, {"y"}},
  };
}

inline double TotalComputeUs(const std::vector<Node>& graph) {
  double total = 0.0;
  for (const auto& node : graph) total += node.compute_us;
  return total;
}

inline double EstimateNoGraphUs(const std::vector<Node>& graph, double launch_overhead_us = 5.0) {
  return TotalComputeUs(graph) + graph.size() * launch_overhead_us + 3.0;
}

inline double EstimateGraphPerIterUs(const std::vector<Node>& graph, int replays, double capture_us = 120.0,
                                     double replay_launch_us = 8.0) {
  if (replays <= 0) return 0.0;
  double total = capture_us + replays * (TotalComputeUs(graph) + replay_launch_us + 3.0);
  return total / replays;
}

}  // namespace runtime
