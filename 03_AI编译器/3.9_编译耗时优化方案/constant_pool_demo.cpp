// 4.9 编译耗时优化方案：GraphConstantPool 抽离常量加载示例
//
// 编译运行：
//   cd /data/ai_infra/AI编译器
//   g++ -O2 -std=c++17 4.9_编译耗时优化方案/constant_pool_demo.cpp -o /tmp/constant_pool_demo
//   /tmp/constant_pool_demo
#include <map>
#include <list>
#include <vector>
#include <cstring>
#include <iostream>

// 占位：实际项目中由 TVM CodeGen 生成这些全局常量
float constant_bev_grid[1024] = {0};
float constant_cam_intrinsic[1024] = {0};
float constant_depth_lut[1024] = {0};
float backbone_weights[1024] = {0};
float transform_weights[1024] = {0};
float bev_fusion_weights[1024] = {0};
float head_weights[1024] = {0};
float* constant_backbone_weights_ptr = backbone_weights;
float* constant_transform_weights_ptr = transform_weights;
float* constant_bev_fusion_weights_ptr = bev_fusion_weights;
float* constant_head_weights_ptr = head_weights;

// 常量统一缓存结构体
struct GraphConstantPool {
    std::map<std::string, const void*> constant_map;
    std::list<const float*> weight_ptr_list;
};

GraphConstantPool load_graph_constants() {
    GraphConstantPool pool;

    pool.constant_map["constant_bev_grid"] = constant_bev_grid;
    pool.constant_map["constant_cam_intrinsic"] = constant_cam_intrinsic;
    pool.constant_map["constant_depth_lut"] = constant_depth_lut;

    std::vector<const float*> all_weight_ptrs = {
        constant_backbone_weights_ptr,
        constant_transform_weights_ptr,
        constant_bev_fusion_weights_ptr,
        constant_head_weights_ptr,
    };
    for (auto ptr : all_weight_ptrs) {
        if (ptr != nullptr) {
            pool.weight_ptr_list.push_back(ptr);
        }
    }
    return pool;
}

int main() {
    GraphConstantPool const_pool = load_graph_constants();

    auto grid_ptr = const_pool.constant_map.at("constant_bev_grid");
    auto cam_ptr = const_pool.constant_map.at("constant_cam_intrinsic");

    for (auto w_ptr : const_pool.weight_ptr_list) {
        (void)w_ptr;
    }

    std::cout << "GraphConstantPool loaded, "
              << const_pool.constant_map.size() << " constants, "
              << const_pool.weight_ptr_list.size() << " weight pointers"
              << std::endl;
    return 0;
}
