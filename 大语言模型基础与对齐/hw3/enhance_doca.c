#include <doca_argp.h>
#include <doca_dpa.h>
#include <mpi.h>
#include <time.h> // 新增：用于时间统计

#define MAX_ITERATIONS 100 // 新增：最大迭代次数
#define MAX_EUS 256       // 新增：最大执行单元数

/* 新增参数结构体 */
struct enhancement_params {
    int iterations;
    int eus;
    bool secure_channel;
    char *server_list; // 多服务器支持
};

/* 注册增强参数 */
static void register_enhanced_params(struct enhancement_params *enh_params) {
    struct doca_argp_param *iter_param = doca_argp_param_create();
    doca_argp_param_set_short_name(iter_param, "i");
    doca_argp_param_set_long_name(iter_param, "iterations");
    doca_argp_param_set_description(iter_param, "Number of iterations");
    doca_argp_param_set_type(iter_param, DOCA_ARGP_TYPE_INT);
    doca_argp_register_param(iter_param);

    struct doca_argp_param *eus_param = doca_argp_param_create();
    doca_argp_param_set_short_name(eus_param, "e");
    doca_argp_param_set_long_name(eus_param, "eus");
    doca_argp_param_set_description(eus_param, "Number of DPA Execution Units");
    doca_argp_param_set_type(eus_param, DOCA_ARGP_TYPE_INT);
    doca_argp_register_param(eus_param);

    struct doca_argp_param *secure_param = doca_argp_param_create();
    doca_argp_param_set_short_name(secure_param, "s");
    doca_argp_param_set_long_name(secure_param, "secure");
    doca_argp_param_set_description(secure_param, "Enable secure channel");
    doca_argp_param_set_type(secure_param, DOCA_ARGP_TYPE_BOOLEAN);
    doca_argp_register_param(secure_param);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    // 初始化增强参数
    struct enhancement_params enh_params = {
        .iterations = 1,
        .eus = 16, // 默认16个EU
        .secure_channel = false,
        .server_list = NULL
    };
    
    // 参数解析增强
    doca_argp_init();
    register_all_to_all_params();
    register_enhanced_params(&enh_params); // 注册新参数
    doca_argp_start(argc, argv);
    
    // 新增：多服务器支持初始化
    if (enh_params.server_list) {
        initialize_multi_server(enh_params.server_list); // 需实现
    }

    // 核心执行循环（新增迭代支持）
    double total_time = 0.0;
    for (int iter = 0; iter < enh_params.iterations; iter++) {
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // 资源准备（含安全通道增强）
        prepare_resources(enh_params.secure_channel); 

        // 增强的DPA内核启动
        doca_dpa_kernel_launch_params_t launch_params = {
            .threads_count = enh_params.eus, // 动态EU数量
            .kernel = alltoall_kernel,
            // ...其他参数
        };
        
        if (enh_params.secure_channel) {
            launch_params.kernel = alltoall_secure_kernel; // 安全内核
        }
        
        doca_dpa_kernel_launch_async(&launch_params);
        
        // 等待完成
        doca_dpa_event_wait_until(comp_event, DOCA_DPA_EVENT_STATE_COMPLETED);
        
        clock_gettime(CLOCK_MONOTONIC, &end);
        double iter_time = (end.tv_sec - start.tv_sec) * 1000.0 + 
                          (end.tv_nsec - start.tv_nsec) / 1000000.0;
        total_time += iter_time;
        
        printf("Iteration %d: %.3f ms\n", iter, iter_time);
        
        // 重置资源（保留配置）
        reset_resources(); 
    }

    // 性能报告
    printf("Avg time: %.3f ms | Throughput: %.2f GB/s\n", 
           total_time / enh_params.iterations,
           calculate_throughput(enh_params)); // 需实现

    // 资源清理
    destroy_enhanced_resources();
    MPI_Finalize();
    return 0;
}

/* 新增函数实现 */
void initialize_multi_server(char *server_list) {
    // 1. 解析服务器列表
    // 2. 建立跨服务器MPI通信通道
    // 3. 分布式端点连接
    // [6,7](@ref)
}

doca_error_t create_secure_channel(doca_dpa_ep_t *ep) {
    // 1. 创建加密上下文
    // 2. 交换安全密钥
    // 3. 配置硬件加密引擎
    // [4,9](@ref)
}

double calculate_throughput(struct enhancement_params params) {
    // 根据消息大小、EU数量和迭代时间计算吞吐量
    size_t total_data = (size_t)params.eus * msg_size * params.iterations;
    return (total_data / (total_time / 1000.0)) / 1e9; // GB/s
}