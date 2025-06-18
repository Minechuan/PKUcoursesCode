### 运行两条指令
针对 dpa all to all:
meson /tmp/try/dpa_a2a -Denable_all_applications=false -Denable_dpa_all_to_all=true
ninja -C /tmp/try/dpa_a2a
mpirun -np 4 /tmp/try/dpa_a2a/dpa_all_to_all/doca_dpa_all_to_all -m 32 -d "mlx5_0"

修改代码添加了时间统计

参数：
1. 固定进程数目为 4 ，改变 message size - the size of the sendbuf and recvbuf

16 1.521223
32 1.525354
...
4096 1.521331


2. 改变线程数量：大小设置为 64 时：
   2 0.919402
   4 1.522374
   8 2.813593
   16 5.647253
   
   
3. 模拟实际应用场景，针对张量并行，模型的各部分参数需要共享梯度，我们假设只有 16 张卡，即 16 个进程，但是有
 16 4096     5.741500

