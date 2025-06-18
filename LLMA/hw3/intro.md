# 说明文档和注意事项

/tmp/build/ 文件夹是在运行
```
cd /opt/mellanox/doca/applications/
meson /tmp/build -Denable_all_applications=false -Denable_dpa_all_to_all=true
ninja -C /tmp/build
```
生成的内容，通过命令行动态设置：
禁用 "构建所有应用程序" 的默认选项；启用 "DPA All-to-All" 应用程序。
然后读取 Meson 生成的 build.ninja 文件，根据目标依赖关系编译：

``dpa_all_to_all/doca_dpa_all_to_all`` 是可执行的文件，之后可以直接运行

以下是需要注意的内容，由于原 doca 中文件不能修改，于是将 doca 文件复制到 /tmp/build/doca_copy/doca 中，可以修改；虽然可能有文件由于权限问题没有复制，但是不影响编译功能。


在我们的作业实现中，主要需要使用到以下两个文件夹中的代码：
```
/tmp/build/doca_copy/doca/applications/dpa_all_to_all/host
/tmp/build/doca_copy/doca/applications/dpa_all_to_all/device
```

```
doca/
└── applications/
    └── dpa_all_to_all/
```

实际上的输出在这里 .core
```
/tmp/build/doca_copy/doca/applications/dpa_all_to_all/host/dpa_all_to_all.c
```

运行逻辑,暂时保存在 /tmp/build/try/name 里面，name 需要自己设置；

```bash
meson /tmp/build/dpa_a2a -Denable_all_applications=false -Denable_dpa_all_to_all=true

ninja -C /tmp/build/try/name

mpirun -np 4 /tmp/build/try/name/dpa_all_to_all/doca_dpa_all_to_all -m 32 -d "mlx5_0"
```

注意：meson-logs/ 是配置内容

---



---

### 修改更改的权限

（我已经修改了）

sudo apt-get update && sudo apt-get install acl   # 安装 ACL（如未安装）
sudo setfacl -R -m u:$USER:rwx /opt/mellanox/doca/applications/dpa_all_to_all



---

可能的方向及分工

## Explore other MPI collective operations

DOCA DPA 示例已经演示了 all-to-all 的基本 kernel 和 host 逻辑，只需要把它改成 all-reduce、broadcast 或 reduce_scatter 等，复用大部分初始化／内存映射／offload 调用；

只需在 dpa_a2a() 这类函数里替换通信模式，并在 kernel 中调整数据分发的调度循环，无需额外网络或安全依赖；

### Broadcast
功能：将根进程（root）中的一份数据，复制并分发给所有参与进程。
应用场景：在多进程中初始化共享参数（比如模型权重、配置结构体等），只需要一个根进程准备，然后广播给所有节点。

通信模式：可以实现成树状（tree-based）或管道化（pipeline）等方式，DPA offload 时可以让 DPU 在网络层面直接完成复制转发，减少 CPU 干预。

### all-reduce

功能：先对每个进程本地的数组/向量执行元素级的归约运算（如求和、最大值、最小值等），然后把结果分发到所有进程，即“归约 + 广播”。
应用场景：分布式训练中的梯度聚合、统计量汇总、全局最值计算等，需要所有进程都获得相同的全局结果。
通信模式：一般用树状分步归约（reduce）再广播（broadcast），或者环形（ring‐allreduce）算法；用 DPA offload 可在 DPU 上直接流水线处理数据分片，进一步降低延迟并提高带宽利用。

### Reduce_scatter

功能：先对所有进程对应位置的数据执行归约操作，然后将归约后的结果分散（scatter）到各个进程上——每个进程只收到归约后某一段（或某几个元素）结果。
应用场景：当最终每个进程只需要全局结果的一个分片时使用，比如分布式矩阵乘法、分布式排序中局部归约等，可减少广播带来的冗余传输。
通信模式：可用分阶段环形算法或树形算法实现；DPA offload 可在 DPU 层面分片处理并直接把对应分片发给目标进程，免去 CPU 二次转发。


---

### 注意事项

1. 由于编译系统复杂，优先考虑在原有的 all to all 的 code base 上修改，所以大家注意备份原有的版本以及别的同学修改的版本。


dpa_a2a_init
    create_dpa_context
    create_dpa_a2a_events
    prepare_dpa_a2a_rdmas
    prepare_dpa_a2a_memory
    prepare_dpa_a2a_memory



## all reduce

```bash
meson /tmp/build/try/ad \
  -Denable_all_applications=false \
  -Denable_allreduce=true
ninja -C /tmp/build/try/ad

#



```

查看当前的 BlueField:  sudo mst status -v


### 参考文档

https://docs.nvidia.com/doca/archive/2-7-0/nvidia+doca+allreduce+application+guide/index.html

```

export UCX_NET_DEVICES=mlx5_2:1 


/tmp/build/try/ad/allreduce/doca_allreduce -r daemon -t 34001 -c 1 -s 100 -o sum -d float -b 16 -i 16

```


针对 busy 的处理方式：

```bash
# 节点 A
export UCX_NET_DEVICES=mlx5_2:1
UCX_LOG_LEVEL=info ./doca_allreduce -r daemon -t 35001 -c 1 -a 192.168.1.100:35001  -s 65535 -o sum -d float -i 16 -b 128
# -a 10.21.211.3:35001 daemon 的本地监听地址


# 节点 B
export UCX_NET_DEVICES=mlx5_2:1
./doca_allreduce -r client -m non-offloaded -t 36001 -a 192.168.1.100:35001  -s 65535 -i 16 -b 128 -o sum -d float
UCX_LOG_LEVEL=info ./doca_allreduce -r client -m non-offloaded -t 34001 -a [fe80::ee0d:9aff:fe63:3072]:35001  -s 65535 -i 16 -b 128 -o sum -d float

# -a 10.21.211.3:35001 要连接的 daemon 地址（节点 A） 
```

从你提供的信息看：

p3 网卡当前只有一个 IPv6 链路本地地址（fe80::...），

没有绑定任何 IPv4 地址，比如 10.21.211.3。

这就是 UCX 报错 RDMA_CM_EVENT_ADDR_ERROR 的原因，因为你用 -a 10.21.211.3:35001 指定的地址并不在 mlx5_2:1（即 p3）这个网卡上。

export UCX_NET_DEVICES=enp3s0f3np3


找到网卡：
ip addr show p3

sudo ip addr add 10.21.211.3/24 dev p3
但是这是之后会出现 SegV 的问题

sudo ip addr del 10.21.211.3/24 dev p3

---

所以不手动添加 ipv4 地址，而是尝试在 -a 后使用 ipv6 地址


ibv_devices 查看 RDMA 的设备


show_gids 是一个用于 显示 RDMA（远程直接内存访问）设备的 Global Identifier (GID) 的工具，通常与 Mellanox/NVIDIA 网卡（如 mlx5 系列）相关。