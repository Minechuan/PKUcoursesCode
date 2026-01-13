#include "../test_utils/common.h"

char _license[] SEC("license") = "GPL";

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

// 目标服务器 IP: 10.0.0.3 (0x0a, 0x00, 0x00, 0x03 -> Little Endian: 0x0300000a)
#define DST_IP 0x0300000a

#define PROXY_IP 0x0400000a
// 代理服务器接收封装报文的 UDP 端口
#define PROXY_PORT 12345 // 相对应的 proxy server 的端口也是 12345

#ifndef LIBBPF_PIN_BY_NAME
#define LIBBPF_PIN_BY_NAME 1
#endif



/*
                                                          |IP from 客户端 to 目标|TCP/UDP|

|IP from 客户端 to 代理服务器 | UDP 代理客户端 to 代理服务端 |IP from 客户端 to 目标|TCP/UDP|

需要添加 ip 头部和 udp 头部进行封装
*/


// 代理规则 Key (五元组简化版，因为源不重要，看目的)

// |IP to 代理服务端 | UDP to 代理服务端 | IP from 局域网| TCP/UDP from 局域网 |Data|
struct proxy_key {
    __u32 dst_ip;   // 规则匹配的目标 IP
    __u16 dst_port; // 规则匹配的目标端口
    __u8 proto;     // 规则匹配的协议 (TCP/UDP)
    __u8 padding;   // 对齐
};

// to: 取消封装的头
// |IP from 局域网|TCP/UDP from 局域网|Data|
// to: 伪装成从代理服务器访问该报文的目标机器
// |IP from 代理服务器|TCP/UDP from 代理服务器端口|




// Map: 存储代理规则
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, struct proxy_key);
    __type(value, __u8); 
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} proxy_rules SEC(".maps");

// 辅助函数：计算 IP 校验和
static __always_inline __u16 calc_ip_csum(struct iphdr *ip) {
    __u32 csum = 0;
    __u16 *ptr = (__u16 *)ip;

    // IP 头固定 20 字节，循环 10 次
    #pragma clang loop unroll(full)
    for (int i = 0; i < 10; i++) {
        csum += ptr[i];
    }

    // Fold carry
    csum = (csum & 0xFFFF) + (csum >> 16);
    csum = (csum & 0xFFFF) + (csum >> 16);
    return ~csum;
}


SEC("xdp_ingress")
int xdp_ingress_func(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    
    // data 为空丢弃
    if ((void *)(eth + 1) > data_end) return XDP_DROP;

    // 不能丢弃 ARP 等非 IPv4，否则会影响邻居解析/连通性
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;

    /*检查 ip 头*/
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_DROP;

    __u16 src_port = 0;
    __u16 dst_port = 0;
    // 解析 L4 获取端口
    // 一定是 TCP 或 UDP
    if (ip->protocol == IPPROTO_TCP) {
        struct tcphdr *tcp = (void *)(ip + 1);
        if ((void *)(tcp + 1) > data_end) return XDP_DROP;
        src_port = tcp->source;
        dst_port = tcp->dest;
    } else if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = (void *)(ip + 1);
        if ((void *)(udp + 1) > data_end) return XDP_DROP;
        src_port = udp->source;
        dst_port = udp->dest;
    } else {
        return XDP_PASS;
    }

    // ------------------------------------------------------
    // 1. 处理配置报文 (src_port == 123/124) 
    //  插入规则 : 123
    //  删除规则 : 124
    // ------------------------------------------------------
    if (src_port == bpf_htons(123) || src_port == bpf_htons(124)) {
        struct proxy_key key = {
            .dst_ip = ip->daddr,     // 规则的目标 IP
            .dst_port = dst_port,    // 规则的目标端口
            .proto = ip->protocol,   // 规则的协议
            .padding = 0
        };

        if (src_port == bpf_htons(123)) {// 插入规则
            __u8 val = 1;
            bpf_map_update_elem(&proxy_rules, &key, &val, BPF_ANY);
        } else {
            bpf_map_delete_elem(&proxy_rules, &key);
        }
        
        // 配置报文不应继续转发，消耗掉
        return XDP_DROP; 
    }

    // ------------------------------------------------------
    // 2. 检查是否命中代理规则
    // ------------------------------------------------------
    struct proxy_key search_key = {
        .dst_ip = ip->daddr,
        .dst_port = dst_port,
        .proto = ip->protocol,
        .padding = 0
    };

    if (!bpf_map_lookup_elem(&proxy_rules, &search_key)) {
        // 未命中规则，正常转发，不使用代理
        return XDP_PASS;
    }

    // ------------------------------------------------------
    // 3. 执行 UDP 封装 (Encapsulation)
    // 封装新的 ethheader 和 ipheader
    // ------------------------------------------------------
    
    // 计算需要扩展的空间: 外层 IP (20) + 外层 UDP (8) = 28 字节
    int headroom = sizeof(struct iphdr) + sizeof(struct udphdr);

    // 在头部增加空间
    if (bpf_xdp_adjust_head(ctx, -headroom)) return XDP_PASS; // 失败则放行原报文

    // 指针变化，必须重新读取
    data = (void *)(long)ctx->data;
    data_end = (void *)(long)ctx->data_end;

    // 新的 Eth 头位置
    struct ethhdr *new_eth = data;
    // 旧的 Eth 头位置 (现在位于 data + headroom)
    struct ethhdr *old_eth = (void *)((char *)data + headroom);

    // 边界检查：确保旧头部在有效范围内
    if ((void *)(old_eth + 1) > data_end) return XDP_DROP;

    // 将 Ethernet 头部移动到最前面
    // 注意：struct 赋值会被编译器优化为 memcpy
    *new_eth = *old_eth;

    // 定义各层头部指针
    // 结构: [Eth] [Outer IP] [Outer UDP] [Inner IP (Old IP)] ...
    struct iphdr *outer_ip = (void *)(new_eth + 1);
    struct udphdr *outer_udp = (void *)(outer_ip + 1);
    struct iphdr *inner_ip = (void *)(outer_udp + 1); // 原报文的 IP 头

    // 边界检查
    if ((void *)(inner_ip + 1) > data_end) return XDP_DROP;

    // 构造外层 IPv4 头
    outer_ip->version = 4; // IPv4
    outer_ip->ihl = 5; // 无选项，头部长度 5 (20 bytes)
    outer_ip->tos = 0; // 通常继承 inner_ip->tos 或设为 0

    // 总长度 = 原总长度 + 28
    outer_ip->tot_len = bpf_htons(bpf_ntohs(inner_ip->tot_len) + headroom); // 外层 IP 总长度
    outer_ip->id = 0; // 不分片或随机
    outer_ip->frag_off = 0;  // 不分片
    outer_ip->ttl = 64; // 一般设为 64
    outer_ip->protocol = IPPROTO_UDP; // 下一层是 UDP
    outer_ip->saddr = inner_ip->saddr; // 源地址保持客户端 IP
    outer_ip->daddr = PROXY_IP;        // 目的地址改为代理服务器 IP
    outer_ip->check = 0;               // 先置 0

    // 计算外层 IP 校验和
    outer_ip->check = calc_ip_csum(outer_ip);

    // 构造外层 UDP 头
    // 源端口随机生成
    outer_udp->source = bpf_htons(PROXY_PORT + (bpf_get_prandom_u32() % 40000));


    outer_udp->dest = bpf_htons(PROXY_PORT); // 发往代理服务器的特定端口
    // UDP 长度 = IP 总长度 - IP 头长度 (20)
    outer_udp->len = bpf_htons(bpf_ntohs(outer_ip->tot_len) - sizeof(struct iphdr));
    outer_udp->check = 0; // UDP 校验和可选，置 0 即可

    // Ethernet 头部中的协议字段已经是 ETH_P_IP (0x0800)，因为是从 old_eth 复制过来的，无需修改

    return XDP_PASS;
}