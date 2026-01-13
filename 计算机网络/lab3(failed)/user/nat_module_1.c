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

// 确保 PIN_BY_NAME 定义存在
#ifndef LIBBPF_PIN_BY_NAME
#define LIBBPF_PIN_BY_NAME 1
#endif

// 路由器的广域网地址 10.0.0.2 (0x0200000a)
#define WAN_IP 0x0200000a


struct five_tuple {
    __u32 ip;
    __u16 port;
    __u8 proto;
    __u8 padding;
};

// Map 1: 上行映射 (LAN -> WAN)
// Key: 原内网 IP + 原端口 + 协议
// Value: 分配的 WAN 端口
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct five_tuple);
    __type(value, __u16);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // 关键：开启 Pinning
} nat_out SEC(".maps");


// Map 2: 下行映射 (WAN -> LAN)
// Key: WAN 端口
// Value: 原内网 IP + 原端口 + 协议
// 必须 Pinning，以便下行模块 nat_module_2 能访问同一个 Map
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u16);
    __type(value, struct five_tuple);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // 关键：开启 Pinning
} nat_in SEC(".maps");





/*
|IP from 客户端 to 代理服务器      | UDP 代理客户端 to 代理服务端           |IP from 客户端 to 目标|TCP/UDP|

|IP from 路由器 WAN to 代理服务器  | UDP from 路由器 NAT 端口 to 代理服务端 |IP from 客户端 to 目标|TCP/UDP|

需要修改 ip 的源地址和 UDP/TCP 的源端口
*/





SEC("tc_egress")
int tc_egress_func(struct __sk_buff* skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return TC_ACT_SHOT;

    // 仅处理 IPv4，否则放行
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return TC_ACT_OK;

    // 解析 IP 头部
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return TC_ACT_SHOT;

    // 如果源地址已经是 WAN IP (比如路由器自己发出的包)，不需要做 SNAT
    if (ip->saddr == WAN_IP) return TC_ACT_OK;

    // 解析 L4 头部
    __u16 old_port = 0;
    __u16 *src_port_ptr = NULL;
    __u16 *check_ptr = NULL;

    __u8 proto = ip->protocol;
    if (proto == IPPROTO_UDP) {
        struct udphdr *udp = (void *)(ip + 1);
        if ((void *)(udp + 1) > data_end) return TC_ACT_SHOT;
        old_port = bpf_ntohs(udp->source);
        src_port_ptr = &udp->source;
        check_ptr = &udp->check;
    } else if (proto == IPPROTO_TCP) {
        struct tcphdr *tcp = (void *)(ip + 1);
        if ((void *)(tcp + 1) > data_end) return TC_ACT_SHOT;
        old_port = bpf_ntohs(tcp->source);
        src_port_ptr = &tcp->source;
        check_ptr = &tcp->check;
    } else {
        return TC_ACT_OK; // 非 TCP/UDP 不处理
    }

    // 构造 five_tuple 作为查找键
    struct five_tuple key = {
        .ip = ip->saddr,
        .port = old_port,
        .proto = proto,
        .padding = 0
    };

    // 1. 查找现有映射
    __u16 *mapped_port = bpf_map_lookup_elem(&nat_out, &key);
    __u16 new_port = 0;

    if (mapped_port) {
        new_port = *mapped_port;
    } else {
        // 2. 如果不存在，分配新端口
        // 范围限制在 1024-61024 之间
        new_port = (bpf_get_prandom_u32() % 60000) + 1024; // 主机字节序

        // 检查端口冲突 (查询 nat_in)
        if (bpf_map_lookup_elem(&nat_in, &new_port)) {
            return TC_ACT_SHOT; // 冲突丢弃，等待重传
        }

        // 建立双向映射
        bpf_map_update_elem(&nat_out, &key, &new_port, BPF_ANY);
        bpf_map_update_elem(&nat_in, &new_port, &key, BPF_ANY);
    }

    // 3. 执行修改 (SNAT)
    // 修改源端口
    __u16 old_port_net = *src_port_ptr;
    __u16 new_port_net = bpf_htons(new_port);
    *src_port_ptr = new_port_net;
    if (proto == IPPROTO_UDP) {
        // UDP 校验和可选，置 0 简化处理
        *check_ptr = 0;
    }


    // 改写源 IP头
    __u32 old_ip_addr = ip->saddr;
    __u32 new_ip_addr = WAN_IP;
    // 更新 IP 校验和
    __u32 ip_csum = (~ip->check & 0xFFFF);
    ip_csum = bpf_csum_diff(&old_ip_addr, 4, &new_ip_addr, 4, ip_csum);
    ip_csum = (ip_csum & 0xFFFF) + (ip_csum >> 16);
    ip_csum = (ip_csum & 0xFFFF) + (ip_csum >> 16);
    ip->check = ~ip_csum;

    // 最后更新 IP 地址
    ip->saddr = new_ip_addr;

    // TCP 需要保持校验和正确（UDP 已置 0）
    if (proto == IPPROTO_TCP) {
        __u32 l4_csum = (~(*check_ptr) & 0xFFFF);
        l4_csum = bpf_csum_diff(&old_ip_addr, 4, &new_ip_addr, 4, l4_csum);
        __be32 old_port_be32 = (__be32)old_port_net;
        __be32 new_port_be32 = (__be32)new_port_net;
        l4_csum = bpf_csum_diff(&old_port_be32, 2, &new_port_be32, 2, l4_csum);
        l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
        l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
        *check_ptr = ~l4_csum;
    }

    return TC_ACT_OK;
}