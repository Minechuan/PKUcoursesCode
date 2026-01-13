#include "../test_utils/common.h"
// #include "utils.h" // 如果没有特别需要可以注释掉
char _license[] SEC("license") = "GPL";

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

#ifndef LIBBPF_PIN_BY_NAME
#define LIBBPF_PIN_BY_NAME 1
#endif

// 路由器的广域网地址 10.0.0.2
#define WAN_IP 0x0200000a 

struct five_tuple {
    __u32 ip;
    __u16 port;
    __u8 proto;
    __u8 padding;
};

// Map: 下行映射 (WAN Port -> LAN Tuple)
// 关键：必须与 nat_module_1.c 中的定义完全一致（包括 Pinning），才能共享
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u16); 
    __type(value, struct five_tuple); 
    __uint(pinning, LIBBPF_PIN_BY_NAME); // 关键：开启 Pinning
} nat_in SEC(".maps");

/*
|IP from 代理服务器 to 路由器 WAN| UDP from 代理服务端 to NAT 端口   |IP from 客户端 to 目标|TCP/UDP|
to
|IP from 代理服务器 to 客户端    | UDP from 代理服务端 to 代理客户端 |IP from 客户端 to 目标|TCP/UDP|

需要修改 ip 的目的地址和 UDP/TCP 的目的端口
*/


/*
XDP_PASS: 递交内核继续处理
XDP_DROP: 丢弃报文
XDP_TX: 将修改后的报文从来的地方发回去

*/

SEC("xdp_ingress")
int xdp_ingress_func(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    // 解析以太网头部
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_DROP;

    // 仅仅处理 IPv4 报文
    // 重要：不能丢弃 ARP 等非 IPv4，否则会导致路由器无法解析 WAN 侧邻居 MAC，进而触发 ICMP host unreachable
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;

    // 解析 IP 头部
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_DROP;

    // 文档要求：若其目的地址不是该路由器已分配的广域网地址，则直接丢弃
    if (ip->daddr != WAN_IP) return XDP_DROP;

    __u16 dst_port = 0;
    __u16 *port_ptr = NULL;
    __u16 *check_ptr = NULL;
    __u8 proto = ip->protocol;

    if (proto == IPPROTO_UDP) {
        struct udphdr *udp = (void *)(ip + 1);
        // 检查 UDP 头部边界
        if ((void *)(udp + 1) > data_end) return XDP_DROP;
        dst_port = bpf_ntohs(udp->dest);
        port_ptr = &udp->dest;
        check_ptr = &udp->check;
    } else if (proto == IPPROTO_TCP) {
        struct tcphdr *tcp = (void *)(ip + 1);
        if ((void *)(tcp + 1) > data_end) return XDP_DROP;
        dst_port = bpf_ntohs(tcp->dest);
        port_ptr = &tcp->dest;
        check_ptr = &tcp->check;
    } else {
        return XDP_PASS;
    }

    // 查找映射表
    struct five_tuple *real_host = bpf_map_lookup_elem(&nat_in, &dst_port);
    
    // 如果没有映射，说明不是 NAT 回包，放行给内核（可能是 Ping 路由器本身等）
    if (!real_host) return XDP_PASS;

    // 防止端口复用/误匹配：协议不同则不做 DNAT
    if (real_host->proto != proto) return XDP_PASS;

    // 修改目的端口
    __u16 new_port = real_host->port; // 主机端口是主机字节序
    __u16 old_port_net = *port_ptr;
    __u16 new_port_net = bpf_htons(new_port);
    *port_ptr = new_port_net;
    if (proto == IPPROTO_UDP) {
        // UDP 校验和可选，置 0 简化处理
        *check_ptr = 0;
    }

    // 修改 IP 目的地址
    __u32 old_ip = ip->daddr;
    __u32 new_ip = real_host->ip;
    


    // 更新 IP 校验和
    __u32 ip_csum = (~ip->check & 0xFFFF);
    ip_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, ip_csum);
    ip_csum = (ip_csum & 0xFFFF) + (ip_csum >> 16);
    ip_csum = (ip_csum & 0xFFFF) + (ip_csum >> 16);
    ip->check = ~ip_csum;

    // 最后更新 IP 地址
    ip->daddr = new_ip;

    // TCP 需要保持校验和正确（UDP 已置 0）
    if (proto == IPPROTO_TCP) {
        __u32 l4_csum = (~(*check_ptr) & 0xFFFF);
        l4_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, l4_csum);
        __be32 old_port_be32 = (__be32)old_port_net;
        __be32 new_port_be32 = (__be32)new_port_net;
        l4_csum = bpf_csum_diff(&old_port_be32, 2, &new_port_be32, 2, l4_csum);
        l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
        l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
        *check_ptr = ~l4_csum;
    }

    // 递交内核网络协议栈继续处理 (路由转发回内网)
    return XDP_PASS;
}