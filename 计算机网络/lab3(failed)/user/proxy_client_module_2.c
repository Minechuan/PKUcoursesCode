#include "../test_utils/common.h"


char _license[] SEC("license") = "GPL";

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include <linux/udp.h>
#include <linux/in.h>
#include <linux/pkt_cls.h> // for TC_ACT_OK
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>


#define PROXY_IP 0x0400000a
// 代理服务器接收封装报文的 UDP 端口
#define PROXY_PORT 12345 

#ifndef LIBBPF_PIN_BY_NAME
#define LIBBPF_PIN_BY_NAME 1
#endif

/*
|IP from 代理服务器 to 客户端 | UDP from 代理服务器端口 | IP from 目标 to 客户端 | TCP/UDP|

to 

|IP from 目标 to 客户端 | TCP/UDP|

TC_ACT_OK: 保持报文继续传递
TC_ACT_SHOT: 丢弃报文
*/




SEC("tc_egress")
int tc_egress_func(struct __sk_buff *skb) {
    void *data_end = (void *)(long)skb->data_end;
    void *data = (void *)(long)skb->data;

    struct ethhdr *eth = data;
    // 先确保以太网头部在数据范围内
    if ((void *)(eth + 1) > data_end) return TC_ACT_SHOT;

    // 仅处理 IPv4
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return TC_ACT_OK;

    // 解析 IP 头部
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return TC_ACT_SHOT;

    // 1. 检查是否来自代理服务器
    // 检查源 IP
    if (ip->saddr != PROXY_IP) return TC_ACT_OK;
    // 检查是否为 UDP 封装
    if (ip->protocol != IPPROTO_UDP) return TC_ACT_OK;

    // 解析 UDP 头部
    struct udphdr *udp = (void *)(ip + 1);
    if ((void *)(udp + 1) > data_end) return TC_ACT_OK;

    // 2. 检查源端口 (进一步确认)
    if (udp->source != bpf_htons(PROXY_PORT)) return TC_ACT_OK;

    // 3. 执行解包 (Decapsulation)
    // 需要移除外层 IP (20 bytes) + 外层 UDP (8 bytes) = 28 bytes

    // 此时报文结构: [ETH] [Outer IP] [Outer UDP] [Inner IP] [Payload]
    // 目标结构:    [ETH] [Inner IP] [Payload]
    
    // bpf_skb_adjust_room 用于在 skb 中增加或减少数据空间
    // BPF_ADJ_ROOM_NET: 相对于网络层 (L3) 开始调整。
    // len: -28 (负数表示移除)
    // 该函数会自动移动 Ethernet 头部以适配新的 L3 起始位置
    __s32 len_diff = -(__s32)(sizeof(struct iphdr) + sizeof(struct udphdr));

    if (bpf_skb_adjust_room(skb, len_diff, BPF_ADJ_ROOM_NET, 0)) {
        // 如果调整失败，丢弃报文
        return TC_ACT_SHOT;
    }

    // 调整成功后，报文已还原为 [ETH] [Inner IP] ...
    // Ethernet 头部的 h_proto 依然是 IP (0x0800)，无需修改
    // 内层报文的校验和是在源端生成的，解包后依然有效，无需重算

    return TC_ACT_OK;
}