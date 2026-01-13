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

#ifndef LIBBPF_PIN_BY_NAME
#define LIBBPF_PIN_BY_NAME 1
#endif

// ================= 常量定义 =================
#define TARGET_IP 0x0300000a                            // 10.0.0.3
#define TARGET_MAC_BYTES {0x02, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6}

#define PROXY_IP 0x0400000a                             // 10.0.0.4
#define PROXY_LISTEN_PORT 12345                        // 代理服务器接收封装报文的 UDP 端口               

#define NAT_ROUTER_IP 0x0200000a                        // 10.0.0.2



/*
上行模块：
1. 去除封装，恢复原始报文
2. 构造行为类似于 NAT 的 SNAT 映射
(局域网地址，局域网端口，传输层协议) -> （代理服务器端口，传输层协议）

下行模块：
1. 查找映射
2. 如果有映射，恢复原始报文

*/





// ================= Map 定义 =================
struct session_key {
    __u32 ip; // 局域网 ip
    __u16 port; // 局域网端口
    __u8 proto; // 传输层协议
    __u8 padding;
};

struct session_val {
    __u16 proxy_port; // 代理服务端端口
    __u16 proto; // 传输层协议
};

struct tunnel_info {
    __u16 tunnel_port;   // outer_udp->source (网络序)
    __u8  router_mac[6]; // 该封装报文的二层源 MAC（NAT 路由器 MAC）
};

// Map 3: 记录 NAT 路由器侧的“隧道信息”（端口 + MAC）
// Key 使用 proxy_port + proto（不引入 MAC 进 key）
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct session_val);
    __type(value, struct tunnel_info);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} tunnel_ports SEC(".maps");

// Map 1: 正向会话
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct session_key);
    __type(value, struct session_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} sessions SEC(".maps");


// Map 2: 反向会话
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, struct session_val);
    __type(value, struct session_key);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} sessions_rev SEC(".maps");



// 端口分配器
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u16);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} proxy_port_alloc SEC(".maps");

// ================= 辅助函数 =================
static __always_inline __u16 alloc_proxy_port() {
    __u32 key = 0;
    __u16 *port = bpf_map_lookup_elem(&proxy_port_alloc, &key);
    __u16 p = 30000;
    if (port) {
        p = *port;
        p++;
        if (p < 30000 || p > 60000) p = 30000;
    }
    bpf_map_update_elem(&proxy_port_alloc, &key, &p, BPF_ANY);
    return bpf_htons(p);
}

static __always_inline __u16 calc_ip_csum(struct iphdr *ip) {
    __u32 csum = 0;
    __u16 *ptr = (__u16 *)ip;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 10; i++) {
        csum += ptr[i];
    }
    csum = (csum & 0xFFFF) + (csum >> 16);
    csum = (csum & 0xFFFF) + (csum >> 16);
    return ~csum;
}

// ================= 主逻辑 =================
SEC("xdp_ingress")
int xdp_ingress_func(struct xdp_md *ctx) {
    // 获取数据指针
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end) return XDP_DROP;
    // 不能丢弃 ARP 等非 IPv4，否则会影响邻居解析/连通性
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;

    // 获取 MAC 信息
    unsigned char self_mac[6];
    __builtin_memcpy(self_mac, eth->h_dest, 6);


    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end) return XDP_DROP;

    // ============================================================
    // 逻辑 1: 来自 NAT 的请求 -> Target
    // ============================================================
    if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = (void *)(ip + 1);
        if ((void *)(udp + 1) > data_end) return XDP_PASS;

        if (udp->dest == bpf_htons(PROXY_LISTEN_PORT)) {
            // 先保存隧道信息（NAT 路由器映射后的端口 + 路由器 MAC），用于回包封装
            struct tunnel_info tinfo = {0};
            tinfo.tunnel_port = udp->source;
            __builtin_memcpy(tinfo.router_mac, eth->h_source, 6);
            
            // 1. 去除 IP 头部和 UDP 头部 (解包)
            // 计算外层头部长度: IP头(20) + UDP头(8) = 28字节 (简化计算，未考虑 IP Options)
            int decap_len = sizeof(struct iphdr) + sizeof(struct udphdr);
            
            // 移动以太网头部：将 Eth Header 向后(内存高地址)移动 28 字节
            // 实际上 XDP 是 adjust_head 增加 data 指针，所以我们需要把 eth 头搬到新 data 的位置
            // [Eth][IP][UDP][InnerIP]...  ->  [Eth][InnerIP]...
            
            // 这里的逻辑稍微 tricky：先 adjust_head，data 指针会变，原来的 eth 指针失效
            // 所以先调整，再重新解析
            if (bpf_xdp_adjust_head(ctx, decap_len)) return XDP_DROP;

            // 重新获取指针
            data = (void *)(long)ctx->data;
            data_end = (void *)(long)ctx->data_end;
            
            struct ethhdr *new_eth = data;
            // struct ethhdr *old_eth_ptr = (void *)((char *)data - decap_len); // 这是一个逻辑位置，实际上不能这么读

            // 更安全的做法：在 adjust_head 之前 memmove，或者利用 bpf_xdp_store_bytes (如果支持)
            // XDP 标准做法：
            // 1. 我们直接把 header 里的 MAC 地址填对就行了，原来的 Eth 头其实可以丢弃或重写
            // 2. 这里我们重构 Eth 头
            if ((void *)(new_eth + 1) > data_end) return XDP_DROP;
            
            // 恢复/构造 Ethernet 头部
            // 原代码可能希望保留 sender MAC，但 adjust_head 后比较麻烦
            // 简单处理：源 MAC 设为 Proxy，目的 MAC 设为 Target (稍后设置)
            new_eth->h_proto = bpf_htons(ETH_P_IP);

            // 解析内层 IP
            struct iphdr *inner_ip = (void *)(new_eth + 1);
            if ((void *)(inner_ip + 1) > data_end) return XDP_DROP;

            // 2. 构造/查找正向映射
            struct session_key key = {0};
            key.ip = inner_ip->saddr;       // 原始 Client IP
            key.proto = inner_ip->protocol;
            key.padding = 0;

            // 获取内层端口
            __u16 *sport_ptr = NULL;
            __u16 *check_ptr = NULL;

            if (inner_ip->protocol == IPPROTO_TCP) {
                struct tcphdr *t = (void *)(inner_ip + 1);
                if ((void *)(t + 1) > data_end) return XDP_DROP;
                key.port = t->source;
                sport_ptr = &t->source;
                check_ptr = &t->check;
            } else if (inner_ip->protocol == IPPROTO_UDP) {
                struct udphdr *u = (void *)(inner_ip + 1);
                if ((void *)(u + 1) > data_end) return XDP_DROP;
                key.port = u->source;
                sport_ptr = &u->source;
                check_ptr = &u->check;
            } else {
                return XDP_DROP; // 不支持的协议
            }

            struct session_val *val = bpf_map_lookup_elem(&sessions, &key);
            __u16 proxy_port;

            if (val) {
                proxy_port = val->proxy_port;
            } else {
                // 3. 不存在则分配，并插入反向映射
                proxy_port = alloc_proxy_port();
                
                struct session_val new_val = {0};
                new_val.proxy_port = proxy_port;
                new_val.proto = key.proto;
                
                // 更新正向表
                bpf_map_update_elem(&sessions, &key, &new_val, BPF_ANY);
                
                // 更新反向表 (Key: ProxyPort+Proto -> Val: ClientIP+Port)
                // 这里反向表的 Key 是 struct session_val
                bpf_map_update_elem(&sessions_rev, &new_val, &key, BPF_ANY);
            }

            // 刷新 tunnel port：Key 用本会话的 proxy_port+proto
            // 注意：proxy_port 是网络序，proto 是主机序（但我们全程一致即可）
            struct session_val tp_key = {0};
            tp_key.proxy_port = proxy_port;
            tp_key.proto = key.proto;
            bpf_map_update_elem(&tunnel_ports, &tp_key, &tinfo, BPF_ANY);

            // 4. 执行 SNAT (Source NAT)
            // 将源 IP 改为 Proxy IP，源端口改为 Proxy Port
            __u32 old_ip = inner_ip->saddr;
            __u32 new_ip = PROXY_IP;
            inner_ip->saddr = new_ip;

            // 更新 IP 校验和
            __u32 l3_csum = (~inner_ip->check & 0xFFFF);
            l3_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, l3_csum);
            inner_ip->check = ~((l3_csum & 0xFFFF) + (l3_csum >> 16));

            // 更新 L4 (TCP/UDP) 端口和校验和
            __u32 old_port_val = *sport_ptr;
            __u32 new_port_val = proxy_port;
            *sport_ptr = proxy_port;

            // 计算 L4 校验和 diff (包含伪头部 IP 变更和端口变更)
            __u32 l4_csum = (~(*check_ptr) & 0xFFFF);
            l4_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, l4_csum);       // IP 变更
            l4_csum = bpf_csum_diff(&old_port_val, 4, &new_port_val, 4, l4_csum); // Port 变更
            l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
            l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
            *check_ptr = ~l4_csum;

            // 5. 修改 MAC 地址 (Proxy -> Target)
            __builtin_memcpy(new_eth->h_source, self_mac, 6);
            unsigned char target_mac[6] = TARGET_MAC_BYTES;
            __builtin_memcpy(new_eth->h_dest, target_mac, 6);

            return XDP_PASS;
        }
    }
    

    // ============================================================
    // 逻辑 2: 来自 Target 的回复 -> NAT
    // ============================================================
    if (ip->daddr == PROXY_IP) {
        
        // 获取目的端口
        __u16 dst_port = 0;
        __u16 *dport_ptr = NULL;
        __u16 *check_ptr = NULL;
        __u8 proto = ip->protocol;

        if (proto == IPPROTO_TCP) {
            struct tcphdr *t = (void *)(ip + 1);
            if ((void *)(t + 1) > data_end) return XDP_PASS;
            dst_port = t->dest;
            dport_ptr = &t->dest;
            check_ptr = &t->check;
        } else if (proto == IPPROTO_UDP) {
            struct udphdr *u = (void *)(ip + 1);
            if ((void *)(u + 1) > data_end) return XDP_PASS;
            dst_port = u->dest;
            dport_ptr = &u->dest;
            check_ptr = &u->check;
        } else {
            return XDP_PASS;
        }

        // 1. 查找反向映射
        struct session_val rev_key = {0};
        rev_key.proxy_port = dst_port;
        rev_key.proto = proto;

        struct session_key *real_client = bpf_map_lookup_elem(&sessions_rev, &rev_key);

        if (real_client) {
            // 2. 执行 DNAT (恢复原始 Client IP 和 Port)
            __u32 old_ip = ip->daddr;
            __u32 new_ip = real_client->ip;
            ip->daddr = new_ip;

            // 更新 IP 校验和
            __u32 l3_csum = (~ip->check & 0xFFFF);
            l3_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, l3_csum);
            ip->check = ~((l3_csum & 0xFFFF) + (l3_csum >> 16));

            // 更新 L4 Port 和校验和
            __u32 old_port = dst_port;
            __u32 new_port = real_client->port;
            *dport_ptr = new_port;

            __u32 l4_csum = (~(*check_ptr) & 0xFFFF);
            l4_csum = bpf_csum_diff(&old_ip, 4, &new_ip, 4, l4_csum);
            l4_csum = bpf_csum_diff(&old_port, 4, &new_port, 4, l4_csum);
            l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
            l4_csum = (l4_csum & 0xFFFF) + (l4_csum >> 16);
            *check_ptr = ~l4_csum;

            int encap_len = sizeof(struct iphdr) + sizeof(struct udphdr);
            
            // 扩展头部空间 (注意参数是负数，表示向前扩展)
            if (bpf_xdp_adjust_head(ctx, -encap_len)) return XDP_DROP;

            data = (void *)(long)ctx->data;
            data_end = (void *)(long)ctx->data_end;
            
            struct ethhdr *new_eth = data;
            // 旧的 Eth 头位置 (现在位于 data + encap_len)
            struct ethhdr *old_eth = (void *)((char *)data + encap_len);

            // 边界检查：确保旧头部在有效范围内 (非常重要，否则校验器会报错)
            if ((void *)(old_eth + 1) > data_end) return XDP_DROP;

            // 将旧的 Ethernet 头部拷贝到最前面
            // 这会自动复制 h_proto，以及当时的 MAC 地址 (Target -> Proxy)
            *new_eth = *old_eth;


            // 回包需要发回 NAT 路由器：MAC 从 tunnel_ports 学习得到
            struct tunnel_info *ti = bpf_map_lookup_elem(&tunnel_ports, &rev_key);
            if (ti) {
                __builtin_memcpy(new_eth->h_dest, ti->router_mac, 6);
            }
            __builtin_memcpy(new_eth->h_source, self_mac, 6);



            // 填充外层 IP
            struct iphdr *outer_ip = (void *)(new_eth + 1);
            struct udphdr *outer_udp = (void *)(outer_ip + 1);
            // 内层 IP 紧接着 Outer UDP
            struct iphdr *payload_ip = (void *)(outer_udp + 1); 

            if ((void *)(payload_ip + 1) > data_end) return XDP_DROP;

            outer_ip->version = 4;
            outer_ip->ihl = 5;
            outer_ip->tos = 0;
            // Total Len = Inner Total Len + 28
            outer_ip->tot_len = bpf_htons(bpf_ntohs(payload_ip->tot_len) + encap_len); 
            outer_ip->id = 0;
            outer_ip->frag_off = 0;
            outer_ip->ttl = 64;
            outer_ip->protocol = IPPROTO_UDP;
            outer_ip->saddr = PROXY_IP;
            outer_ip->daddr = NAT_ROUTER_IP;
            outer_ip->check = 0;
            outer_ip->check = calc_ip_csum(outer_ip);

            // 填充外层 UDP
            outer_udp->source = bpf_htons(PROXY_LISTEN_PORT);
            // 外层 UDP 目的端口必须是 NAT 路由器为隧道分配的端口（即请求报文的 outer_udp->source）
            outer_udp->dest = ti ? ti->tunnel_port : real_client->port;
            // 简单起见，假设通信是对称的
            outer_udp->len = bpf_htons(bpf_ntohs(outer_ip->tot_len) - sizeof(struct iphdr));
            outer_udp->check = 0; // UDP Checksum 可选

            // 这里必须发出去（ns4 不是路由器，XDP_PASS 只会上送内核并被丢弃）
            return XDP_PASS;
        }
    }

    return XDP_PASS;

}