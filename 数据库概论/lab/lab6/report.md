# FinanceBench BM25 + 向量混合检索实验报告

小组成员：毛川，芮思铭，关睿轩


## 1. 任务目标

本项目基于 FinanceBench 开源样本构建财报问答场景下的 Retrieve 实验，目标是比较以下三类检索方案在相关证据召回上的表现：

1. 纯 BM25 稀疏检索；
2. 基于开源 Embedding 模型的稠密向量检索；
3. BM25 与向量检索的混合检索融合。

实验重点不调用 LLM，也不调用任何外部 API，仅评估检索阶段能否将标注证据所在页面召回到 Top-K 结果中。核心指标为 Recall@5、Recall@10 和 MRR。

## 2. 数据与评测设置

使用 FinanceBench 开源数据：

- 问题文件：`financebench-main/data/financebench_open_source.jsonl`
- PDF 财报目录：`financebench-main/pdfs`
- 问题数量：150
- 涉及文档数量：84
- 评测正例：每个问题的 `evidence` 字段中标注的 `(evidence_doc_name, evidence_page_num)`

本实验将每个检索返回的 chunk 映射回其来源页面。如果 Top-K 检索结果中任一 chunk 来自人工标注的 evidence 页面，则认为该问题在 Recall@K 上命中。

## 3. 方法实现

代码位于 `code/financebench_retrieval`，主要模块如下：

- `pdf.py`：使用 PyMuPDF 抽取 PDF 页面文本，并进行基础清洗；
- `chunking.py`：按固定字符窗口进行页面内分块，默认 `chunk_size=1200`、`chunk_overlap=200`；
- `bm25.py`：自实现 BM25 检索，避免依赖额外 BM25 包；
- `dense.py`：使用 `BAAI/bge-base-en-v1.5` 生成归一化向量，并用余弦相似度检索；
- `fusion.py`：实现 Min-Max 分数归一化后的线性加权融合；
- `metrics.py`：实现 Recall@K 与 MRR；
- `experiment.py` 与 `cli.py`：组织完整实验流程和命令行入口。

向量模型使用：

```text
BAAI/bge-base-en-v1.5
```

线性融合公式为：

```text
score = w * normalized_bm25_score + (1 - w) * normalized_dense_score
```

其中 `w` 表示 BM25 权重。本次实验遍历：

```text
w = 0.0, 0.1, 0.2, ..., 1.0
```

检索候选池大小为 50，最终统计 Recall@5、Recall@10 和 MRR。

## 4. 实验结果

完整结果已保存至：

```text
code/outputs/metrics.csv
code/outputs/per_query_results.jsonl
```

指标汇总如下：

| 方法 | BM25 权重 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 1.0 | 0.1333 | 0.1667 | 0.1118 |
| Dense/BGE | 0.0 | 0.3733 | 0.4733 | 0.2739 |
| Hybrid | 0.0 | 0.3733 | 0.4733 | 0.2739 |
| Hybrid | 0.1 | 0.3733 | 0.4867 | 0.2711 |
| Hybrid | 0.2 | 0.3733 | 0.4800 | 0.2674 |
| Hybrid | 0.3 | 0.3600 | 0.4867 | 0.2626 |
| Hybrid | 0.4 | 0.3400 | 0.4533 | 0.2613 |
| Hybrid | 0.5 | 0.3000 | 0.3867 | 0.2285 |
| Hybrid | 0.6 | 0.2067 | 0.3400 | 0.1776 |
| Hybrid | 0.7 | 0.1667 | 0.2600 | 0.1543 |
| Hybrid | 0.8 | 0.1400 | 0.1933 | 0.1327 |
| Hybrid | 0.9 | 0.1333 | 0.1733 | 0.1238 |
| Hybrid | 1.0 | 0.1333 | 0.1667 | 0.1122 |

## 5. 结果分析

从实验结果可以看出，纯 BM25 在 FinanceBench 财报问答场景下表现明显弱于 BGE 向量检索。BM25 的 Recall@5 仅为 0.1333，Recall@10 为 0.1667；而 Dense/BGE 分别达到 0.3733 和 0.4733。这说明该任务中的问题表达与财报证据文本之间存在较强的语义匹配需求，单纯依靠关键词匹配难以稳定召回证据页。

混合检索在较低 BM25 权重下取得了更好的 Recall@10。其中 `bm25_weight=0.1` 时，Recall@10 从纯 Dense 的 0.4733 提升到 0.4867，Recall@5 保持 0.3733 不变。这说明少量 BM25 分数可以补充财报中的专业术语、会计科目、年份、表格关键词等精确匹配信号，从而改善 Top-10 召回。

当 BM25 权重继续增大时，整体效果开始下降。`bm25_weight >= 0.5` 后，Recall@5、Recall@10 和 MRR 均明显低于纯 Dense。这表明当前分块与页级证据评测设置下，BM25 更适合作为辅助信号，而不适合作为主导检索信号。

MRR 最高的是纯 Dense/BGE，值为 0.2739；`bm25_weight=0.1` 的 MRR 为 0.2711，略低于纯 Dense。这说明混合检索虽然改善了 Top-10 召回，但可能将部分正例页面从更靠前的位置移到稍后位置。因此如果目标是“尽量在 Top-10 中覆盖证据”，推荐使用混合权重 0.1；如果目标是“尽量让第一个正确证据排得更靠前”，纯 Dense 略优。

## 6. 最优配置

以 Recall@10 为主要优化目标时，最优配置为：

```text
Embedding 模型：BAAI/bge-base-en-v1.5
融合方式：Min-Max 归一化 + 线性加权融合
BM25 权重：0.1
Dense 权重：0.9
Recall@5：0.3733
Recall@10：0.4867
MRR：0.2711
```

该配置相比纯 BM25 有显著提升：

```text
Recall@5:  0.1333 -> 0.3733
Recall@10: 0.1667 -> 0.4867
MRR:       0.1118 -> 0.2711
```

相比纯 Dense，该配置在 Recall@10 上有小幅提升：

```text
Recall@10: 0.4733 -> 0.4867
```

## 7. BAAI/bge-m3 追加实验

在完成 `BAAI/bge-base-en-v1.5` 实验后，进一步尝试了更强的多语言/多功能 Embedding 模型：

```text
BAAI/bge-m3
```

实验设置保持不变：

- 使用同一批 150 个 FinanceBench 问题；
- 使用同样的 PDF 抽取结果和 chunk 切分方式；
- BM25 检索、候选池大小、融合方式、权重网格均保持一致；
- 输出目录：`code/outputs_bge_m3`

bge-m3 结果如下：

| 方法 | BM25 权重 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 1.0 | 0.1333 | 0.1667 | 0.1118 |
| Dense/bge-m3 | 0.0 | 0.3533 | 0.4400 | 0.2771 |
| Hybrid | 0.0 | 0.3533 | 0.4400 | 0.2768 |
| Hybrid | 0.1 | 0.3533 | 0.4400 | 0.2788 |
| Hybrid | 0.2 | 0.3600 | 0.4467 | 0.2810 |
| Hybrid | 0.3 | 0.3467 | 0.4200 | 0.2847 |
| Hybrid | 0.4 | 0.3267 | 0.3800 | 0.2779 |
| Hybrid | 0.5 | 0.3067 | 0.3667 | 0.2277 |
| Hybrid | 0.6 | 0.2067 | 0.3467 | 0.1739 |
| Hybrid | 0.7 | 0.1733 | 0.2467 | 0.1514 |
| Hybrid | 0.8 | 0.1400 | 0.1933 | 0.1340 |
| Hybrid | 0.9 | 0.1333 | 0.1733 | 0.1234 |
| Hybrid | 1.0 | 0.1333 | 0.1667 | 0.1126 |

bge-m3 的最佳 Recall@10 配置为：

```text
BM25 权重：0.2
Dense 权重：0.8
Recall@5：0.3600
Recall@10：0.4467
MRR：0.2810
```

bge-m3 的最佳 MRR 配置为：

```text
BM25 权重：0.3
Dense 权重：0.7
Recall@5：0.3467
Recall@10：0.4200
MRR：0.2847
```

与 `BAAI/bge-base-en-v1.5` 的最优 Recall@10 配置相比：

| 模型 | 最优 BM25 权重 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|
| bge-base-en-v1.5 | 0.1 | 0.3733 | 0.4867 | 0.2711 |
| bge-m3 | 0.2 | 0.3600 | 0.4467 | 0.2810 |

从结果看，bge-m3 并没有在 Recall@5 或 Recall@10 上超过 bge-base-en-v1.5。bge-base 的最佳 Recall@10 为 0.4867，而 bge-m3 的最佳 Recall@10 为 0.4467。不过 bge-m3 的 MRR 更高，最佳 MRR 达到 0.2847，高于 bge-base 的 0.2739。这说明 bge-m3 在部分已命中的问题上能把证据页面排得更靠前，但整体覆盖到的 evidence 页面数量少于 bge-base。

这一现象可能与当前实验设置有关：FinanceBench 开源问题和财报文本均为英文，而 `bge-base-en-v1.5` 是英文检索模型，可能更适合该任务；`bge-m3` 的优势更多体现在多语言、多粒度和长文本检索等场景，在当前英文页级 evidence 召回任务中没有转化为更高 Recall。

## 8. 结论

本实验完成了 FinanceBench 财报数据的文本抽取、清洗、分块、BM25 检索、BGE 向量检索、分数归一化融合与权重消融评估，并额外比较了 `BAAI/bge-base-en-v1.5` 与 `BAAI/bge-m3` 两个 Embedding 模型。结果表明，在当前页级 evidence 命中评测下，向量检索是主要有效信号，BM25 适合作为低权重辅助信号。

最终推荐的混合检索参数为 `bm25_weight=0.1`、`dense_weight=0.9`。该配置在 Recall@10 上优于单一 Dense 检索，并远优于单一 BM25 检索，符合本任务中“通过 BM25 + 向量混合检索提升财报相关证据召回率”的目标。

如果以 Recall@10 为主指标，推荐使用：

```text
Embedding 模型：BAAI/bge-base-en-v1.5
BM25 权重：0.1
Dense 权重：0.9
Recall@10：0.4867
```

如果以 MRR 为主指标，bge-m3 的 `bm25_weight=0.3` 配置略优：

```text
Embedding 模型：BAAI/bge-m3
BM25 权重：0.3
Dense 权重：0.7
MRR：0.2847
```

综合本任务“提升相关文档召回率”的目标，最终仍建议采用 `BAAI/bge-base-en-v1.5 + BM25 0.1 线性融合` 作为主配置。

后续可进一步尝试的方向包括：

1. 使用更细粒度的表格解析与表格专用分块策略；
2. 将 BM25 实现改为倒排索引以提升实验速度；
3. 比较 RRF 融合与线性融合；
4. 对不同问题类型分别调参，例如指标型问题可能更依赖关键词和数字匹配；
5. 使用 reranker 在 Top-50 候选上进行二阶段排序。
