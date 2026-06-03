# Follow Top Papers — Figure Reproduction Series

Python 复现 AI 顶会论文经典实验图，配套小红书"跟着顶刊学绘图"专栏教学素材。

每期目录下：
- `reproduce.py` — 单文件 matplotlib 复现脚本，可直接运行
- `extracted_data.json` — 从论文像素校准/表格抄录得到的全部数值数据（reproduce.py 读这个跑）
- `manifest.json` — 元数据清单（论文标题 / 来源 / 3 张图说明）
- `xhs_publish/` — 小红书发布素材：封面 + 3 张原图vs复现对比 + 3 张代码截图 + 色卡

> ep04/ep05 是早期复现，`extracted_data.json` 当时还没引入规范，缺失。

## 期数索引

| 期数 | 论文 | 来源 | 绘图类型 |
|------|------|------|----------|
| 01 | [EMMA: An Enhanced MultiModal Reasoning Benchmark](./ep01_EMMA/) | ICML 2025 | 饼图 / 分组柱形 / 嵌套环形 |
| 02 | [Scaling LLM Test-Time Compute Optimally](./ep02_TestTimeCompute/) | ICLR 2025 | 分组柱形×2 / 折线 |
| 03 | [Scaling Laws for Neural Language Models](./ep03_ScalingLaws/) | — | 对数折线 / 幂律拟合 / 缩放定律 |
| 04 | [WildBench: Benchmarking LLMs with Challenging Tasks from Real Users](./ep04_WildBench/) | ICLR 2025 | 直方图 / 饼图 / 雷达图 |
| 05 | [Towards Precise Scaling Laws for Video Diffusion Transformers](./ep05_VideoScalingLaw/) | CVPR 2025 | 热力图 / 折线图 / 散点图 |
| 06 | [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](./ep06_ChatbotArena/) | ICML 2024 | 水平柱形 / 森林图 / 置信带折线 |
| 07 | [DeepSeek-R1: Incentivizing Reasoning Capability via Reinforcement Learning](./ep07_DeepSeekR1/) | — | 分组柱形 / 多系列折线 / 双轴折线 |
| 08 | [MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark](./ep08_MMLUPro/) | — | 饼图 / 分组柱形 / 误差线 |
| 09 | [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in LLMs](./ep09_GSMSymbolic/) | NeurIPS 2024 | 排序柱形 / 核密度分布 / 多子图分布 |
| 10 | [MixEval: Deriving Wisdom of the Crowd from LLM Benchmark Mixtures](./ep10_MixEval/) | NeurIPS 2024 | 散点回归 / 参考线柱形 / 标注热力图 |
| 11 | [Sycophancy is an Educational Safety Risk: Why LLM Tutors Need Sycophancy Benchmarks](./ep11_Sycophancy/) | arXiv (ICLR 2026 sub) | 分组柱形 / 热力图 / 柱形图 |
| 12 | [MANSU benchmark study](./ep12_MANSU/) | — | 直方图 / 直方图 / 散点图 |
| 13 | [GroupMemBench: Benchmarking LLM Agent Memory in Multi-Party Conversations](./ep13_GroupMemBench/) | arXiv (NeurIPS 2026 sub) | 分组柱形 / 热力图 / 堆叠柱形 |
| 14 | [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](./ep14_LargeLanguageMonkeys/) | NeurIPS 2024 | 多面板折线 / 幂律拟合 / 排序柱形 |
| 15 | [SaaSBench: Benchmarking SaaS workflows](./ep15_SaaSBench/) | arXiv 2605.15777 | 水平柱形 / 分组堆叠 / 100% 堆叠 |
| 16 | [AgentAtlas: Mapping the Landscape of LLM Agents](./ep16_AgentAtlas/) | arXiv 2026.05 | 多系列折线 / 水平堆叠柱形 / 散点连线 |
| 17 | [Coconut: Reasoning in Latent Space for Large Language Models](./ep17_Coconut/) | ICLR 2025 | 柱形图 / 折线图 / 折线图 |
| 18 | [ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning](./ep18_ShieldAgent/) | ICML 2025 | 分组柱形 / 分组柱形 / 多系列折线 |
| 19 | [Tree of Thoughts: Deliberate Problem Solving with LLMs](./ep19_TreeOfThoughts/) | NeurIPS 2023 | 折线图 / 柱形图 / 柱形图 |
| 20 | [Self-Discover: LLMs Self-Compose Reasoning Structures](./ep20_SelfDiscover/) | ICML 2024 | 分组柱形 / 散点图 / 分组柱形 |
| 21 | [Voyager: An Open-Ended Embodied Agent with Large Language Models](./ep21_Voyager/) | arXiv | 折线图 / 折线图 / 折线图 |
| 22 | [RxEval: A Prescription-Level Benchmark for Evaluating LLM Medication Recommendation](./ep22_RxEval/) | arXiv 2026.05 (2605.14543) | 长尾折线 / 柱形图 / 对比直方图 |
| 23 | [AgentKernelArena: Generalization-Aware Benchmarking of GPU Kernel Optimization Agents](./ep23_AgentKernelArena/) | arXiv 2026.05 (2605.16819) | grouped_bar / stacked_bar / scatter |

## License

复现代码 MIT，图表对应论文版权归原作者。
