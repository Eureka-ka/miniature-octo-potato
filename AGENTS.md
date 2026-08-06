# AGENTS.md — 项目说明（中文）

## 一、项目概述
- 赛题：2025 年数学建模大赛 **A 题「数学建模论文智能评估系统与多智能体优化方法」**。
- 目标：在原有工作基础上，优化 **问题1**（综合评价指标体系与自动评分）、补全 **问题2**（质量与可量化文本特征的关联/预测/小样本稳定性），并完成 **问题3**（论文优化策略：AI生成痕迹检测、逻辑断层识别、修改方案、优化后得分预测）。
- 语言：全部中文；实现语言 Python；全程离线可复现（jieba 为可选增强）。

## 二、数据与路径
| 项目 | 路径 |
|---|---|
| 输入数据 | `E:\数学建模大赛\选题D\选题A\附件{1,2,3}`（只读，**不改动原文件**） |
| 输出目录 | `C:\Users\huangxiyan\Documents\选题A\output\` |
| 代码目录 | `C:\Users\huangxiyan\Documents\选题A\code\` |
| GitHub | `https://github.com/Eureka-ka/miniature-octo-potato`（私有，分支 `main`） |

- 附件1：30 篇论文（跨 A/B/C 三题；**25.pdf 为纯图片扫描件，无文字层，已排除**）。
- 附件2：10 篇同赛题论文（B 题；**2-8.pdf 为纯图片扫描件，无文字层，已排除**，有效样本 n=9）。
- 附件3：3 篇论文（A 题康养；题目预设为"中等"，当前模型基线亦为"中等"）。

## 三、目录结构
```
code/
├─ common.py            # 共享：文本提取、AHP、隶属度校准、PIL图表、分位定级
├─ quantify.py          # 问题1：16个二级指标逐一专属量化（jieba可选）
├─ problem1.py          # 问题1：AHP+模糊综合评价，五级分级
├─ problem2_features.py # 问题2：12项可量化特征 + 质量得分
├─ problem2_stats.py    # 问题2：GRA/相关/回归/调整因子/小样本稳定性
├─ problem3.py          # 问题3：基线/AI痕迹/逻辑断层/修改方案/优化预测
├─ build_report.py      # 生成中文 Word 报告
└─ run_all.py           # 一键全流程
output/
├─ problem1_results.xlsx / problem1_indicators.csv / problem1_minmax.json
├─ problem2_features.xlsx|csv / problem2_stats.xlsx / problem2_papers.json
├─ problem3_results.xlsx
├─ figs/*.png
└─ 选题A_建模报告.docx / .pdf
```

## 四、运行方式
- 解释器：Codex 捆绑 Python（`C:\Users\huangxiyan\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`）。
- 依赖：`pdfplumber pandas numpy openpyxl python-docx PIL reportlab`；`jieba`（可选，`python -m pip install jieba`，缺失时 quantify.py 自动降级）。
- 一键运行：`python code/run_all.py`（依次：问题1→问题2→问题3→生成报告）。
- 单模块运行：`python -c "import sys; sys.path.insert(0,'code'); import problem1; problem1.run()"`。
- 报告 docx→pdf：需要启动 Microsoft Word（COM 导出），非代码内完成。

## 五、核心方法
### 问题1（problem1.py + quantify.py）
- 指标体系：4 个一级维度（U1 逻辑严密性 / U2 方法合理性 / U3 公式推导规范性 / U4 结构与文本规范）× 16 个二级指标。
- AHP 权重：来自 `AHP_A.m` 的 5 个专家判断矩阵（`common.JUDGMENT_MATRICES`），算术平均法+特征值法取均值，全部 **CR<0.1**。
- 指标量化：`quantify.py` 中 16 个专属方法（分节覆盖、结果-验证闭环、四类断层规则、模型词典匹配、行级公式检测、符号说明表、行首章节检测、重复句占比等）。
- 归一化：`common.apply_refs`——ratio 达标线 0.8；density / low 取样本 p80；count 取 max(8, p80)。
- 隶属度：`common.calibrate_centers`——分位中心 5/30/50/70/95；近常数指标（p90−p10<0.15）用固定中心 [0.15,0.30,0.50,0.70,0.85]。
- 评分：`FCEScorer` → B=A·R，综合得分 Q=B·[90,75,60,45,25]。
- 分级：**相对分位定级** `percentile_bands`（85/65/40/20 分位；本次阈值：优秀≥73.0、良好≥70.3、中等≥65.9、及格≥61.7，以下为不及格）。
- 当前结果（29 篇自动评分）：**优秀5 / 良好5 / 中等7 / 及格6 / 不及格6**（得分区间 55.9~78.4）；25.pdf 标记图像型排除。

### 问题2（problem2_features.py + problem2_stats.py）
- 12 项可量化特征（在剔除代码附录页后的正文上计算：内容页数/内容字符数、行首标题章节、行级公式、逐式编号、引用-条目）→ Z-score标准化 → 4 个综合维度（篇幅/公式/逻辑连接/参考文献）；质量得分沿用问题1评分模型（n=9）。
- 分析：灰色关联度(GRA)、Pearson/Spearman+Bootstrap95%CI+置换检验、关键特征识别、岭回归(λ=1.0)+OLS、LOO-CV、Bootstrap(1000次)、单样本剔除敏感性。
- 质量调整因子：k=1+λ·(F−F̄)/F̄（λ=0.30），基于特征剖面相对位置校准基础得分，不依赖回归外推能力。
- 关键结论：可量化文本特征与质量得分的关联整体较弱且方向不一（篇幅类弱正相关、连词/公式规范类弱负相关）；小样本下模型定位为"关联识别与方向判断"，不宜直接打分。

### 问题3（problem3.py）
- 基线：3-1/3-2/3-3 = 59.12 / 60.97 / 59.04（均"中等"，与题目预设一致）。
- AI 痕迹检测：6 个离线统计特征（句长/段落均匀度、套话密度、连接词密度、低具体性、重复率），权重 0.20/0.15/0.20/0.10/0.20/0.15；分档 <0.40 低、0.40~0.55 中、≥0.55 高。
- 逻辑断层识别：规则扫描（公式化不足、缺假设、结论未验证、缺灵敏度/稳健性、摘要要素不全、参考文献不足、公式编号/符号说明缺失）。
- 修改方案：薄弱指标（s<0.5）+ 检测问题 → `TEMPLATE` 建议；增益 `uplift_for`（<0.3→+0.25、<0.5→+0.18、<0.7→+0.10、否则+0.04）。
- 优化后得分预测：提升对应指标 s 后重跑评分模型；当前预测 61.39 / 64.43 / 61.82（仍为中等）。

## 六、关键可调参数（改这里即可调结果）
- `common.py`：`GRADE_SCORES`、`level_by_score` / `percentile_bands`、`JUDGMENT_MATRICES`（AHP判断矩阵）、各关键词词典、`calibrate_centers` 分位与常数阈值(0.15)、`compute_refs` 达标线。
- `quantify.py`：`KIND`（每个指标的归一方式）、各 `q_uXX` 量化函数、`AI_WEIGHTS`/`AI_LEVELS`、`uplift_for`、`TEMPLATE`。
- `problem2_stats.py`：`KEY_FEATURES`（预测模型自变量，当前=["内容字符数"]）、`LAMBDA`（岭回归正则）、`LAMBDA_F`（调整因子校准强度，当前0.30）。
- 随机种子：固定（Bootstrap seed=42、置换 seed=7），结果可复现。

## 七、已知限制
- 25.pdf、2-8.pdf 为纯图片扫描件（无文字层），不参与自动提取与统计建模；脚本预留人工补录入口。
- AI 生成痕迹检测为离线文本统计启发式，非大语言模型困惑度检测，结果仅供参考。
- 问题3的"优化后得分"为基于修改建议的模型模拟预测，非论文正文重写。
- 图表用 openpyxl 原生图表 + PIL 手绘（未安装 matplotlib）。

## 八、工作约定
- 所有源文件以 UTF-8（无 BOM）写入；PowerShell 管道传中文需先 `$OutputEncoding=[Text.Encoding]::UTF8` 且设 `PYTHONIOENCODING=utf-8`。
- 保持离线：不联网、不额外装库（jieba 已装，可选）。
- 沙箱中 `.git` 只读，git 写操作（add/commit/push）需提权。
- 每次修改后：`python code/run_all.py` 全量重跑，重新生成报告并（如需）用 Word 转 PDF 校验。