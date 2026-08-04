# -*- coding: utf-8 -*-
"""生成完整中文 Word 报告：覆盖问题1/2/3 的模型、结果、图表与分析。"""
import os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import common as C
import problem1

OUT = C.OUT
FIGS = C.FIGS
REPORT = os.path.join(OUT, "选题A_建模报告.docx")

DARK = RGBColor(0x1F, 0x3B, 0x64)
ACCENT = RGBColor(0x44, 0x72, 0xC4)
GRAY = RGBColor(0x66, 0x66, 0x66)

def cn(run, name="宋体", size=10.5, bold=False, color=None, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if color is not None:
        run.font.color.rgb = color

def para(doc, text, size=10.5, bold=False, align=None, space_after=6, color=None, indent=True):
    p = doc.add_paragraph()
    if align: p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Pt(size * 2)
    r = p.add_run(text)
    cn(r, size=size, bold=bold, color=color)
    return p

def heading(doc, text, level=1):
    h = doc.add_heading("", level=level)
    r = h.add_run(text)
    cn(r, name="黑体", size=(16 if level == 1 else 13 if level == 2 else 11.5),
       bold=True, color=(DARK if level == 1 else ACCENT))
    h.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    h.paragraph_format.space_after = Pt(6)
    return h

def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    cn(r, name="黑体", size=9.5, bold=True, color=GRAY)
    return p

def figure(doc, path, cap, width=5.6):
    if os.path.exists(path):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(width))
        caption(doc, cap)
    else:
        para(doc, f"[缺图] {cap}", size=9, color=GRAY)

def add_table(doc, df, caption_text, font_size=8.5, max_rows=None, transpose=False,
              col_widths=None, highlight_last_col=True):
    if transpose:
        df = df.T.reset_index()
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
    n_rows, n_cols = df.shape
    t = doc.add_table(rows=n_rows + 1, cols=n_cols)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, col in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(col))
        cn(r, name="黑体", size=font_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        shd = cell._tc.get_or_add_tcPr().makeelement(qn("w:shd"), {qn("w:val"): "clear", qn("w:fill"): "4472C4"})
        cell._tc.get_or_add_tcPr().append(shd)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            r = cell.paragraphs[0].add_run("" if v is None else str(v))
            cn(r, size=font_size)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption(doc, caption_text)
    return t

# ---------------- 数据加载 ----------------
def load_p1():
    calib = C.load_json(os.path.join(OUT, "problem1_minmax.json"))
    df = pd.read_csv(os.path.join(OUT, "problem1_indicators.csv"), encoding="utf-8-sig")
    return calib, df

def load_p2_features():
    return pd.read_csv(os.path.join(OUT, "problem2_features.csv"), encoding="utf-8-sig")

def load_p2_stats():
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(OUT, "problem2_stats.xlsx"), data_only=True)
    def rd(sheet):
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        return pd.DataFrame(rows[1:], columns=rows[0])
    return {s: rd(s) for s in ["灰色关联度-维度", "灰色关联度-原始特征", "相关性分析",
                                "回归模型", "LOO交叉验证", "Bootstrap系数", "敏感性-单样本剔除",
                                "质量调整因子"]}

def load_p3():
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(OUT, "problem3_results.xlsx"), data_only=True)
    def rd(sheet):
        ws = wb[sheet]
        rows = list(ws.iter_rows(values_only=True))
        return pd.DataFrame(rows[1:], columns=rows[0])
    return {s: rd(s) for s in ["基线评分", "AI痕迹检测", "逻辑断层与问题定位",
                                "具体修改方案", "优化前后对比", "边际贡献"]}

# ---------------- 报告 ----------------
def main():
    calib, p1df = load_p1()
    p2f = load_p2_features()
    p2s = load_p2_stats()
    p3 = load_p3()

    doc = Document()
    # 页面边距
    for sec in doc.sections:
        sec.left_margin = Cm(2.5); sec.right_margin = Cm(2.5)
        sec.top_margin = Cm(2.5); sec.bottom_margin = Cm(2.5)

    # ===== 标题 =====
    tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = tp.add_run("A题 数学建模论文智能评估系统与多智能体优化方法")
    cn(r, name="黑体", size=18, bold=True, color=DARK)
    sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = sp.add_run("——问题1/2优化建模与问题3优化策略建模报告")
    cn(r, name="黑体", size=14, bold=True, color=ACCENT)
    doc.add_paragraph()

    # ===== 摘要 =====
    heading(doc, "摘要", 1)
    para(doc, "本报告针对“数学建模论文智能评估系统与多智能体优化方法”赛题，在已有问题1、问题2工作基础上进行系统优化，并完成问题3。"
              "问题1沿用“层次分析法(AHP)+模糊综合评价(FCE)”骨架，将“逻辑严密性、方法合理性、公式推导规范性、结构与文本规范”4个一级维度拆解为16个可量化二级指标，"
              "权重由专家判断矩阵复算并全部通过一致性检验(CR<0.1)，隶属度采用连续函数替代原5档硬编码，分级改用综合得分法；对附件1中30篇论文评分分级，"
              "其中29篇可自动识别（1篇为纯图片扫描件，已标记排除），结果集中于良好(16篇)与中等(12篇)。")
    para(doc, "问题2在特征提取—极差归一—灰色关联分析的基础上，补充了相关分析、关键特征识别、质量调整因子、质量预测模型与系统的小样本稳定性分析。"
              "结果表明：可量化特征与质量得分的关联整体较弱，逻辑连词总频次、规范编号公式占比、参考文献数量为相对关键的正向特征；"
              "在9篇小样本下留一交叉验证R²<0，说明特征预测模型外推能力不足，据此设计可靠性加权的质量调整因子并保守取k≈1。")
    para(doc, "问题3基于问题1评分模型与问题2关键特征，构建了包含AI生成痕迹检测（离线文本统计特征）、逻辑断层识别、逐篇修改方案与优化后得分预测的完整优化策略。"
              "对附件3中3篇论文的评估显示：3-1、3-3结构完整但基线已达优秀区间，3-2为良好；三篇AI辅助指数均较低；"
              "针对其薄弱指标（公式标注规范率、假设适配度、论证闭环完整率等）给出修改方案后，预测得分分别提升至82.81、70.84、84.57。")
    para(doc, "关键词：模糊综合评价；层次分析法；灰色关联分析；质量预测；小样本稳定性；AI生成痕迹检测；论文优化策略", bold=False)

    # ===== 一、问题重述 =====
    heading(doc, "一、问题重述与背景", 1)
    para(doc, "数学建模竞赛论文具有结构标准化、逻辑严密、符号规范等特殊要求，其质量评估涉及多维度复杂判断。"
              "随着大语言模型与智能写作工具的普及，自动识别论文逻辑缺陷、评估建模质量并提供优化建议的智能系统成为教育评价领域的重要研究方向。")
    para(doc, "问题1：基于附件1的30篇参赛论文，建立论文质量综合评价指标体系，将核心维度拆解为可量化二级指标，构建自动评分模型并进行五级质量分级，"
              "说明指标体系规范依据与权重设定合理性。")
    para(doc, "问题2：基于附件2的10篇同赛题论文，建立统计模型分析论文质量与可量化文本特征（篇幅结构、公式密度、逻辑连接词、参考文献规范性等）的关联，"
              "识别关键特征，引入论文质量调整因子，建立基于关键特征的质量预测模型，并分析小样本条件下模型的稳定性。")
    para(doc, "问题3：结合问题1评分模型与问题2关键特征，设计论文优化策略（含AI生成痕迹检测、逻辑断层识别与修正），"
              "针对附件3中3篇“中等”质量论文给出具体修改方案、AI辅助程度评估及优化后的质量得分预测。")

    # ===== 二、问题分析 =====
    heading(doc, "二、总体思路", 1)
    para(doc, "三个问题构成“评价—关联—优化”闭环：问题1建立可解释的评分模型（AHP+FCE）并输出16个二级指标得分；"
              "问题2从文本特征角度揭示质量的关键驱动因素并给出预测与稳定性结论；问题3以问题1的薄弱指标与问题2的关键特征为输入，"
              "生成针对性修改方案并模拟优化后的质量得分。全流程使用Python实现，输入数据读取自附件目录，输出统一写入output/目录，全程离线可复现。")

    # ===== 三、模型假设与符号说明 =====
    heading(doc, "三、模型假设与符号说明", 1)
    para(doc, "假设：(1) 论文PDF文本层可完整提取，关键词统计能反映论文的结构与表达特征；(2) 评委对同一篇论文的评价虽存在主观差异，但可通过多方法融合降低偏差；"
              "(3) 图像型论文（无文字层）不参与自动特征提取，需人工复核。")
    para(doc, "主要符号：U1~U4为一级指标（逻辑严密性、方法合理性、公式推导规范性、结构与文本规范）；u11~u44为16个二级指标；"
              "A为一级权重向量，A1~A4为二级权重向量；R为模糊评判矩阵；B为综合隶属度向量；Q为综合质量得分。")

    # ===== 四、问题1 =====
    heading(doc, "四、问题1：综合评价指标体系与自动评分模型", 1)

    heading(doc, "4.1 指标体系构建依据", 2)
    para(doc, "指标体系以全国大学生数学建模竞赛评阅要点（摘要与问题重述、模型假设、建模与求解、结果分析、模型检验与推广、写作规范）和题目给定的核心维度"
              "（“逻辑严密性”“方法合理性”）为规范依据，构建“逻辑严密性U1、方法合理性U2、公式推导规范性U3、结构与文本规范U4”4个一级维度，"
              "并进一步拆解为16个可量化二级指标（逻辑连接词密度、赛题呼应覆盖率、论证闭环完整率、逻辑断层条数；模型匹配度、假设适配度、多方案对比完备度、"
              "方法创新量化值；有效公式密度、公式标注规范率、推导步骤完整度、符号统一度；标准章节完整率、摘要要素完整度、参考文献规范率、文本冗余度）。")

    heading(doc, "4.2 AHP权重确定与一致性检验", 2)
    para(doc, "采用AHP确定权重：由专家对一级指标及各级二级指标的重要性两两比较得到判断矩阵（附录A），"
              "分别用算术平均法与特征值法求权重并取均值，计算一致性比例CR。下表给出权重与一致性结果，全部CR<0.1，一致性可接受。")
    wrows = []
    for k, v in calib["weights"].items():
        wrows.append([k, "[" + ", ".join(f"{x:.4f}" for x in v["w"]) + "]",
                      f"{v['lam']:.4f}", f"{v['CI']:.4f}", f"{v['CR']:.4f}"])
    add_table(doc, pd.DataFrame(wrows, columns=["判断矩阵", "权重向量", "λmax", "CI", "CR"]),
              "表1 AHP判断矩阵权重与一致性检验结果", font_size=9)

    heading(doc, "4.3 二级指标量化与隶属度函数", 2)
    para(doc, "每个二级指标给出可计算原始值：覆盖率类指标采用关键词命中率（描述性词典覆盖率达标线取60%），密度类指标采用“次数/总字符数”（以参照样本80分位为达标线），"
              "数量类指标（参考文献条数）以8条为达标基线，反向指标（逻辑断层、文本冗余）取补。原始值经参照锚定映射为[0,1]绝对得分s。")
    para(doc, "隶属度采用连续函数：以参照样本（29篇可识别论文）各指标得分的5%/30%/50%/70%/95%分位作为“不及格—及格—中等—良好—优秀”五个等级的中心，"
              "常数指标按绝对水平固定中心，由得分s线性插值得到五级隶属度向量（归一化后和为1），从而替代原方案中5档硬编码向量，消除结果扎堆与边界敏感问题。")

    heading(doc, "4.4 综合评价与分级", 2)
    para(doc, "对论文第j个维度，由该维度4个指标的隶属度向量构成评判矩阵Rj，得Bj=Aj·Rj；令R=[B1,B2,B3,B4]ᵀ，综合隶属度B=A·R。"
              "综合质量得分Q=B·[90,75,60,45,25]ᵀ，按阈值Q≥80为优秀、65~80为良好、50~65为中等、35~50为及格、<35为不及格进行分级。"
              "分级采用综合得分法而非最大隶属度法，避免边界样本误判。")

    heading(doc, "4.5 结果与分析", 2)
    p1sum = p1df[["论文名称", "赛题类型", "综合得分", "等级"]].copy()
    add_table(doc, p1sum, "表2 附件1三十篇论文评价结果（综合得分与等级）", font_size=8.5)
    para(doc, "30篇论文中，25.pdf为纯图片扫描件（无文字层），已标记为“图像型论文”并排除出自动评分；其余29篇自动识别完成。"
              "等级分布为：良好16篇、中等12篇、及格1篇（12.pdf，得分44.37），无论文达到80分优秀阈值。"
              "最高分03.pdf（78.03）在内容与结构指标上接近优秀，但其参考文献缺失（引用条数为0）与逻辑连接词密度偏低成为主要短板；"
              "该结果与“本批次论文整体质量集中于中等—良好、普遍存在参考文献与稳定性分析薄弱”的人工观感一致。")
    figure(doc, os.path.join(FIGS, "p1_grade_dist.png"), "图1 问题1三十篇论文质量等级分布", 4.6)
    figure(doc, os.path.join(FIGS, "p1_dim_compare.png"), "图2 代表论文各维度得分对比", 5.0)
    figure(doc, os.path.join(FIGS, "p1_radar_top_bottom.png"), "图3 最优与最差论文二级指标雷达图", 4.6)

    # ===== 五、问题2 =====
    heading(doc, "五、问题2：质量与可量化文本特征的关联及预测模型", 1)

    heading(doc, "5.1 特征体系与质量得分", 2)
    para(doc, "沿用“篇幅结构、公式、逻辑连接、参考文献”四个综合维度，共12项可量化特征：总页数、总字符数、章节完整度、段落平均长度（反向）、公式总数、"
              "公式密度、规范编号公式占比、逻辑连词总频次、连词密度、参考文献数量。各特征经极差归一（反向指标取补）后按维度等权合成综合得分。"
              "论文质量得分沿用问题1评分模型（同一套AHP权重与校准参数），2-8.pdf为图像型论文，予以排除，有效样本n=9。")
    p2show = p2f[["论文名称", "总页数", "总字符数", "公式总数", "规范编号公式占比", "逻辑连词总频次",
                  "参考文献数量", "篇幅结构特征综合得分", "公式特征综合得分", "逻辑连接特征综合得分",
                  "参考文献特征综合得分", "综合质量得分", "等级"]].copy()
    add_table(doc, p2show, "表3 附件2论文可量化特征与质量得分（节选关键列）", font_size=8)

    heading(doc, "5.2 灰色关联分析", 2)
    gd = p2s["灰色关联度-维度"]
    add_table(doc, gd, "表4 综合维度与质量得分的灰色关联度", font_size=9)
    para(doc, "四个综合维度中，篇幅结构(0.592)、逻辑连接(0.572)、参考文献(0.573)与质量得分的灰色关联度接近，公式特征(0.535)略低，"
              "说明篇幅与逻辑表达对质量的影响相对更直接。")
    figure(doc, os.path.join(FIGS, "p2_gra_dims.png"), "图4 综合维度灰色关联度", 4.6)

    heading(doc, "5.3 相关分析与关键特征识别", 2)
    corr = p2s["相关性分析"]
    corr_top = corr.reindex(corr["Pearson_r"].abs().sort_values(ascending=False).index).head(6)
    add_table(doc, corr_top[["特征", "Pearson_r", "Spearman_rho", "Bootstrap_CI下限", "Bootstrap_CI上限", "置换检验p", "GRA"]],
              "表5 相关分析（|r|前6）", font_size=8.5)
    para(doc, "章节完整度虽GRA较高（0.754），但其取值在该9篇中近乎恒定（仅4/6与5/6两档），负相关主要由个别样本驱动，不视为关键特征。"
              "结合GRA、相关方向与语义，确定关键特征为：逻辑连词总频次（GRA 0.630，r=0.385）、规范编号公式占比（GRA 0.571，r=0.390）、"
              "参考文献数量（GRA 0.594，r=0.191），三者分别对应逻辑连接、公式规范与参考文献规范性。")
    figure(doc, os.path.join(FIGS, "p2_corr.png"), "图5 特征与质量得分的Pearson相关(±95%CI)", 4.8)

    heading(doc, "5.4 质量预测模型", 2)
    reg = p2s["回归模型"]
    add_table(doc, reg, "表6 岭回归预测模型（标准化系数/原始尺度系数与精度）", font_size=8.5)
    para(doc, "以三个关键特征为自变量、质量得分为因变量建立标准化线性回归，并引入岭回归（λ=1.0）抑制多重共线性。"
              "样本内拟合R²=0.24，但留一交叉验证R²<0、RMSE约12.8分，说明在9篇小样本下特征预测模型的外推能力不足——"
              "这正是小样本稳定性分析的结论：模型可解释关联方向，但不宜直接用于打分，需以保守方式引入调整因子。")

    heading(doc, "5.5 质量调整因子", 2)
    para(doc, "为兼顾评阅主观差异与数据视角，定义质量调整因子：k_i=1+α·w·(ŷ_i−Q_i)/Q_i+β·(F_i−F̄)/F̄。"
              "其中α=0.3为回归调整收缩系数，w=clip(R²_LOO/0.5,0,1)为模型外推可靠性权重，β=0.1为特征剖面分量，"
              "F_i为论文特征剖面与理想剖面的灰色关联度。当R²_LOO<0（本数据集w=0）时回归分量不启用，仅保留温和的特征剖面校正，"
              "避免引入不可靠的预测噪声。调整后得分Q_adj=Q·k。")
    adj = p2s["质量调整因子"]
    add_table(doc, adj, "表7 质量调整因子与调整后得分", font_size=8.5)
    figure(doc, os.path.join(FIGS, "p2_adjust.png"), "图6 调整因子作用前后对比", 5.0)

    heading(doc, "5.6 小样本稳定性分析", 2)
    para(doc, "稳定性从三方面量化：(1) 留一交叉验证：9次“留一篇、训八篇”外推，岭回归R²<0、RMSE≈12.8，OLS更差，表明样本量过小导致外推不稳定；"
              "(2) Bootstrap(1000次)：系数分布均值与95%置信区间较宽，反映估计不确定；"
              "(3) 单样本剔除敏感性：剔除单篇后系数最大相对变化达数十个百分点，进一步印证小样本敏感性。"
              "这些结果共同说明：在n=9的小样本下，应谨慎使用特征回归结果，模型宜定位为“关联识别与方向判断”，而非精确打分。")
    loo = p2s["LOO交叉验证"]
    add_table(doc, loo, "表8 留一交叉验证：实际与预测质量得分", font_size=8.5)
    bs = p2s["Bootstrap系数"]
    add_table(doc, bs, "表9 Bootstrap系数分布（岭回归，1000次）", font_size=8.5)
    figure(doc, os.path.join(FIGS, "p2_loo.png"), "图7 留一交叉验证：实际 vs 预测", 4.8)
    figure(doc, os.path.join(FIGS, "p2_coef_bootstrap.png"), "图8 关键特征标准化系数(Bootstrap±95%CI)", 4.8)

    # ===== 六、问题3 =====
    heading(doc, "六、问题3：论文优化策略与预测", 1)

    heading(doc, "6.1 基线评估", 2)
    base = p3["基线评分"][["论文", "综合得分", "等级", "AI辅助指数", "AI辅助程度"]].copy()
    add_table(doc, base, "表10 附件3三篇论文基线评估", font_size=9)
    para(doc, "按问题1模型，三篇论文基线得分为：3-1 81.54（优秀）、3-2 66.11（良好）、3-3 81.01（优秀）。"
              "与题目“中等”预设存在差异：本模型侧重结构与文本特征，而三篇论文篇幅完整、结构规范、关键词覆盖充分，故在结构类指标上得分高；"
              "题目“中等”更可能基于人工评阅对建模深度与结果可信度的综合判断，这正是关键词特征模型的局限。"
              "本节仍按题目要求将三篇作为优化对象，聚焦其相对薄弱指标给出修改方案。")
    figure(doc, os.path.join(FIGS, "p3_radar.png"), "图9 三篇论文二级指标得分雷达图", 4.6)

    heading(doc, "6.2 AI生成痕迹检测（离线文本统计特征法）", 2)
    para(doc, "构建6个离线统计特征：句长均匀度、段落均匀度、套话密度（“综上所述、本研究、值得注意的是”等）、连接词密度、"
              "低具体性（数字密度取补）与重复率，加权（0.20/0.15/0.20/0.10/0.20/0.15）合成AI辅助指数，映射为低/中/高三档。"
              "该方法无需外部API，可完全复现；其局限在于仅为统计启发式，不能等同于大语言模型的困惑度检测，结果仅供参考。")
    ai = p3["AI痕迹检测"][["论文", "句长均匀度", "段落均匀度", "套话密度", "连接词密度", "低具体性", "重复率",
                           "AI辅助指数", "辅助程度"]].copy()
    add_table(doc, ai, "表11 AI生成痕迹检测结果", font_size=8.5)
    para(doc, "三篇论文句长与段落长度变异系数均较高（约0.70/0.67），数字密度高（48~61‰），整体更接近人工写作风格；"
              "3-1的套话密度偏高（1.90‰），AI辅助指数0.285为三篇最高，但均未达到“中/高”档。结论：三篇论文的AI生成痕迹整体较低。")
    figure(doc, os.path.join(FIGS, "p3_ai_index.png"), "图10 三篇论文AI辅助指数", 4.4)

    heading(doc, "6.3 逻辑断层识别", 2)
    para(doc, "通过规则扫描识别逻辑断层：模型以文字描述为主而公式化不足、未显式给出假设、结论未回扣结果验证、"
              "缺少灵敏度/稳健性/收敛性分析、摘要“结果—结论”要素不全、参考文献不足、关键公式缺少编号或符号说明。")
    iss = p3["逻辑断层与问题定位"]
    add_table(doc, iss, "表12 逻辑断层与问题定位", font_size=8.5)

    heading(doc, "6.4 具体修改方案", 2)
    para(doc, "针对每篇论文得分低于0.5的薄弱指标及检测问题，给出按章节组织的具体修改建议（详见附录C与output/problem3_results.xlsx）。")
    plan = p3["具体修改方案"][["论文", "薄弱指标", "当前得分", "问题定位", "具体修改建议", "预计改进后得分"]].copy()
    add_table(doc, plan, "表13 具体修改方案（节选）", font_size=8)

    heading(doc, "6.5 优化后质量得分预测", 2)
    para(doc, "按“薄弱指标得分越低、可改进空间越大”的原则设定增益（<0.30→+0.25，<0.50→+0.18，<0.70→+0.10，否则+0.04），"
              "模拟修改后的指标得分并重跑问题1评分模型，得到优化前后对比与各指标边际贡献。")
    oa = p3["优化前后对比"]
    add_table(doc, oa, "表14 优化前后质量得分预测对比", font_size=9)
    figure(doc, os.path.join(FIGS, "p3_before_after.png"), "图11 优化前后得分对比", 4.8)
    mg = p3["边际贡献"]
    add_table(doc, mg, "表15 各指标改进的边际得分贡献", font_size=8.5)

    # ===== 七、模型评价 =====
    heading(doc, "七、模型评价与灵敏度分析", 1)
    para(doc, "优点：(1) 指标体系具有明确规范依据，权重来自通过一致性检验的专家判断矩阵，可解释性强；(2) 连续隶属度与综合得分法提升分级稳健性；"
              "(3) 问题2给出完整的相关—GRA—回归—稳定性链条，并诚实报告小样本下模型外推能力不足；(4) 问题3将评分模型与优化策略闭环衔接。")
    para(doc, "不足：(1) 关键词统计对语义深度、创新性等维度捕捉有限；(2) 图像型论文无法自动处理；(3) n=9时回归与调整因子置信度有限；"
              "(4) AI痕迹检测为统计启发式，非真实模型困惑度。")
    para(doc, "灵敏度：分级阈值与等级分值、隶属度中心分位、关键词词典与达标线均可能影响结果。本文采用参照样本分位校准并公开全部参数（constants in code），"
              "便于复核；对阈值±5分的扰动，等级总体保持稳定（良好/中等两档边界论文除外），表明模型对参数扰动有一定鲁棒性。")

    # ===== 八、结论 =====
    heading(doc, "八、结论", 1)
    para(doc, "本报告完成了问题1评分模型的优化（指标依据、权重与CR、连续隶属度、综合得分分级，30篇论文分级结果良好16/中等12/及格1/图像型1），"
              "补全了问题2的关联分析、关键特征识别、质量调整因子、预测模型与小样本稳定性分析，"
              "并构建了问题3的“基线评估—AI痕迹检测—逻辑断层识别—修改方案—优化后得分预测”完整优化策略。"
              "全流程离线可复现，代码与数据结果见code/与output/目录。")

    # ===== 附录A =====
    doc.add_page_break()
    heading(doc, "附录A：判断矩阵与权重计算", 1)
    import common as Cc
    for k, M in Cc.JUDGMENT_MATRICES.items():
        w = calib["weights"][k]
        mat = pd.DataFrame(M, index=["i1", "i2", "i3", "i4"], columns=["j1", "j2", "j3", "j4"])
        para(doc, f"{k}：权重=({', '.join(f'{x:.4f}' for x in w['w'])}), λmax={w['lam']:.4f}, CR={w['CR']:.4f}", size=9.5)
        add_table(doc, mat, "", font_size=8)

    # ===== 附录B =====
    heading(doc, "附录B：问题2特征数据与统计结果", 1)
    add_table(doc, p2f.drop(columns=["数据来源", "赛题类型"]), "表B1 附件2论文12项特征、综合维度与质量得分", font_size=7.5)
    gra_raw = p2s["灰色关联度-原始特征"]
    add_table(doc, gra_raw, "表B2 原始特征灰色关联度", font_size=8.5)
    sens = p2s["敏感性-单样本剔除"]
    add_table(doc, sens, "表B3 单样本剔除敏感性（岭回归系数相对变化%）", font_size=8)

    # ===== 附录C =====
    heading(doc, "附录C：问题3修改方案明细", 1)
    add_table(doc, plan, "表C1 三篇论文具体修改方案明细", font_size=7.5)

    doc.save(REPORT)
    print("报告已生成:", REPORT)
    return REPORT

if __name__ == "__main__":
    main()