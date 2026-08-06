# -*- coding: utf-8 -*-
"""选题A 共享工具模块
包含：路径配置、PDF文本提取、关键词词典、指标原始值计算、
AHP权重计算(判断矩阵来自 AHP_A.m)、连续隶属度函数、极差归一化、PIL图表绘制。
"""
import os
import re
import json
import math
import numpy as np
import pdfplumber
from PIL import Image, ImageDraw, ImageFont

# ---------------- 路径 ----------------
BASE = r"E:\数学建模大赛\选题D\选题A"
ATT1 = os.path.join(BASE, "附件1")
ATT2 = os.path.join(BASE, "附件2")
ATT3 = os.path.join(BASE, "附件3")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
OUT = os.path.join(REPO, "output")
FIGS = os.path.join(OUT, "figs")
for _d in (OUT, FIGS):
    os.makedirs(_d, exist_ok=True)

# ---------------- 等级与分值 ----------------
GRADE_NAMES = ["不及格", "及格", "中等", "良好", "优秀"]
GRADE_SCORES = np.array([25.0, 45.0, 60.0, 75.0, 90.0])
# 综合得分 -> 等级 阈值（沿用用户方案，报告中有论证）
def level_by_score(s):
    if s >= 80: return "优秀"
    if s >= 65: return "良好"
    if s >= 50: return "中等"
    if s >= 35: return "及格"
    return "不及格"

def percentile_bands(scores, pcts=(85, 65, 40, 20)):
    """按批次得分分位确定五级阈值（相对定级）：
    返回 {'优秀':t85, '良好':t65, '中等':t40, '及格':t20}，score>=t 为该级及以上。"""
    arr = np.asarray(scores, dtype=float)
    t85, t65, t40, t20 = np.percentile(arr, list(pcts))
    return {"优秀": float(t85), "良好": float(t65), "中等": float(t40), "及格": float(t20)}

def level_by_band(score, bands):
    if score >= bands["优秀"]: return "优秀"
    if score >= bands["良好"]: return "良好"
    if score >= bands["中等"]: return "中等"
    if score >= bands["及格"]: return "及格"
    return "不及格"

# ---------------- AHP（判断矩阵来自 AHP_A.m）----------------
JUDGMENT_MATRICES = {
    "M(一级)": np.array([[1, 1/2, 2, 3],
                          [2, 1, 3, 4],
                          [1/2, 1/3, 1, 3/2],
                          [1/3, 1/4, 2/3, 1]]),
    "M1(U1逻辑严密性)": np.array([[1, 1/3, 1/3, 2],
                                   [3, 1, 1, 4],
                                   [3, 1, 1, 4],
                                   [1/2, 1/4, 1/4, 1]]),
    "M2(U2方法合理性)": np.array([[1, 2, 3, 4],
                                   [1/2, 1, 2, 3],
                                   [1/3, 1/2, 1, 2],
                                   [1/4, 1/3, 1/2, 1]]),
    "M3(U3公式推导规范性)": np.array([[1, 1/2, 1/4, 1/2],
                                       [2, 1, 1/3, 1],
                                       [4, 3, 1, 3],
                                       [2, 1, 1/3, 1]]),
    "M4(U4结构与文本规范)": np.array([[1, 1, 3, 2],
                                       [1, 1, 3, 2],
                                       [1/3, 1/3, 1, 1/2],
                                       [1/2, 1/2, 2, 1]]),
}
_RI = {1: 0.0, 2: 0.0, 3: 0.52, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36,
       8: 1.41, 9: 1.46, 10: 1.49, 11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59}

def ahp(M):
    """算术平均法 + 特征值法取均值求权重，返回权重与一致性指标。"""
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    w1 = (M / M.sum(axis=0)).mean(axis=1)                      # 算术平均
    vals, vecs = np.linalg.eig(M)
    i = int(np.argmax(vals.real))
    lam = vals.real[i]
    v = np.abs(vecs[:, i].real)
    w2 = v / v.sum()                                           # 特征值法
    w = (w1 + w2) / 2.0
    CI = (lam - n) / (n - 1)
    CR = CI / _RI[n] if _RI.get(n) else 0.0
    return {"w": w, "w1": w1, "w2": w2, "lam": float(lam), "CI": float(CI), "CR": float(CR), "n": n}

def compute_all_weights():
    out = {}
    for k, M in JUDGMENT_MATRICES.items():
        out[k] = ahp(M)
    return out

# ---------------- 文本提取 ----------------
def extract_pdf(path):
    """返回 (pages列表, 全文)"""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                pages.append(p.extract_text() or "")
    except Exception:
        pages = []
    full = "\n".join(pages)
    return pages, full

def is_image_only(full):
    """纯图片扫描件（无文字层）判定：有效字符极少。"""
    chars = len(re.sub(r"\s+", "", full))
    return chars < 300

def clean_text(full):
    return re.sub(r"\s+", "", full).lower()

# ---------------- 关键词词典（在原方案基础上扩充） ----------------
CONNECTORS = ["因此", "故而", "由此可得", "综上", "进而", "所以", "可得", "结果表明",
              "从而", "于是", "综上所述", "因而", "故", "那么", "据此", "由此", "基于此"]
CLOSURE = ["验证", "结果吻合", "检验", "仿真", "证明", "分析", "灵敏度分析",
           "稳定性检验", "误差分析", "对比分析", "收敛性", "有效性"]
GAP_MARKERS = ["缺少推导", "无解释", "跳过步骤", "未给出推导", "缺乏推导", "直接给出"]   # 越低越好
MODEL_MATCH = ["模型", "选用模型", "适用性", "建模", "模型建立", "构建模型", "数学模型", "建立模型"]
ASSUME = ["假设", "前提条件", "假设合理", "作如下假设", "基本假设"]
COMPARE = ["对比", "优劣分析", "比较", "方案对比", "对比分析", "优劣"]
INNOVATION = ["创新点", "改进", "优化", "创新", "引入", "改进的", "融合"]
FORMULA_LABEL = ["公式编号", "符号说明", "式("]
DERIVE = ["步骤", "推导过程", "求解过程", "推导", "求解", "计算步骤"]
SYMBOL = ["符号定义", "符号说明", "变量定义", "记作", "令"]
CHAPTERS = ["摘要", "引言", "问题分析", "模型", "结论", "参考文献"]
ABSTRACT_EL = ["目的", "结果", "结论", "针对", "本文"]
REDUNDANT = ["重复赘述", "如前所述", "综上所述", "再次强调", "重复上述", "不言而喻"]  # 越低越好

PROBLEM_KEY = {
    "A": ["康养", "养老", "健康"],
    "B": ["数字教育", "教师", "评价", "数字素养", "胜任力"],
    "C": ["抑郁", "情绪", "生理信号", "心电", "量表"],
}
RESPONSE_WORDS = {
    "A": ["康养", "养老", "健康", "床位", "模型", "健康指标", "供需", "养老服务", "老年", "医疗", "资源"],
    "B": ["数字", "教育", "教师", "模型", "回归", "评价", "培训", "教学", "素养", "胜任力", "指标"],
    "C": ["抑郁", "情绪", "心电", "心率", "预测", "量表", "可穿戴", "情感", "信号"],
}

def classify_topic(full):
    text = clean_text(full)
    score = {k: sum(1 for w in v if w in text) for k, v in PROBLEM_KEY.items()}
    mx = max(score.values()) if score else 0
    if mx == 0:
        return "人工审核"
    return sorted([k for k, v in score.items() if v == mx])[0]

# ---------------- 文本分析 ----------------
_EQ = re.compile(r"(?<![<>=!])=(?![\d=])|\u2248|\u2264|\u2265")   # = ≈ ≤ ≥（排除==、!=、<=、>=、=\d）
_LABEL = re.compile(r"\(\s*\d")
_CITE = re.compile(r"\[\d+\]")

def count_formulas(full):
    return len(_EQ.findall(full))

def count_labeled_formulas(full):
    cnt = 0
    for m in _EQ.finditer(full):
        if _LABEL.search(full[m.end():m.end() + 30]):
            cnt += 1
    return cnt

def count_references(full):
    pos = -1
    for tag in ["参考文献", "References", "REFERENCES"]:
        i = full.find(tag)
        if i != -1:
            pos = i
            break
    if pos == -1:
        return 0
    return len(_CITE.findall(full[pos:]))

def analyze_text(pages, full):
    """提取一篇论文的通用文本统计量。"""
    clean = re.sub(r"\s+", "", full).lower()
    total_chars = len(clean)
    paragraphs = [p.strip() for p in full.split("\n") if len(p.strip()) > 0]
    sentences = [s.strip() for s in re.split(r"[。！？；\n]+", full) if len(re.sub(r"\s+", "", s)) > 0]
    formula_count = count_formulas(full)
    labeled_formula_count = count_labeled_formulas(full)
    citations = len(_CITE.findall(full))
    ref_count = count_references(full)
    connector_count = sum(clean.count(w) for w in CONNECTORS)
    def hit_ratio(words):
        hits = sum(1 for w in words if w in clean)
        return hits / len(words) if words else 0.0
    def hit_count(words):
        return sum(clean.count(w) for w in words)
    return dict(pages=len(pages), full=full, clean=clean, total_chars=total_chars,
                paragraphs=paragraphs, sentences=sentences,
                formula_count=formula_count, labeled_formula_count=labeled_formula_count,
                citations=citations, ref_count=ref_count, connector_count=connector_count,
                hit_ratio=hit_ratio, hit_count=hit_count)

# ---------------- 问题1 二级指标 ----------------
# 指标定义: id, 名称, 所属一级维度, 方向(+1越大越好/-1越小越好), 原始值函数
def p1_raw_indicators(t, topic):
    return {
        "u11": t["connector_count"] / max(1, t["total_chars"]) * 1000.0,          # 逻辑连接词密度(‰)
        "u12": t["hit_ratio"](RESPONSE_WORDS[topic]),                             # 赛题呼应覆盖率
        "u13": t["hit_ratio"](CLOSURE),                                           # 论证闭环完整率
        "u14": t["hit_count"](GAP_MARKERS),                                       # 逻辑断层条数(越低越好)
        "u21": t["hit_ratio"](MODEL_MATCH),                                       # 模型匹配度
        "u22": t["hit_count"](ASSUME) / max(1, t["total_chars"]) * 1000.0,       # 假设适配度(频率密度)
        "u23": t["hit_ratio"](COMPARE),                                           # 多方案对比完备度
        "u24": t["hit_ratio"](INNOVATION),                                        # 方法创新量化值
        "u31": t["formula_count"] / max(1, t["total_chars"]) * 1000.0,            # 有效公式密度(‰)
        "u32": 0.6 * min(1.0, t["labeled_formula_count"] / max(1, t["formula_count"]))
               + 0.4 * t["hit_ratio"](FORMULA_LABEL),                             # 公式标注规范率
        "u33": t["hit_ratio"](DERIVE),                                            # 推导步骤完整度
        "u34": t["hit_ratio"](SYMBOL),                                            # 符号统一度
        "u41": t["hit_ratio"](CHAPTERS),                                          # 标准章节完整率
        "u42": t["hit_ratio"](ABSTRACT_EL),                                       # 摘要要素完整度
        "u43": t["citations"],                                                    # 参考文献规范率(引用条数)
        "u44": t["hit_count"](REDUNDANT),                                         # 文本冗余度(越低越好)
    }

INDICATOR_META = {
    "u11": ("逻辑连接词密度", "U1", 1), "u12": ("赛题呼应覆盖率", "U1", 1),
    "u13": ("论证闭环完整率", "U1", 1), "u14": ("逻辑断层条数", "U1", -1),
    "u21": ("模型匹配度", "U2", 1), "u22": ("假设适配度", "U2", 1),
    "u23": ("多方案对比完备度", "U2", 1), "u24": ("方法创新量化值", "U2", 1),
    "u31": ("有效公式密度", "U3", 1), "u32": ("公式标注规范率", "U3", 1),
    "u33": ("推导步骤完整度", "U3", 1), "u34": ("符号统一度", "U3", 1),
    "u41": ("标准章节完整率", "U4", 1), "u42": ("摘要要素完整度", "U4", 1),
    "u43": ("参考文献规范率", "U4", 1), "u44": ("文本冗余度", "U4", -1),
}
DIM_NAMES = {"U1": "逻辑严密性", "U2": "方法合理性", "U3": "公式推导规范性", "U4": "结构与文本规范"}
U_ORDER = ["U1", "U2", "U3", "U4"]

# ---------------- 归一化与隶属度 ----------------
def minmax_norm(values, direction=1):
    """极差归一化；direction=-1 时反向(越小越好)。常数序列取0.5。"""
    arr = np.asarray(values, dtype=float)
    vmin, vmax = arr.min(), arr.max()
    if abs(vmax - vmin) < 1e-12:
        return np.full_like(arr, 0.5)
    s = (arr - vmin) / (vmax - vmin)
    if direction < 0:
        s = 1.0 - s
    return s

def compute_refs(raw_by_ind, kinds):
    """由参照样本计算各指标得分参照值（达标线），供新论文沿用。"""
    refs = {}
    for ind, arr in raw_by_ind.items():
        a = np.asarray(arr, dtype=float)
        k = kinds[ind]
        if k == "ratio":
            ref = 0.8                       # 覆盖率/达成率达标线 80%
        elif k == "density":
            ref = float(np.percentile(a, 80))
        elif k == "low":
            ref = float(np.percentile(a, 80))
        elif k == "count":
            ref = max(8.0, float(np.percentile(a, 80)))
        else:
            raise ValueError(k)
        refs[ind] = ref
    return refs

def apply_refs(arr, kind, ref):
    """用固定参照值把原始值映射为 s∈[0,1]（新论文沿用参照样本的参照值）。"""
    arr = np.asarray(arr, dtype=float)
    if kind == "ratio":
        return np.minimum(1.0, arr / ref)
    if kind == "density":
        if ref <= 0:
            return np.full_like(arr, 0.5)
        return np.minimum(1.0, arr / ref)
    if kind == "low":
        if ref <= 0:
            return np.ones_like(arr)
        return 1.0 - np.minimum(1.0, arr / ref)
    if kind == "count":
        return np.minimum(1.0, arr / ref)
    raise ValueError(kind)

def zscore_norm(arr, direction=1):
    """Z-score 标准化：z=(x-mean)/std；std≈0 时取 0；direction<0 取负（反向指标）。"""
    a = np.asarray(arr, dtype=float)
    std = a.std()
    if std < 1e-12:
        z = np.zeros_like(a)
    else:
        z = (a - a.mean()) / std
    if direction < 0:
        z = -z
    return z

def membership_vector(s):
    """连续隶属度函数：由归一化得分 s∈[0,1] 得到五级隶属度向量(和为1)。"""
    s = float(np.clip(s, 0.0, 1.0))
    mu = np.array([
        float(np.clip((0.30 - s) / 0.20, 0.0, 1.0)),   # 不及格
        max(0.0, 1.0 - abs(s - 0.30) / 0.20),          # 及格
        max(0.0, 1.0 - abs(s - 0.50) / 0.20),          # 中等
        max(0.0, 1.0 - abs(s - 0.70) / 0.20),          # 良好
        float(np.clip((s - 0.50) / 0.20, 0.0, 1.0)),   # 优秀
    ])
    total = mu.sum()
    if total <= 0:
        mu[2] = 1.0
    else:
        mu = mu / total
    return mu

def fce_score(norm_scores, weights):
    """模糊综合评价：
    norm_scores: dict U1..U4 -> [4个二级指标归一化得分]
    weights: dict 一级权重A 与 A1..A4
    返回 (B向量, 综合得分)
    """
    Bs = []
    for j, dim in enumerate(U_ORDER):
        Rj = np.array([membership_vector(s) for s in norm_scores[dim]])  # (4,5)
        Aj = weights[f"A{j+1}"]
        Bs.append(Aj @ Rj)
    R_total = np.vstack(Bs)                                # (4,5)
    A = weights["A"]
    B = A @ R_total
    B = B / B.sum()
    score = float(B @ GRADE_SCORES)
    return B, score


# ---------------- 模糊综合评价器（等级阈值按参照样本分位校准）----------------
def calibrate_centers(s_by_ind):
    """由参照样本各指标得分的分位确定五级隶属度中心与宽度。
    中心 = p10,p30,p50,p70,p90（不及格..优秀），宽度=(p90-p10)/4（最小0.06）。"""
    centers, widths = {}, {}
    for ind, arr in s_by_ind.items():
        a = np.asarray(arr, dtype=float)
        if float(np.percentile(a, 90) - np.percentile(a, 10)) < 0.15:
            # 常数指标：按绝对水平映射（固定中心），避免“人人中等”的伪居中
            c = np.array([0.15, 0.30, 0.50, 0.70, 0.85])
            w = 0.20
        else:
            c = np.percentile(a, [5, 30, 50, 70, 95])
            w = max(0.06, float((c[4] - c[0]) / 4.0))
        centers[ind] = c
        widths[ind] = w
    return centers, widths

class FCEScorer:
    """AHP权重 + 模糊综合评价：B = A·R，综合得分 = B·等级分值。"""
    def __init__(self, weights, centers, widths):
        self.A = {"A": weights["M(一级)"]["w"]}
        for i in range(1, 5):
            key = [k for k in weights if k.startswith(f"M{i}(")][0]
            self.A[f"A{i}"] = weights[key]["w"]
        self.centers = centers
        self.widths = widths

    def membership(self, ind, s):
        c = self.centers[ind]; w = self.widths[ind]
        s = float(np.clip(s, 0.0, 1.0))
        mu = np.array([
            float(np.clip((c[1] - s) / w, 0.0, 1.0)) if s <= c[1] else max(0.0, 1.0 - (s - c[0]) / w),
            max(0.0, 1.0 - abs(s - c[1]) / w),
            max(0.0, 1.0 - abs(s - c[2]) / w),
            max(0.0, 1.0 - abs(s - c[3]) / w),
            float(np.clip((s - c[3]) / w, 0.0, 1.0)) if s >= c[3] else max(0.0, 1.0 - (c[4] - s) / w),
        ])
        total = mu.sum()
        if total <= 0:
            mu[2] = 1.0
        else:
            mu = mu / total
        return mu

    def score(self, norm_scores, ind_order):
        """norm_scores: dict 维度 -> [4个指标得分]；ind_order: 维度->指标id列表。"""
        Bs = []
        for j, dim in enumerate(U_ORDER):
            Rj = np.array([self.membership(ind, norm_scores[dim][k])
                           for k, ind in enumerate(ind_order[dim])])
            Bs.append(self.A[f"A{j+1}"] @ Rj)
        R_total = np.vstack(Bs)
        B = self.A["A"] @ R_total
        B = B / B.sum()
        score = float(B @ GRADE_SCORES)
        return B, score

# ---------------- PIL 图表 ----------------
def _font(size, bold=False):
    cands = [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyhbd.ttc",
             r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc",
             r"C:\Windows\Fonts\Deng.ttf"]
    if bold:
        cands = [r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\msyhbd.ttc",
                 r"C:\Windows\Fonts\Dengb.ttf"] + cands
    for c in cands:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()

def _base_image(w, h, bg=(255, 255, 255)):
    img = Image.new("RGB", (w, h), bg)
    return img, ImageDraw.Draw(img)

def bar_chart(labels, values, path, title="", ylabel="", color=(0x44, 0x72, 0xC4),
              show_values=True, rotate=0, ymax=None, fmt="{:.0f}", figsize=(920, 540),
              grid_color=(0xE0, 0xE0, 0xE0), text_color=(0x33, 0x33, 0x33)):
    img, d = _base_image(*figsize)
    fmt0 = fmt.strip("{}:")
    w, h = figsize
    ml, mr, mt, mb = 90, 30, 70, 90
    pw, ph = w - ml - mr, h - mt - mb
    f_t = _font(24, bold=True); f_l = _font(16); f_v = _font(15); f_a = _font(14)
    vmax = ymax if ymax is not None else (max(values) * 1.18 if values else 1)
    if vmax <= 0: vmax = 1
    d.text((w / 2, 26), title, fill=text_color, font=f_t, anchor="mm")
    # grid + axis
    for g in range(5):
        gy = mt + ph - ph * g / 4
        d.line([(ml, gy), (w - mr, gy)], fill=grid_color, width=1)
        d.text((ml - 10, gy), f"{vmax * g / 4:{fmt0}}", fill=text_color, font=f_v, anchor="rm")
    d.line([(ml, mt), (ml, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    d.line([(ml, mt + ph), (w - mr, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    if ylabel:
        d.text((22, mt + ph / 2), ylabel, fill=text_color, font=f_l, anchor="mm")
    n = max(1, len(labels))
    bw = pw / n * 0.62
    for i, (lab, val) in enumerate(zip(labels, values)):
        x0 = ml + pw / n * i + (pw / n - bw) / 2
        bh = ph * (val / vmax)
        d.rectangle([x0, mt + ph - bh, x0 + bw, mt + ph], fill=color)
        if show_values:
            d.text((x0 + bw / 2, mt + ph - bh - 12), f"{val:{fmt0}}", fill=(0x1F, 0x3B, 0x64),
                   font=f_v, anchor="ms")
        if rotate:
            d.text((x0 + bw / 2, mt + ph + 14), str(lab), fill=text_color, font=f_a, anchor="ms")
        else:
            d.text((x0 + bw / 2, mt + ph + 14), str(lab), fill=text_color, font=f_a, anchor="ma")
    img.save(path)
    return path

def grouped_bar(labels, series, path, title="", ylabel="", colors=None,
                ymax=None, fmt="{:.2f}", figsize=(980, 560), rotate=0):
    """series: [(name, [values...]), ...]"""
    img, d = _base_image(*figsize)
    fmt0 = fmt.strip("{}:")
    w, h = figsize
    ml, mr, mt, mb = 90, 30, 70, 100
    pw, ph = w - ml - mr, h - mt - mb
    f_t = _font(24, bold=True); f_l = _font(16); f_v = _font(14); f_a = _font(14); f_g = _font(15)
    allv = [v for _, vs in series for v in vs]
    vmax = ymax if ymax is not None else (max(allv) * 1.18 if allv else 1)
    if vmax <= 0: vmax = 1
    if colors is None:
        colors = [(0x44, 0x72, 0xC4), (0xED, 0x7D, 0x31), (0x70, 0xAD, 0x47), (0x8E, 0x44, 0xAD)]
    d.text((w / 2, 26), title, fill=(0x33, 0x33, 0x33), font=f_t, anchor="mm")
    for g in range(5):
        gy = mt + ph - ph * g / 4
        d.line([(ml, gy), (w - mr, gy)], fill=(0xE0, 0xE0, 0xE0), width=1)
        d.text((ml - 10, gy), f"{vmax * g / 4:{fmt0}}", fill=(0x33, 0x33, 0x33), font=f_v, anchor="rm")
    d.line([(ml, mt), (ml, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    d.line([(ml, mt + ph), (w - mr, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    if ylabel:
        d.text((22, mt + ph / 2), ylabel, fill=(0x33, 0x33, 0x33), font=f_l, anchor="mm")
    n = len(labels); k = len(series)
    gw = pw / n
    bw = gw * 0.72 / k
    for i, lab in enumerate(labels):
        for j, (name, vs) in enumerate(series):
            val = vs[i]
            x0 = ml + gw * i + (gw - bw * k) / 2 + j * bw
            bh = ph * (val / vmax)
            d.rectangle([x0, mt + ph - bh, x0 + bw, mt + ph], fill=colors[j % len(colors)])
            d.text((x0 + bw / 2, mt + ph - bh - 10), f"{val:{fmt0}}", fill=(0x1F, 0x3B, 0x64),
                   font=f_v, anchor="ms")
        if rotate:
            d.text((ml + gw * i + gw / 2, mt + ph + 16), str(lab), fill=(0x33, 0x33, 0x33),
                   font=f_a, anchor="ms")
        else:
            d.text((ml + gw * i + gw / 2, mt + ph + 16), str(lab), fill=(0x33, 0x33, 0x33),
                   font=f_a, anchor="ma")
    if k > 1:
        lx = ml
        for j, (name, _) in enumerate(series):
            d.rectangle([lx, mt + 8, lx + 18, mt + 26], fill=colors[j % len(colors)])
            d.text((lx + 24, mt + 17), name, fill=(0x33, 0x33, 0x33), font=f_g, anchor="lm")
            lx += 24 + d.textlength(name, font=f_g) + 24
    img.save(path)
    return path

def line_chart(xs, series, path, title="", xlabel="", ylabel="", colors=None,
               ymax=None, ymin=None, figsize=(960, 560), fmt="{:.1f}", legend=True):
    img, d = _base_image(*figsize)
    fmt0 = fmt.strip("{}:")
    w, h = figsize
    ml, mr, mt, mb = 90, 30, 70, 90
    pw, ph = w - ml - mr, h - mt - mb
    f_t = _font(24, bold=True); f_l = _font(16); f_v = _font(14); f_a = _font(14); f_g = _font(15)
    allv = [v for _, vs in series for v in vs]
    vmin = ymin if ymin is not None else (min(allv) * 0.9 if allv else 0)
    vmax = ymax if ymax is not None else (max(allv) * 1.1 if allv else 1)
    if colors is None:
        colors = [(0x44, 0x72, 0xC4), (0xED, 0x7D, 0x31), (0x70, 0xAD, 0x47)]
    d.text((w / 2, 26), title, fill=(0x33, 0x33, 0x33), font=f_t, anchor="mm")
    for g in range(5):
        gy = mt + ph - ph * g / 4
        d.line([(ml, gy), (w - mr, gy)], fill=(0xE0, 0xE0, 0xE0), width=1)
        val = vmin + (vmax - vmin) * g / 4
        d.text((ml - 10, gy), f"{val:{fmt0}}", fill=(0x33, 0x33, 0x33), font=f_v, anchor="rm")
    d.line([(ml, mt), (ml, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    d.line([(ml, mt + ph), (w - mr, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    if ylabel:
        d.text((22, mt + ph / 2), ylabel, fill=(0x33, 0x33, 0x33), font=f_l, anchor="mm")
    n = len(xs)
    def X(i): return ml + (pw / (n - 1)) * i if n > 1 else ml + pw / 2
    def Y(v): return mt + ph - ph * (v - vmin) / (vmax - vmin)
    for j, (name, vs) in enumerate(series):
        col = colors[j % len(colors)]
        pts = [(X(i), Y(v)) for i, v in enumerate(vs)]
        d.line(pts, fill=col, width=3)
        for (x0, y0) in pts:
            d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill=col)
    for i, lab in enumerate(xs):
        d.text((X(i), mt + ph + 16), str(lab), fill=(0x33, 0x33, 0x33), font=f_a, anchor="ma")
    if legend:
        lx = ml
        for j, (name, _) in enumerate(series):
            col = colors[j % len(colors)]
            d.line([(lx, mt + 17), (lx + 24, mt + 17)], fill=col, width=3)
            d.text((lx + 30, mt + 17), name, fill=(0x33, 0x33, 0x33), font=f_g, anchor="lm")
            lx += 30 + d.textlength(name, font=f_g) + 28
    img.save(path)
    return path

def radar_chart(categories, series, path, title="", ymax=1.0, colors=None, figsize=(760, 660)):
    """series: [(name, [values])] 值域 [0, ymax]"""
    img, d = _base_image(*figsize)

    w, h = figsize
    cx, cy, R = w / 2, h / 2 - 10, min(w, h) / 2 - 90
    f_t = _font(24, bold=True); f_a = _font(15); f_g = _font(16); f_v = _font(13)
    d.text((w / 2, 30), title, fill=(0x33, 0x33, 0x33), font=f_t, anchor="mm")
    n = len(categories)
    def pt(i, r):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return (cx + r * math.cos(ang), cy + r * math.sin(ang))
    for g in range(1, 6):
        r = R * g / 5
        d.polygon([pt(i, r) for i in range(n)], outline=(0xD0, 0xD0, 0xD0), fill=None)
    for i in range(n):
        x, y = pt(i, R)
        d.line([(cx, cy), (x, y)], fill=(0xD0, 0xD0, 0xD0), width=1)
        tx, ty = pt(i, R + 34)
        d.text((tx, ty), str(categories[i]), fill=(0x33, 0x33, 0x33), font=f_a, anchor="mm")
    if colors is None:
        colors = [(0x44, 0x72, 0xC4), (0xED, 0x7D, 0x31), (0x70, 0xAD, 0x47)]
    for j, (name, vals) in enumerate(series):
        col = colors[j % len(colors)]
        pts = [pt(i, R * max(0.02, min(1.0, vals[i] / ymax))) for i in range(n)]
        d.polygon(pts, outline=col, fill=col + (40,), width=3)
    if len(series) > 1:
        ly = cy + R + 70
        lx = cx - 80
        for j, (name, _) in enumerate(series):
            col = colors[j % len(colors)]
            d.rectangle([lx, ly, lx + 18, ly + 16], fill=col)
            d.text((lx + 24, ly + 8), name, fill=(0x33, 0x33, 0x33), font=f_g, anchor="lm")
            ly += 26
    img.save(path)
    return path

def hbar_chart(labels, values, path, title="", errors=None, color=(0x44, 0x72, 0xC4),
               fmt="{:.2f}", figsize=(900, 560), xmax=None, xmin=0):
    img, d = _base_image(*figsize)
    fmt0 = fmt.strip("{}:")
    w, h = figsize
    ml, mr, mt, mb = 150, 70, 70, 60
    pw, ph = w - ml - mr, h - mt - mb
    f_t = _font(24, bold=True); f_l = _font(15); f_v = _font(14); f_a = _font(14)
    allv = [v for v in values]
    vmax = xmax if xmax is not None else (max(allv) * 1.2 if allv else 1)
    d.text((w / 2, 26), title, fill=(0x33, 0x33, 0x33), font=f_t, anchor="mm")
    for g in range(5):
        gx = ml + pw * g / 4
        d.line([(gx, mt), (gx, mt + ph)], fill=(0xE0, 0xE0, 0xE0), width=1)
        d.text((gx, mt + ph + 12), f"{vmax * g / 4:{fmt0}}", fill=(0x33, 0x33, 0x33), font=f_v, anchor="ma")
    d.line([(ml, mt), (ml, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    d.line([(ml, mt + ph), (w - mr, mt + ph)], fill=(0x80, 0x80, 0x80), width=2)
    n = len(labels)
    bh = ph / n * 0.62
    for i, (lab, val) in enumerate(zip(labels, values)):
        y0 = mt + ph / n * i + (ph / n - bh) / 2
        bw = pw * float(np.clip((val - xmin) / (vmax - xmin), 0.0, 1.0)) if vmax != xmin else 0.0
        d.rectangle([ml, y0, ml + bw, y0 + bh], fill=color)
        d.text((ml - 10, y0 + bh / 2), str(lab), fill=(0x33, 0x33, 0x33), font=f_l, anchor="rm")
        d.text((ml + bw + 6, y0 + bh / 2), f"{val:{fmt0}}", fill=(0x1F, 0x3B, 0x64), font=f_v, anchor="lm")
        if errors is not None and errors[i] is not None:
            e = errors[i]
            ex0 = ml + pw * float(np.clip((val - e - xmin) / (vmax - xmin), 0.0, 1.0)) if vmax != xmin else ml
            ex1 = ml + pw * float(np.clip((val + e - xmin) / (vmax - xmin), 0.0, 1.0)) if vmax != xmin else ml
            d.line([(ex0, y0 + bh / 2), (ex1, y0 + bh / 2)], fill=(0xED, 0x7D, 0x31), width=3)
    img.save(path)
    return path

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)