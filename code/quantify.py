# -*- coding: utf-8 -*-
"""问题1 二级指标专属量化模块（16个指标逐一使用专门方法）。

设计原则：
- 每个指标有独立的量化逻辑（分节覆盖、闭环检测、断层规则、模型匹配、
  公式行检测、符号覆盖率、行首章节检测、n-gram 冗余等）。
- jieba 可用时用于 模型/方法名抽取、摘要要素判断、句级重复检测；
  缺失时自动降级为 词典正则+字符级，不影响运行。
"""
import re
import numpy as np

try:
    import jieba
    _HAS_JIEBA = True
except Exception:
    _HAS_JIEBA = False

# ---------------- 通用文本结构工具 ----------------
def _norm(s):
    """去掉空白（用于匹配“摘 要”这类被空格拆开的标题）。"""
    return re.sub(r"\s+", "", s)

def abstract_region(full):
    m = re.search(r"摘\s*要", full)
    if not m:
        m = re.search(r"abstract", full, re.IGNORECASE)
    if not m:
        return ""
    idx = m.start()
    ends = [full.find(w, idx + 2) for w in ["关键词", "引言", "绪论", "第一章", "一、"]]
    ends = [e for e in ends if e != -1]
    end = min(ends) if ends else min(len(full), idx + 3000)
    return full[idx:end]

_EQ_LINE = re.compile(r"[=≈≤≥∑∫√×÷]")
def detect_equation_lines(full):
    """按行检测公式行：含数学运算符、中文占比低（排除判断矩阵比较等含=的中文长句）。"""
    lines = full.split("\n")
    eq_idx, eq_texts = [], []
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s or len(s) > 200:
            continue
        if re.match(r"^(图|表)\s*\d", s):
            continue
        if not _EQ_LINE.search(s):
            continue
        if re.search(r"[=≈≤≥]\s*\d+(\.\d+)?\s*(元|万|%|人|个|公里|km|m|年|月)", s):
            continue
        body = re.sub(r"\s+", "", s)
        cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
        ratio = cjk / len(body) if body else 1.0
        has_sym = bool(re.search(r"[A-Za-z\u03b1-\u03c9\u0391-\u03a9]", s))
        if ratio < 0.5 or (has_sym and len(body) <= 80 and ratio < 0.75):
            eq_idx.append(i)
            eq_texts.append(s)
    return eq_idx, eq_texts

_LABEL = re.compile(r"\(\s*\d+(\.\d+)?\s*\)")
def count_labeled_equations(full, eq_idx):
    """统计带 (n) 编号的公式行：标签出现在本行或随后2行内即算已标注。"""
    lines = full.split("\n")
    cnt = 0
    for i in eq_idx:
        ctx = "\n".join(lines[i:min(len(lines), i + 3)])
        if _LABEL.search(ctx):
            cnt += 1
    return cnt

def parse_references(full):
    """返回 (正文引用条数, 参考文献区条目数)。"""
    intext = len(re.findall(r"\[\d+\]", full))
    pos = -1
    for tag in ["参考文献", "References", "REFERENCES"]:
        i = full.find(tag)
        if i != -1:
            pos = i
            break
    if pos == -1:
        return intext, 0
    ref_text = full[pos:]
    entries = len(re.findall(r"\[\d+\]", ref_text))
    entry_lines = sum(1 for ln in ref_text.split("\n") if re.match(r"^\s*\[\d+\]", ln))
    return intext, max(entries, entry_lines)

_SYM_CHARS = r"A-Za-z\u03b1-\u03c9\u0391-\u03a9"
_SYM_DEF = [
    re.compile(r"(?:记|设|令)\s*([%s])\s*(?:为|表示|是)" % _SYM_CHARS),
    re.compile(r"其中[，,、]?\s*([%s])\s*(?:表示|为|是)" % _SYM_CHARS),
    re.compile(r"式中[，,、]?\s*([%s])\s*(?:表示|为|是)" % _SYM_CHARS),
    re.compile(r"这里[，,、]?\s*([%s])\s*(?:表示|为|是)" % _SYM_CHARS),
    re.compile(r"([%s])\s*[:：=]\s*(?:表示|为|表示的是)" % _SYM_CHARS),
]
def extract_defined_symbols(full):
    syms = set()
    for pat in _SYM_DEF:
        for m in pat.finditer(full):
            syms.add(m.group(1))
    return syms

def extract_formula_symbols(eq_texts):
    syms = set()
    for s in eq_texts:
        for ch in s:
            if re.match(r"[%s]" % _SYM_CHARS, ch):
                syms.add(ch)
    return syms

def ngram_repetition_rate(clean, n=10):
    if len(clean) < n + 1:
        return 0.0
    seen = set(); covered = set()
    for i in range(len(clean) - n + 1):
        g = clean[i:i + n]
        if g in seen:
            covered.update(range(i, i + n))
        else:
            seen.add(g)
    return len(covered) / max(1, len(clean))

def dup_sentence_rate(sentences):
    seen = {}
    for s in sentences:
        k = re.sub(r"\s+", "", s)
        if k:
            seen[k] = seen.get(k, 0) + 1
    dup = sum(v - 1 for v in seen.values() if v > 1)
    return dup / max(1, len(seen))

# ---------------- 行首章节检测（标题级，含目录行，目录亦反映章节存在）----------------
U41_SECTIONS = [
    ("摘要", ["摘要"]),
    ("引言/问题重述", ["引言", "绪论", "问题重述", "问题提出", "问题描述"]),
    ("问题分析", ["问题分析", "问题一", "问题二", "问题三", "问题四", "问题"]),
    ("模型假设", ["模型假设", "基本假设", "假设条件"]),
    ("模型建立与求解", ["模型建立", "模型构建", "模型的建立", "建模", "模型求解", "求解方法"]),
    ("结果分析", ["结果分析", "仿真结果", "实验结果", "结果与分析"]),
    ("模型检验/灵敏度", ["模型检验", "灵敏度", "稳定性", "误差分析", "稳健性"]),
    ("结论", ["结论", "小结", "总结"]),
    ("参考文献", ["参考文献", "references"]),
]
def line_start_keyword(full, words):
    """关键词出现在行首（剥编号/空格/全角标点后），或出现在较短标题行内。"""
    for ln in full.split("\n"):
        s = _norm(ln)
        s = re.sub(r"^[0-9.．·()（）、，,：:一二三四五六七八九十百章篇节\s]+", "", s)
        for w in words:
            if s.startswith(w):
                return True
            if len(s) <= 32 and w in s:
                return True
    return False

# ---------------- 模型/方法词典（用于 u21/u23/u24）----------------
MODEL_LEXICON = {
    "优化": ["线性规划", "整数规划", "目标规划", "动态规划", "粒子群", "遗传算法", "模拟退火",
             "蚁群算法", "贪心算法", "贪婪算法", "启发式", "深度学习", "强化学习", "深度强化学习",
             "DQN", "支持向量", "最优化", "凸优化", "多目标优化", "SLSQP", "梯度下降", "牛顿法", "爬山法"],
    "预测": ["回归", "线性回归", "逻辑回归", "时间序列", "ARIMA", "指数平滑", "灰色预测",
             "神经网络", "BP神经网络", "LSTM", "随机森林", "支持向量机", "SVM", "XGBoost",
             "梯度提升", "马尔可夫", "插值", "曲线拟合", "最小二乘"],
    "评价": ["层次分析法", "AHP", "熵权法", "熵值法", "TOPSIS", "灰色关联", "模糊综合评价",
             "模糊综合", "主成分", "因子分析", "数据包络", "DEA", "聚类", "K-means", "加权平均",
             "德尔菲法", "变异系数", "耦合协调"],
}
_ALL_MODEL_WORDS = {w for ws in MODEL_LEXICON.values() for w in ws}

def extract_model_names(clean):
    found = {cat: set() for cat in MODEL_LEXICON}
    for cat, words in MODEL_LEXICON.items():
        for w in words:
            if w.lower() in clean:
                found[cat].add(w)
    return found

# ---------------- 代码页检测（剔除代码附录，避免污染篇幅/公式特征）----------------
CODE_KEYWORDS = [
    "import ", "def ", "for i in", "while ", "print(", "return ", "subplot", "plt.",
    "numpy", "pandas", "if __name__", "class ", "end;", "library(", "read.table",
    "summary(", "lm(", "ggplot", "int main", "scanf", "printf", "#include", "void ",
    "else if", "try:", "except", "lambda ", "range(", "for i=", "= [", "disp(",
    "eig(", "zeros(", "ones(", "repmat", "read.csv", "pd.read", "plt.show",
]

def is_code_page(text):
    """判断一页是否以程序代码为主（避免把数学公式页/内容页误判为代码）。"""
    if not text or len(text) < 40:
        return False
    body = re.sub(r"\s+", "", text)
    cjk = len(re.findall(r"[\u4e00-\u9fff]", body))
    ratio = cjk / max(1, len(body))
    kw = sum(1 for k in CODE_KEYWORDS if k.lower() in text.lower())
    return kw >= 2 or (kw >= 1 and ratio < 0.25)

def strip_code_pages(pages):
    """剔除代码页，返回 (正文页列表, 正文全文)。若全部页被判为代码则回退为全部页。"""
    body = [p for p in pages if not is_code_page(p)]
    if not body:
        body = pages
    return body, "\n".join(body)

# ---------------- 各指标用到的词典 ----------------
CONN_CAT = {
    "因果": ["因此", "所以", "因而", "故", "从而", "由此", "据此", "于是", "故而"],
    "转折": ["然而", "但是", "不过", "而", "却"],
    "递进": ["进而", "此外", "进一步", "同时", "并且", "而且"],
    "总结": ["综上", "综上所述", "总之", "结果表明", "可得"],
}
RESULT_MARKERS = ["结果表明", "结果显示", "实验结果表明", "得到", "实现", "达到", "提升至", "发现", "验证了"]
VERIFY_MARKERS = ["验证", "检验", "仿真", "误差", "对比", "灵敏度", "吻合", "证明", "分析", "一致性", "收敛"]
SECT_BUCKETS = [
    ("摘要", ["摘要"]),
    ("分析", ["问题重述", "问题分析", "引言", "绪论"]),
    ("模型", ["模型建立", "模型求解", "模型假设", "模型的建立", "建模"]),
    ("结果", ["结果分析", "模型检验", "结论", "小结"]),
]
QUALIFIERS = ["忽略", "近似", "当", "不失一般性", "假定", "视作"]
DERIVE_LINKS = ["由式", "代入", "联立", "化简可得", "即", "从而得到", "解得", "进一步可得", "令", "将"]
INNO_PATTERNS = ["提出", "改进的", "引入", "融合", "相比", "优于", "克服", "创新", "设计了", "构建了", "首次"]
FORMULA_NUM = re.compile(r"\(\s*(\d{1,2})\s*\)")

def _region_between(full, start_words, end_words):
    """定位 start_words 首次出现与其后 end_words 首次出现之间的文本。"""
    s = -1
    for w in start_words:
        i = full.find(w)
        if i != -1:
            s = i
            break
    if s == -1:
        return ""
    e = len(full)
    for w in end_words:
        i = full.find(w, s + 1)
        if i != -1:
            e = min(e, i)
    return full[s:e]

# ---------------- 16 个专属量化函数 ----------------
def q_u11(t):
    sents = t["sentences"]
    n = max(1, len(sents))
    with_conn = 0
    cat_cnt = {k: 0 for k in CONN_CAT}
    for s in sents:
        hit = False
        for cat, words in CONN_CAT.items():
            c = sum(s.count(w) for w in words)
            if c:
                hit = True
                cat_cnt[cat] += c
        if hit:
            with_conn += 1
    ratio = with_conn / n
    total = sum(cat_cnt.values())
    if total:
        ps = np.array([c / total for c in cat_cnt.values()])
        ps = ps[ps > 0]
        H = float(-np.sum(ps * np.log(ps)) / np.log(4))
    else:
        H = 0.0
    return 0.7 * ratio + 0.3 * H

def q_u12(t, topic):
    from common import RESPONSE_WORDS
    core = RESPONSE_WORDS[topic]
    full = t["full"]
    whole = t["hit_ratio"](core)
    covs = []
    for _, bucket_words in SECT_BUCKETS:
        if bucket_words[0] == "摘要":
            txt = abstract_region(full)
        elif bucket_words[0] == "结果":
            txt = _region_between(full, ["结果分析", "结论", "小结"], ["参考文献"])
        else:
            txt = _region_between(full, bucket_words, ["结果", "结论", "参考文献"])
        if txt:
            hits = sum(1 for w in core if w in txt)
            covs.append(hits / max(1, len(core)))
    if covs:
        return 0.5 * whole + 0.5 * float(np.mean(covs))
    return whole

def q_u13(t):
    sents = t["sentences"]
    n = len(sents)
    res_idx = [i for i, s in enumerate(sents) if any(m in s for m in RESULT_MARKERS)]
    if not res_idx:
        return 0.0
    infra = any(w in t["clean"] for w in ["验证", "检验", "灵敏度", "误差分析", "对比分析", "仿真", "模型检验", "收敛性"])
    closed = 0
    for i in res_idx:
        ctx = "".join(sents[max(0, i - 3):min(n, i + 4)])
        if any(v in ctx for v in VERIFY_MARKERS):
            closed += 1
    nearby = closed / len(res_idx)
    return 0.6 * nearby + 0.4 * (1.0 if infra else 0.0)

def q_u14(t):
    full, clean = t["full"], t["clean"]
    jumps = 0
    # (a) 公式编号跳号：相邻编号缺口计1次（最多3次），避免噪声放大
    nums = sorted({int(m) for m in FORMULA_NUM.findall(full) if int(m) <= 50})
    if len(nums) >= 3:
        for a, b in zip(nums, nums[1:]):
            if b - a > 1:
                jumps += 1
        jumps = min(jumps, 3)
    # (b) 假设声明但未复用
    if "假设" in clean:
        first_model = clean.find("模型")
        reuse = (clean.count("假设") >= 2) or (first_model != -1 and first_model < clean.rfind("假设"))
        if not reuse:
            jumps += 1
    # (c) 模型段无公式
    if "模型" in clean and len(detect_equation_lines(full)[0]) < 3:
        jumps += 1
    # (d) “可得/解得”后紧跟公式但前文无推导衔接
    sents = t["sentences"]
    for i, s in enumerate(sents):
        if re.search(r"(?:可|即|从而|由上式)得\s*[^。；]{0,12}[=≈≤≥∑∫]", s):
            prev = "".join(sents[max(0, i - 2):i])
            if not any(w in prev for w in ["由", "代入", "根据", "联立", "化简", "推导", "计算", "式("]):
                jumps += 1
    return min(jumps, 8)

def q_u21(t):
    found = extract_model_names(t["clean"])
    counts = {c: len(v) for c, v in found.items()}
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total

def q_u22(t):
    assume_sents = [s for s in t["sentences"]
                    if any(w in s for w in ["假设", "设", "前提条件", "忽略", "近似", "不失一般性"])]
    s1 = min(1.0, len(assume_sents) / 5.0)
    clean = t["clean"]
    if "假设" in clean:
        first_model = clean.find("模型")
        reuse = 1.0 if (clean.count("假设") >= 2 or (first_model != -1 and first_model < clean.rfind("假设"))) else 0.5
    else:
        reuse = 0.0
    q = sum(1 for s in assume_sents if any(w in s for w in QUALIFIERS))
    s3 = q / max(1, len(assume_sents))
    return 0.5 * s1 + 0.3 * reuse + 0.2 * s3

def q_u23(t):
    clean = t["clean"]
    found = extract_model_names(clean)
    total_models = sum(len(v) for v in found.values())
    m1 = min(1.0, total_models / 3.0)
    cmp_terms = sum(clean.count(w) for w in ["对比", "比较", "优于", "低于", "精度", "误差", "性能"])
    m2 = min(1.0, cmp_terms / 5.0)
    figtab = len(re.findall(r"(?:图|表)\s*\d+", t["full"]))
    m3 = min(1.0, figtab / 4.0)
    return 0.4 * m1 + 0.4 * m2 + 0.2 * m3

def q_u24(t):
    clean = t["clean"]
    # 创新句式“种类”覆盖率（出现即计，与篇幅无关）
    s1 = sum(1 for w in INNO_PATTERNS if w in clean) / len(INNO_PATTERNS)
    rare = 0
    if _HAS_JIEBA:
        for tok in jieba.cut(clean):
            if re.search(r"(算法|模型|方法|网络|学习|法)$", tok) and len(tok) >= 2:
                if tok not in _ALL_MODEL_WORDS:
                    rare += 1
    else:
        rare = len(re.findall(r"(改进|自适应|多目标|混合|深度|智能|融合)[\u4e00-\u9fa5]{1,5}(算法|模型|方法)", clean))
    s2 = min(1.0, rare / 5.0)
    return 0.6 * s1 + 0.4 * s2

def q_u31(t):
    lines = t["full"].split("\n")
    total_lines = max(1, len(lines))
    n_eq = len(detect_equation_lines(t["full"])[0])
    return n_eq / total_lines * 1000.0

def q_u32(t):
    eq_idx, _ = detect_equation_lines(t["full"])
    n_eq = len(eq_idx)
    n_lab = count_labeled_equations(t["full"], eq_idx)
    r1 = n_lab / max(1, n_eq)
    sym_sec = 1.0 if any(w in t["clean"] for w in ["符号说明", "变量定义", "符号定义"]) else 0.0
    return 0.7 * r1 + 0.3 * sym_sec

def q_u33(t):
    clean = t["clean"]
    links = sum(clean.count(w) for w in DERIVE_LINKS)
    s1 = min(1.0, links / 8.0)
    proc = 1.0 if any(w in clean for w in ["步骤", "算法", "流程", "伪代码", "求解过程", "迭代"]) else 0.0
    return 0.7 * s1 + 0.3 * proc

def q_u34(t):
    full = t["full"]
    tag_idx = -1
    for tag in ["符号说明", "变量定义", "符号定义", "符号表"]:
        i = full.find(tag)
        if i != -1:
            tag_idx = i
            break
    if tag_idx != -1:
        zone = full[tag_idx:tag_idx + 1500]
        entries = 0
        for ln in zone.split("\n")[:30]:
            if re.search(r"[%s].{0,20}(表示|为|是|单位|指)" % _SYM_CHARS, ln):
                entries += 1
        return min(1.0, entries / 8.0)
    _, eq_texts = detect_equation_lines(full)
    defined = extract_defined_symbols(full)
    fsym = extract_formula_symbols(eq_texts)
    if not fsym:
        return 0.5
    return len(defined & fsym) / len(fsym)

def q_u41(t):
    hit = sum(1 for _, words in U41_SECTIONS if line_start_keyword(t["full"], words))
    return hit / len(U41_SECTIONS)

def q_u42(t):
    zone = abstract_region(t["full"])
    if not zone:
        return 0.0
    elements = {
        "目的": ["针对", "为了解决", "旨在", "目的"],
        "方法": ["本文", "采用", "构建", "建立", "提出", "运用", "基于", "利用"],
        "结果": ["结果表明", "结果显示", "得到", "实现", "达到", "提升"],
        "结论": ["结论", "建议", "意义", "价值", "说明"],
    }
    hit = sum(1 for ws in elements.values() if any(w in zone for w in ws))
    return hit / 4.0

def q_u43(t):
    intext, entries = parse_references(t["full"])
    s1 = min(1.0, intext / 10.0)
    s2 = min(1.0, entries / 10.0)
    denom = max(1, max(intext, entries))
    cons = 1.0 - min(1.0, abs(intext - entries) / denom)
    return 0.5 * s1 + 0.3 * s2 + 0.2 * cons

def q_u44(t):
    dup = dup_sentence_rate(t["sentences"])
    return dup

QUANTIFIERS = {
    "u11": q_u11, "u12": q_u12, "u13": q_u13, "u14": q_u14,
    "u21": q_u21, "u22": q_u22, "u23": q_u23, "u24": q_u24,
    "u31": q_u31, "u32": q_u32, "u33": q_u33, "u34": q_u34,
    "u41": q_u41, "u42": q_u42, "u43": q_u43, "u44": q_u44,
}
KIND = {"u11": "ratio", "u12": "ratio", "u13": "ratio", "u14": "low",
        "u21": "ratio", "u22": "ratio", "u23": "ratio", "u24": "ratio",
        "u31": "density", "u32": "ratio", "u33": "ratio", "u34": "ratio",
        "u41": "ratio", "u42": "ratio", "u43": "ratio", "u44": "low"}

def quantify_all(t, topic="A"):
    """返回 16 个二级指标的原始量化值 dict。"""
    out = {}
    for ind in QUANTIFIERS:
        out[ind] = QUANTIFIERS[ind](t, topic) if ind == "u12" else QUANTIFIERS[ind](t)
    return out