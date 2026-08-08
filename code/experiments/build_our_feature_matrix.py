# -*- coding: utf-8 -*-
"""实验：按参考论文 p1_extract_features.py 的 22 列特征公式，生成"我们的特征矩阵"。
差异点（只改特征矩阵，不动 p4 算法）：
1) 文本用我们的流程提取：pdfplumber + 剔除代码附录页后的正文（参考代码用原始全文，含代码污染）；
2) equations 用我们的行级公式检测 detect_equation_lines（参考代码用 $ 等 LaTeX 标记，在 PDF 中恒为 0）；
3) 其余计数/密度/句式/术语/AI三特征均照抄参考公式，保证 p4 算法口径不变。
输出：output/experiments/all_paper_features_ours.csv（与参考同名同列）。
"""
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C
import quantify

# ---- 参考论文的词典与正则（照抄，保证口径一致）----
LOGICAL_CONNECTORS = [
    "因此", "所以", "因而", "从而", "于是", "故", "因为", "由于", "基于", "鉴于",
    "如果", "若", "假设", "假定", "设", "则", "那么", "即", "也就是说", "换言之",
    "首先", "其次", "再次", "最后", "然后", "接着", "一方面", "另一方面", "此外", "另外",
    "同时", "然而", "但是", "不过", "尽管", "虽然", "显然", "可见", "由此", "综上所述",
    "总之", "根据", "依照", "按照", "不仅", "而且", "并且", "以及", "换言之",
    "确切地说", "具体而言", "同理", "类似地", "相应地", "反之", "相反", "相对而言",
    "特别地", "尤其", "尤其是",
]
MODELING_TERMS = [
    "模型", "建模", "算法", "优化", "预测", "分类", "聚类", "回归", "神经网络", "决策树",
    "随机森林", "支持向量机", "AHP", "TOPSIS", "熵权", "灰色关联", "模糊", "主成分",
    "线性规划", "整数规划", "动态规划", "蒙特卡洛", "模拟", "遗传算法", "粒子群", "蚁群",
    "深度学习", "机器学习", "训练", "测试", "验证", "交叉验证", "拟合", "泛化",
    "目标函数", "约束条件", "决策变量", "参数", "权重", "灵敏度", "鲁棒性", "收敛",
    "迭代", "求解",
]
REF_PATTERNS = [r"\[\d+\]", r"\[\d+[-–]\d+\]", r"\[\d+(?:,\d+)*\]"]
SECTION_PATTERNS = [r"[一二三四五六七八九十]、", r"（[一二三四五六七八九十]）",
                    r"\d+\.\d+", r"\d+\.", r"第[一二三四五六七八九十\d]+章", r"第[一二三四五六七八九十\d]+节"]

def analyze_sentences(text):
    sentences = re.split(r"[。！？；\n]+", text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
    if not sentences:
        return 0, 0, 0
    lengths = [len(s) for s in sentences]
    return len(sentences), float(np.mean(lengths)), float(np.std(lengths))

def terminology_richness(text):
    chinese_words = re.findall(r"[一-鿿]{2,}", text)
    if not chinese_words:
        return 0.0
    return len(set(chinese_words)) / len(chinese_words)

def detect_ai_patterns(text):
    sentences = re.split(r"[。！？\n]+", text)
    s_starts = [s.strip()[:8] for s in sentences if len(s.strip()) > 10]
    if len(s_starts) < 2:
        return 0.0, 0.0, 0.0
    start_counter = Counter(s_starts)
    repeated = sum(v - 1 for v in start_counter.values() if v > 1)
    rep_ratio = repeated / len(s_starts)
    sent_lengths = [len(s) for s in re.split(r"[。！？\n]+", text) if len(s.strip()) > 3]
    if len(sent_lengths) < 2:
        burstiness = 0.0
    else:
        m = np.mean(sent_lengths); burstiness = float(np.std(sent_lengths) / m) if m > 0 else 0.0
    paragraphs = text.split("\n\n")
    para_lengths = [len(p) for p in paragraphs if len(p.strip()) > 20]
    if len(para_lengths) < 2:
        para_uniformity = 0.0
    else:
        m = np.mean(para_lengths); para_uniformity = float(np.std(para_lengths) / m) if m > 0 else 0.0
    return rep_ratio, burstiness, para_uniformity

def paper_features(name, full, page_count):
    """按参考公式计算单篇的 22 列特征（full 为剔除代码页后的正文）。"""
    text = full
    chinese_chars = len(re.findall(r"[一-鿿]", text))
    english_words = len(re.findall(r"[a-zA-Z]+", text))
    total_chars = len(text)
    logical_connectors = sum(text.count(w) for w in LOGICAL_CONNECTORS)
    modeling_terms = sum(text.count(w) for w in MODELING_TERMS)
    references = sum(len(re.findall(p, text)) for p in REF_PATTERNS)
    equations = len(quantify.detect_equation_lines(text)[0])   # 我们的行级公式检测（修正其恒为0的问题）
    section_count = sum(len(re.findall(p, text)) for p in SECTION_PATTERNS)
    sent_count, avg_len, std_len = analyze_sentences(text)
    term_rich = terminology_richness(text)
    rep_ratio, burstiness, para_uniformity = detect_ai_patterns(text)
    chars_per_page = chinese_chars / page_count if page_count > 0 else 0
    logical_density = logical_connectors / (chinese_chars / 1000) if chinese_chars > 0 else 0
    formula_density = equations / page_count if page_count > 0 else 0
    ref_density = references / page_count if page_count > 0 else 0
    model_term_density = modeling_terms / (chinese_chars / 1000) if chinese_chars > 0 else 0
    return {
        "paper_id": name, "page_count": page_count, "chinese_chars": chinese_chars,
        "english_words": english_words, "total_chars": total_chars,
        "logical_connectors": logical_connectors, "modeling_terms": modeling_terms,
        "references": references, "equations": equations, "section_count": section_count,
        "sentence_count": sent_count, "avg_sentence_length": avg_len, "std_sentence_length": std_len,
        "terminology_richness": term_rich, "chars_per_page": chars_per_page,
        "logical_density": logical_density, "formula_density": formula_density,
        "ref_density": ref_density, "model_term_density": model_term_density,
        "ai_repetition_ratio": rep_ratio, "ai_burstiness": burstiness,
        "ai_para_uniformity": para_uniformity,
    }

def run():
    rows = []
    for att, prefix in [(C.ATT1, "A1_"), (C.ATT2, "A2_"), (C.ATT3, "A3_")]:
        for name in sorted(os.listdir(att)):
            if not name.lower().endswith(".pdf"):
                continue
            pages, full = C.extract_pdf(os.path.join(att, name))
            if C.is_image_only(full):
                continue  # 与参考一致：剔除扫描件（A1_25、A2_2-8）
            body_pages, body_full = quantify.strip_code_pages(pages)
            body_full = "\n\n".join(body_pages)   # 以页为段落块，使 ai_para_uniformity 有区分度
            paper_id = prefix + name.replace(".pdf", "")
            rows.append(paper_features(paper_id, body_full, len(pages)))
    df = pd.DataFrame(rows)
    out = os.path.join(C.OUT, "experiments", "all_paper_features_ours.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("已生成我们的特征矩阵:", out, "| 行数:", len(df))
    print(df.head(3).to_string())
    return df

if __name__ == "__main__":
    run()