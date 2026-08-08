#!/usr/bin/env python3
"""
参考论文 p4_ai_optimization.py 的复制（沿用其算法），仅做环境适配：
- 输入特征矩阵改为 ours（output/experiments/all_paper_features_ours.csv）；
- 移除未使用的 sklearn 导入与 p1 结果读取；输出目录改到 output/experiments。
"""
import numpy as np
import pandas as pd
import os
import json
from collections import Counter
# (移除未使用的 sklearn 导入)

# ============================================================
# Configuration
# ============================================================
INPUT_CSV = r"C:\Users\huangxiyan\.codex\worktrees\d16a\选题A\output\experiments\all_paper_features_ours.csv"
P1_RESULTS = r"C:\Users\huangxiyan\.codex\worktrees\d16a\选题A\output\problem1_indicators.csv"  # 未使用，仅占位
P1_WEIGHTS = r"C:\Users\huangxiyan\.codex\worktrees\d16a\选题A\output\problem1_minmax.json"  # 未使用，仅占位
OUTPUT_DIR = r"C:\Users\huangxiyan\.codex\worktrees\d16a\选题A\output\experiments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("PROBLEM 3: AI Detection, Logical Gap Analysis & Optimization")
print("=" * 70)

# ============================================================
# 1. Load data
# ============================================================
df_all = pd.read_csv(INPUT_CSV)
# p1_results = pd.read_csv(P1_RESULTS)  # 未使用，注释掉
# p1_weights = pd.read_csv(P1_WEIGHTS)  # 未使用，注释掉

# Filter Attachment 3 papers
df_a3 = df_all[df_all['paper_id'].str.startswith('A3_')].copy()
print(f"\nLoaded {len(df_a3)} Attachment 3 papers (medium quality, to be optimized)")

paper_ids = df_a3['paper_id'].values

# ============================================================
# 2. AI-Generated Content Detection
# ============================================================
print("\n" + "=" * 70)
print("Step 1: AI-Generated Content Detection")
print("=" * 70)

def compute_ai_indicators(df):
    """
    Compute comprehensive AI detection indicators:
    1. Sentence pattern repetition (repetition_ratio)
    2. Burstiness (sentence length variation)
    3. Paragraph uniformity (structural monotony)
    4. Logical connector consistency
    5. Terminology redundancy
    """
    indicators = {}

    # 1. AI Repetition Score: higher = more AI-like pattern repetition
    indicators['ai_rep_score'] = df['ai_repetition_ratio'].values

    # 2. Burstiness: AI text tends to have low burstiness (uniform sentences)
    # Normal burstiness for human writing is ~0.8-1.2
    indicators['burstiness_score'] = 1.0 / (np.abs(df['ai_burstiness'].values - 0.7) + 0.3)

    # 3. Paragraph uniformity: AI text has highly uniform paragraph lengths
    indicators['para_uniformity_score'] = df['ai_para_uniformity'].values

    # 4. Logical connector density anomaly
    log_density = df['logical_density'].values
    log_mean = df_all[df_all['paper_id'].str.startswith('A1_')]['logical_density'].mean()
    indicators['logic_anomaly'] = np.abs(log_density - log_mean) / (log_mean + 1e-10)

    # 5. Terminology richness deviation from mean
    term_rich = df['terminology_richness'].values
    term_mean = df_all[df_all['paper_id'].str.startswith('A1_')]['terminology_richness'].mean()
    indicators['term_richness_dev'] = np.abs(term_rich - term_mean) / (term_mean + 1e-10)

    return indicators

ai_indicators = compute_ai_indicators(df_a3)

# Compute composite AI score (higher = more likely AI-generated)
ai_keys = ['ai_rep_score', 'burstiness_score', 'para_uniformity_score',
           'logic_anomaly', 'term_richness_dev']

# Normalize each indicator to [0, 1]
ai_normalized = {}
for key in ai_keys:
    vals = ai_indicators[key]
    ai_normalized[key] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)

# Weighted composite AI score
ai_weights_dict = {
    'ai_rep_score': 0.30,
    'burstiness_score': 0.25,
    'para_uniformity_score': 0.20,
    'logic_anomaly': 0.15,
    'term_richness_dev': 0.10,
}

ai_composite = np.zeros(len(df_a3))
for key, w in ai_weights_dict.items():
    ai_composite += w * ai_normalized[key]

# Categorize AI involvement level
def categorize_ai(score):
    if score < 0.3:
        return "低(Low) - 主要为人工撰写"
    elif score < 0.5:
        return "中低(Medium-Low) - 部分AI辅助"
    elif score < 0.7:
        return "中高(Medium-High) - 较多AI参与"
    else:
        return "高(High) - 可能存在大量AI生成内容"

for i, (pid, score) in enumerate(zip(paper_ids, ai_composite)):
    print(f"\n  {pid}: AI Score = {score:.4f}")
    print(f"    → {categorize_ai(score)}")
    print(f"    句子重复度: {ai_normalized['ai_rep_score'][i]:.3f}")
    print(f"    句式变化性: {ai_normalized['burstiness_score'][i]:.3f}")
    print(f"    段落均匀度: {ai_normalized['para_uniformity_score'][i]:.3f}")
    print(f"    逻辑异常度: {ai_normalized['logic_anomaly'][i]:.3f}")

# ============================================================
# 3. Logical Gap Identification
# ============================================================
print("\n" + "=" * 70)
print("Step 2: Logical Gap Identification and Analysis")
print("=" * 70)

def identify_logical_gaps(df):
    """
    Identify logical gaps using quantitative indicators:
    1. Transition completeness: ratio of logical connectors to sentences
    2. Argument depth: modeling terms per logical argument
    3. Evidence density: references and equation support
    4. Structure coherence: section-to-content alignment
    """
    gaps = {}

    # 1. Transition Completeness Score
    # Low logical_density may indicate insufficient transitions
    all_log_mean = df_all['logical_density'].mean()
    gaps['transition_gap'] = np.maximum(0, all_log_mean - df['logical_density'].values) / (all_log_mean + 1e-10)

    # 2. Argument Support Gap
    # Low ref_density combined with high modeling terms → unsupported claims
    ref_model_ratio = df['ref_density'].values / (df['model_term_density'].values + 1e-10)
    all_rm_ratio_mean = (df_all['ref_density'] / (df_all['model_term_density'] + 1e-10)).mean()
    gaps['evidence_gap'] = np.maximum(0, all_rm_ratio_mean - ref_model_ratio) / (all_rm_ratio_mean + 1e-10)

    # 3. Structure-Content Coherence Gap
    # High section_count but low chars_per_page → shallow content
    structure_ratio = df['section_count'].values / (df['chars_per_page'].values + 1e-10)
    gaps['structure_content_gap'] = np.abs(structure_ratio - structure_ratio.mean()) / (structure_ratio.std() + 1e-10)

    # 4. Terminology Consistency Gap
    # Low terminology_richness → repetitive vocabulary
    tr_mean = df_all['terminology_richness'].mean()
    gaps['vocabulary_gap'] = np.maximum(0, tr_mean - df['terminology_richness'].values) / (tr_mean + 1e-10)

    # 5. Formula-Rigor Gap
    # Low formula_density → insufficient mathematical formalization
    fd_mean = df_all[df_all['paper_id'].str.startswith('A1_')]['formula_density'].mean()
    gaps['formula_gap'] = np.maximum(0, fd_mean - df['formula_density'].values) / (fd_mean + 1e-10)

    return gaps

gap_indicators = identify_logical_gaps(df_a3)

gap_keys = ['transition_gap', 'evidence_gap', 'structure_content_gap',
            'vocabulary_gap', 'formula_gap']
gap_names = {
    'transition_gap': '逻辑过渡缺失',
    'evidence_gap': '论据支撑不足',
    'structure_content_gap': '结构-内容不匹配',
    'vocabulary_gap': '术语丰富度不足',
    'formula_gap': '数学形式化不足',
}

for i, pid in enumerate(paper_ids):
    print(f"\n  {pid} Logical Gap Analysis:")
    for key in gap_keys:
        val = gap_indicators[key][i]
        severity = "严重" if val > 0.5 else "中等" if val > 0.2 else "轻微"
        bar = "█" * int(val * 20)
        print(f"    {gap_names[key]:20s}: {val:.4f} [{severity}] {bar}")

# ============================================================
# 4. Comprehensive Optimization Diagnosis
# ============================================================
print("\n" + "=" * 70)
print("Step 3: Comprehensive Optimization Diagnosis")
print("=" * 70)

# Load P1 weights to understand which dimensions matter most
dimension_weights = {
    "C1_结构完整性": 0.1599,
    "C2_方法严谨性": 0.4185,
    "C3_逻辑连贯性": 0.2625,
    "C4_学术规范性": 0.0973,
    "C5_表达质量": 0.0618,
}

# Compute dimension scores for each paper
feature_names = [
    "section_count", "page_count", "chars_per_page",    # C1
    "formula_density", "model_term_density", "modeling_terms",  # C2
    "logical_density", "logical_connectors", "avg_sentence_length",  # C3
    "ref_density", "references", "terminology_richness",    # C4
    "chinese_chars", "sentence_count", "english_words",     # C5
]

dim_indices = {
    "C1_结构完整性": [0, 1, 2],
    "C2_方法严谨性": [3, 4, 5],
    "C3_逻辑连贯性": [6, 7, 8],
    "C4_学术规范性": [9, 10, 11],
    "C5_表达质量": [12, 13, 14],
}

# Get feature values for A3 papers
X_a3 = df_a3[feature_names].values

# Normalize using all papers' stats
X_all = df_all[feature_names].values
X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_a3_norm = (X_a3 - X_min) / (X_max - X_min + 1e-10)

# Compute dimension scores
dim_scores_a3 = {}
for dim_name, indices in dim_indices.items():
    dim_score = X_a3_norm[:, indices].mean(axis=1)
    dim_scores_a3[dim_name] = dim_score

# Generate optimization recommendations
print("\nOptimization Recommendations:")
print("-" * 70)

optimization_plans = {}

for i, pid in enumerate(paper_ids):
    print(f"\n{'='*60}")
    print(f"  Paper {pid} - Optimization Plan")
    print(f"{'='*60}")

    plan = {
        "paper_id": str(pid),
        "ai_involvement": float(ai_composite[i]),
        "ai_category": categorize_ai(ai_composite[i]),
        "dimension_scores": {},
        "recommendations": [],
        "expected_improvements": {},
    }

    # AI detection findings
    if ai_composite[i] > 0.5:
        print(f"  ⚠ AI检测: AI参与度较高 ({ai_composite[i]:.2f})")
        print(f"    建议: 增加人工润色，提高句式多样性，减少模板化表达")
        plan["recommendations"].append({
            "area": "AI检测",
            "priority": "高",
            "issue": "AI生成痕迹明显",
            "suggestion": "增加人工润色，提高句式多样性，减少模板化表达，添加个性化分析",
            "expected_improvement": "+8~12分",
        })

    # Dimension analysis
    for dim_name, score in dim_scores_a3.items():
        plan["dimension_scores"][dim_name] = float(score[i])
        if score[i] < 0.3:
            plan["expected_improvements"][dim_name] = "+10~15分"
        elif score[i] < 0.5:
            plan["expected_improvements"][dim_name] = "+5~8分"
        else:
            plan["expected_improvements"][dim_name] = "+1~3分"

    # Find weakest dimensions
    weakest_dims = sorted(dim_scores_a3.items(),
                         key=lambda x: x[1][i])[:3]

    print(f"\n  维度得分诊断:")
    for dim_name, score in weakest_dims:
        weight = dimension_weights[dim_name]
        print(f"    {dim_name}: {score[i]:.3f} (权重: {weight:.2f}) ★ 需优化")

    # Logical gap fixes
    print(f"\n  逻辑完整性修补建议:")
    for key in gap_keys:
        val = gap_indicators[key][i]
        if val > 0.3:
            suggestion = ""
            if key == 'transition_gap':
                suggestion = "增加段落间过渡句，使用'因此'、'从而'、'综上所述'等逻辑连接词，确保论证层层递进"
            elif key == 'evidence_gap':
                suggestion = "为每个核心论点补充文献引用或数据支撑，提升论据密度"
            elif key == 'structure_content_gap':
                suggestion = "调整章节结构，确保每个章节内容充分展开，避免标题多而内容空"
            elif key == 'vocabulary_gap':
                suggestion = "丰富学术术语使用，避免同义词重复，增加专业术语的多样性"
            elif key == 'formula_gap':
                suggestion = "增加数学公式推导，将定性描述转化为定量模型表达"

            print(f"    [{gap_names[key]}]: {suggestion}")
            plan["recommendations"].append({
                "area": "逻辑修补",
                "priority": "高" if val > 0.5 else "中",
                "issue": gap_names[key],
                "suggestion": suggestion,
                "expected_improvement": f"+{val*10:.0f}~{val*15:.0f}分",
            })

    # Specific optimization score estimates
    ai_penalty = ai_composite[i] * 15  # AI involvement penalty
    logic_penalty = np.mean([gap_indicators[k][i] for k in gap_keys]) * 20

    baseline = 50  # Assume baseline medium quality
    potential_improvement = ai_penalty + logic_penalty
    optimized_score = min(100, baseline + potential_improvement)

    print(f"\n  📊 优化预测:")
    print(f"    当前预估分数: ~{baseline - ai_penalty/2:.0f}分")
    print(f"    AI检测优化潜力: +{ai_penalty:.1f}分")
    print(f"    逻辑修补优化潜力: +{logic_penalty:.1f}分")
    print(f"    优化后预估分数: ~{optimized_score:.0f}分")

    plan["predicted_current"] = float(baseline - ai_penalty/2)
    plan["predicted_optimized"] = float(optimized_score)
    plan["improvement_potential"] = float(potential_improvement)

    optimization_plans[str(pid)] = plan

# ============================================================
# 5. AI-Assistance Degree Assessment
# ============================================================
print("\n" + "=" * 70)
print("Step 4: AI-Assistance Degree Assessment")
print("=" * 70)

# AI assistance categorization based on multiple indicators
def assess_ai_assistance(df, ai_score):
    """
    Assess the degree of AI assistance for each paper.
    Categories:
    - Independent (human-dominant)
    - AI-assisted (AI as tool)
    - AI-augmented (substantial AI contribution)
    - AI-generated (AI-dominant)
    """
    assessments = []

    for i in range(len(df)):
        s = ai_score[i]

        # Consider multiple signals
        signals = {
            "AI重复度": min(1.0, df['ai_repetition_ratio'].values[i] * 3),
            "句式均匀度": min(1.0, 1.0 / (df['ai_burstiness'].values[i] + 0.3)),
            "段落一致性": min(1.0, df['ai_para_uniformity'].values[i] * 2),
            "术语集中度": min(1.0, 1.0 - df['terminology_richness'].values[i]),
        }

        if s < 0.3:
            category = "独立撰写(Independent)"
            ai_degree = np.round(s * 100, 1)
            description = "论文主要体现人工撰写特征，AI参与度较低"
        elif s < 0.5:
            category = "AI辅助(AI-Assisted)"
            ai_degree = np.round(s * 100, 1)
            description = "论文存在AI辅助痕迹，但主体为人工完成"
        elif s < 0.7:
            category = "AI增强(AI-Augmented)"
            ai_degree = np.round(s * 100, 1)
            description = "论文中AI贡献较大，需注意原创性和个性化表达"
        else:
            category = "AI主导(AI-Dominant)"
            ai_degree = np.round(s * 100, 1)
            description = "论文可能存在大量AI生成内容，需大幅人工重写"

        assessments.append({
            "paper_id": df['paper_id'].values[i],
            "ai_category": category,
            "ai_percentage": ai_degree,
            "description": description,
            "detailed_signals": {k: round(float(v), 4) for k, v in signals.items()},
        })

        print(f"\n  {df['paper_id'].values[i]}:")
        print(f"    类别: {category}")
        print(f"    AI度: {ai_degree}%")
        print(f"    {description}")

    return assessments

assessments = assess_ai_assistance(df_a3, ai_composite)

# ============================================================
# 6. Save Results
# ============================================================
print("\n" + "=" * 70)
print("Saving Results")
print("=" * 70)

# Save optimization plans
with open(os.path.join(OUTPUT_DIR, "p3_optimization_plans.json"), "w", encoding="utf-8") as f:
    json.dump(optimization_plans, f, ensure_ascii=False, indent=2)

# Save AI assessments
with open(os.path.join(OUTPUT_DIR, "p3_ai_assessments.json"), "w", encoding="utf-8") as f:
    json.dump(assessments, f, ensure_ascii=False, indent=2)

# Save logical gap analysis
gap_df = pd.DataFrame({
    "paper_id": paper_ids,
    **{gap_names[k]: np.round(gap_indicators[k], 4) for k in gap_keys},
})
gap_df.to_csv(os.path.join(OUTPUT_DIR, "p3_logical_gaps.csv"), index=False, encoding="utf-8-sig")

# Save dimension scores
dim_df = pd.DataFrame({
    "paper_id": paper_ids,
    **{dim_name: np.round(dim_scores_a3[dim_name], 4) for dim_name in dimension_weights.keys()},
})
dim_df.to_csv(os.path.join(OUTPUT_DIR, "p3_dimension_scores.csv"), index=False, encoding="utf-8-sig")

# Save AI detection summary
ai_summary = pd.DataFrame(assessments)
ai_summary.to_csv(os.path.join(OUTPUT_DIR, "p3_ai_summary.csv"), index=False, encoding="utf-8-sig")

print("All Problem 3 results saved.")
print("\n✓ Problem 3 Complete!")
