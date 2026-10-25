"""
模块三：电影类型分类 — 分类任务
==========================================
运行: python src/module3_classification.py
前提: 先运行 module1_preprocessing.py
对比: CART 决策树 (单棵,基尼系数) vs 随机森林 (Bagging集成,200棵树)
评估: 准确率 / 精确率 / 召回率 / F1 / 混淆矩阵
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from config import (DT_MAX_DEPTH_LIST, RF_N_ESTIMATORS_LIST, RANDOM_SEED)
from utils import (print_section, print_subsection, show_plot,
                   evaluate_classification, load_processed_data)
# 注意: 中文字体配置已在 utils.py 全局设置，无需重复


def build_cls_data(movies_df, ratings_df, genre_names):
    """
    构建分类数据集: 电影特征(22维) → 预测主要类型(17类)

    标签策略: 一部电影可能属于多个类型(如 Action|Comedy)，
    取第一个非 unknown 的类型作为主类型做单标签多分类。
    """
    print_subsection("构建分类数据集")

    # 从评分表聚合电影统计特征（平均评分/评分人数/评分标准差）
    ms = ratings_df.groupby('movie_id')['rating'].agg(['mean', 'count', 'std'])
    ms.columns = ['avg_rating', 'num_ratings', 'std_rating']
    ms['std_rating'] = ms['std_rating'].fillna(0)

    df = movies_df.merge(ms, on='movie_id', how='inner')
    df['movie_year'] = df['release_date'].apply(
        lambda x: int(x.split('-')[-1]) if pd.notna(x) and x != '' else 1990)

    # 取第一个非 unknown 的类型作为标签
    def primary(row):
        for g in genre_names:
            if g != 'unknown' and row.get(g, 0) == 1:
                return g
        return 'unknown'

    df['genre'] = df.apply(primary, axis=1)
    df = df[df['genre'] != 'unknown'].copy()  # 移除无法确定类型的电影

    # 特征列: 4个统计特征 + 18个类型标志
    feat_cols = ['movie_year', 'avg_rating', 'num_ratings', 'std_rating']
    for g in genre_names:
        if g != 'unknown' and g in df.columns:
            feat_cols.append(g)

    X = df[feat_cols].values.astype(np.float32)
    y = LabelEncoder().fit_transform(df['genre'])  # 字符串标签→整数
    class_names = sorted(df['genre'].unique())
    print(f"  样本: {len(df)}  类别: {len(class_names)}  特征: {len(feat_cols)}")
    return X, y, class_names, feat_cols


def run():
    print_section("模块三：电影类型分类 — 决策树 vs 随机森林")

    # ---- 加载数据 ----
    movies_df = load_processed_data("movies_df")
    metadata = load_processed_data("metadata")
    genre_names = metadata["genre_names"]
    full_df = load_processed_data("full_df")
    ratings_df = full_df[['movie_id', 'user_id', 'rating', 'timestamp']].copy()

    # ---- 构建分类数据 ----
    X, y, class_names, feat_names = build_cls_data(movies_df, ratings_df, genre_names)

    # ---- 移除样本数<3的稀有类别（无法分层采样） ----
    rare = {c for c, cnt in Counter(y).items() if cnt < 3}
    if rare:
        keep = np.array([yi not in rare for yi in y])
        X, y = X[keep], y[keep]
        remaining = sorted(set(y))                        # 保留类别的旧标签（升序）
        mapper = {o: n for n, o in enumerate(remaining)}
        y = np.array([mapper[yi] for yi in y])
        class_names = [class_names[o] for o in remaining]  # 同步更新类别名，避免与标签错位
        print(f"  移除稀有类后: {len(set(y))} 类, {len(y)} 样本")

    # ---- 数据集划分（30%测试，分层采样保持类别比例） ----
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=RANDOM_SEED)
    print(f"  训练集: {len(X_tr)}  测试集: {len(X_te)}")

    # ================================================================
    # 模型一：CART 决策树
    # 基尼系数 Gini = 1 - Σpₖ²，pₖ 为第k类占比。Gini=0 最纯，Gini≈1 最杂。
    # max_depth 控制树深度，交叉验证选最优值。
    # ================================================================
    print_subsection("模型一：CART 决策树")
    print("  原理: 基于基尼系数(Gini Impurity)递归二分分裂，构建树形分类规则")

    grid = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=RANDOM_SEED),
                        {'max_depth': DT_MAX_DEPTH_LIST}, cv=5, scoring='accuracy')
    grid.fit(X_tr, y_tr)
    bd = grid.best_params_['max_depth']
    bd_str = str(bd) if bd is not None else '不限深度'
    print(f"  最优 max_depth = {bd_str}")
    for d, s in zip(DT_MAX_DEPTH_LIST, grid.cv_results_['mean_test_score']):
        d_str = str(d) if d else '不限'
        print(f"    max_depth={d_str:4s}  →  交叉验证准确率 = {s:.4f}"
              f"{'  ★ 最优' if d == bd else ''}")

    dt = grid.best_estimator_
    dt_m = evaluate_classification(y_te, dt.predict(X_te))  # 复用 utils 评估函数

    # ================================================================
    # 模型二：随机森林 (Bagging 集成)
    # 1. Bootstrap 有放回采样 → N个不同训练子集
    # 2. 每子集训练一棵决策树（分裂时随机选特征 → 更多样性）
    # 3. 200棵树投票 → 得票最多者为最终分类
    # ================================================================
    print_subsection("模型二：随机森林 (Bagging 集成)")
    print("  原理: Bootstrap采样 → 训练多棵决策树 → 投票决定最终分类")

    # n_estimators 对比实验：观察不同树数量下的精度变化
    # 采样多个 n 值展示收敛趋势
    n_list = RF_N_ESTIMATORS_LIST
    train_sc, test_sc = [], []
    for n in n_list:
        rf = RandomForestClassifier(n_estimators=n, random_state=RANDOM_SEED, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        train_sc.append(rf.score(X_tr, y_tr))
        test_sc.append(rf.score(X_te, y_te))
        print(f"  n_estimators={n:3d}  →  训练准确率={train_sc[-1]:.4f}  "
              f"测试准确率={test_sc[-1]:.4f}")

    # 最终模型：n=200
    rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    rf_m = evaluate_classification(y_te, rf.predict(X_te))  # 复用 utils 评估函数

    # ================================================================
    # 模型对比表
    # ================================================================
    print("\n  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║              分类模型对比 (测试集)                             ║")
    print("  ╠══════════════╦══════════╦══════════╦══════════╦══════════════╣")
    print("  ║ 模型          ║ 准确率    ║ 精确率    ║ 召回率    ║ F1-Score     ║")
    print("  ╠══════════════╬══════════╬══════════╬══════════╬══════════════╣")
    print(f"  ║ CART 决策树   ║ {dt_m['Accuracy']:.4f}   ║ {dt_m['Precision']:.4f}   ║ "
          f"{dt_m['Recall']:.4f}   ║ {dt_m['F1']:.4f}     ║")
    print(f"  ║ 随机森林(n=200)║ {rf_m['Accuracy']:.4f}   ║ {rf_m['Precision']:.4f}   ║ "
          f"{rf_m['Recall']:.4f}   ║ {rf_m['F1']:.4f}     ║")
    print("  ╚══════════════╩══════════╩══════════╩══════════╩══════════════╝")

    # 判定最优模型
    if rf_m['Accuracy'] >= dt_m['Accuracy']:
        print(f"\n  ★ 最优模型: 随机森林 (准确率={rf_m['Accuracy']:.4f})")
    else:
        print(f"\n  ★ 最优模型: 决策树 (准确率={dt_m['Accuracy']:.4f})")

    print(f"""
  【随机森林最优参数】
  ├─ n_estimators  : 200
  ├─ criterion     : gini (基尼系数)
  ├─ max_depth     : None (不限深度)
  ├─ bootstrap     : True (有放回采样)
  └─ 集成策略       : Bagging → 多数投票
  """)

    # ================================================================
    # 图1：n_estimators vs 准确率曲线
    # 蓝线=训练集，橙线=测试集，整体上升→收敛
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_list, train_sc, 'o-', color='#4472C4', lw=2, ms=8, label='训练集准确率')
    ax.plot(n_list, test_sc, 's-', color='#ED7D31', lw=2, ms=8, label='测试集准确率')
    ax.set_xlabel('决策树数量（棵）', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('随机森林：决策树数量对分类精度的影响', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11), ax.grid(True, alpha=0.3)
    # 放大 y 轴展示趋势（5棵树时已约97%，之后整体上升并收敛）
    ax.set_ylim(0.975, 1.005)
    plt.tight_layout()
    plt.savefig('presentation/figures/p_n_estimators.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_n_estimators.png")
    show_plot(fig)

    # ================================================================
    # 图2：特征重要性 Top 15
    # 基于各特征在分裂时减少的基尼系数总量排序
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    imp = rf.feature_importances_
    idx = np.argsort(imp)[-15:]  # 取最重要的15个特征的索引
    ax.barh(range(15), imp[idx][::-1],
            color=plt.cm.Blues(np.linspace(0.4, 0.95, 15))[::-1])
    ax.set_yticks(range(15))
    ax.set_yticklabels([feat_names[i] for i in idx][::-1])
    ax.set_xlabel('重要性得分', fontsize=12)
    ax.set_title('随机森林特征重要性（前 15 名）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('presentation/figures/p_feature_importance.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_feature_importance.png")
    show_plot(fig)

    # ================================================================
    # 图3：混淆矩阵热力图
    # 行=真实类别，列=预测类别，对角线=正确分类
    # ================================================================
    cm = confusion_matrix(y_te, rf.predict(X_te))
    # 只保留有样本的类别（避免空行/列）
    used = sorted(set(y_te) | set(rf.predict(X_te)))
    used_names = [class_names[i] if i < len(class_names) else str(i) for i in used]
    cm_used = cm[np.ix_(used, used)]  # 用高级索引取出子矩阵

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_used, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=used_names, yticklabels=used_names)
    ax.set_xlabel('预测类别', fontsize=12), ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title('随机森林混淆矩阵（测试集）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('presentation/figures/p_confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_confusion_matrix.png")
    show_plot(fig)

    print_section("模块三完成！")
    return rf


if __name__ == "__main__":
    os.makedirs('presentation/figures', exist_ok=True)
    run()
