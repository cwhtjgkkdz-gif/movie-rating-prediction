"""
工具函数模块 — 三个模块共享，消除重复代码
=============================================
提供: 全局中文配置 / 分节打印 / 图表弹窗 / 回归评估 / 分类评估 / 数据加载
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, accuracy_score, precision_score,
                              recall_score, f1_score)
from config import PROCESSED_DATA_DIR

# ============================================================
# 全局设置 — 导入本模块即生效，各模块无需重复配置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块


# ============================================================
# 控制台打印辅助
# ============================================================
def print_section(title):
    """打印主标题（60个等号分隔线）"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_subsection(title):
    """打印子标题（虚线前缀）"""
    print(f"\n  --- {title} ---")


# ============================================================
# 图表显示
# ============================================================
def show_plot(fig=None):
    """
    弹窗显示图表，用户关闭窗口后继续执行。
    tight_layout 自动调整子图间距，close('all') 释放内存。
    """
    if fig:
        plt.tight_layout()
    plt.show()
    plt.close('all')


# ============================================================
# 回归评估（模块二使用）
# ============================================================
def evaluate_regression(y_true, y_pred, model_name="Model"):
    """
    计算三个回归指标:
      MAE  = (1/n)Σ|y-ŷ|        平均绝对误差，单位同评分
      RMSE = √[(1/n)Σ(y-ŷ)²]    均方根误差，对大误差惩罚更重
      R²   = 1 - SSres/SStot    决定系数，1=完美, 0=均值水平
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n  [{model_name}] MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


# ============================================================
# 分类评估（模块三使用）
# ============================================================
def evaluate_classification(y_true, y_pred):
    """
    计算四个分类指标（weighted 平均，按各类样本数加权）:
      Accuracy  = 正确数/总数
      Precision = TP/(TP+FP)  预测为某类中真正属于该类的比例
      Recall    = TP/(TP+FN)  某类样本中被正确找出的比例
      F1        = 2*P*R/(P+R) 精确率和召回率的调和平均
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"    Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}


# ============================================================
# 共享数据加载 — 模块二/三复用，避免各自写加载逻辑
# ============================================================
def load_processed_data(what):
    """
    加载模块一保存的预处理数据。

    参数 what:
      "movies_df" → 电影信息 DataFrame
      "users_df"  → 用户信息 DataFrame
      "full_df"   → 完整合并+特征构造后的 DataFrame
      "split"     → (X_train, X_val, X_test, y_train, y_val, y_test) 六元组
      "metadata"  → dict (feature_names, genre_names, n_train 等)

    示例:
      X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data("split")
      movies_df = load_processed_data("movies_df")
    """
    if isinstance(what, str):
        what = [what]

    result = []
    for w in what:
        if w == "movies_df":
            result.append(pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "movies_df.pkl")))
        elif w == "users_df":
            result.append(pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "users_df.pkl")))
        elif w == "full_df":
            result.append(pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "full_df.pkl")))
        elif w == "split":
            data = np.load(os.path.join(PROCESSED_DATA_DIR, "split_data.npz"))
            result.append((data['X_train'], data['X_val'], data['X_test'],
                           data['y_train'], data['y_val'], data['y_test']))
        elif w == "metadata":
            with open(os.path.join(PROCESSED_DATA_DIR, "metadata.pkl"), "rb") as f:
                result.append(pickle.load(f))

    return result[0] if len(result) == 1 else tuple(result)
