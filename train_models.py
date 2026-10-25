"""
训练并保存所有模型，供 app.py 预测服务加载
=============================================
运行: python train_models.py
前提: 先运行 src/module1_preprocessing.py（生成 data/processed/）
输出: models/ 目录下 7 个文件
      ridge_model.pkl   — Ridge 岭回归（α=100.0）
      mlp_model.pkl     — MLP 神经网络（64→32→16）
      dt_model.pkl      — CART 决策树（max_depth=None）
      rf_model.pkl      — 随机森林（n_estimators=200）
      reg_meta.pkl      — 回归元数据（35特征名 + 19类型名）
      cls_meta.pkl      — 分类元数据（22特征名 + 17类别名）
      refs.pkl          — 参考值（年龄范围、职业列表等）
"""
import os, sys, pickle, joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import (PROCESSED_DATA_DIR, RANDOM_SEED,
                    MLP_HIDDEN_LAYERS, MLP_EPOCHS, MLP_BATCH_SIZE)

os.makedirs('models', exist_ok=True)

# ============================================================
# 1. 加载模块一预处理好的数据
# ============================================================
print("[1/4] 加载预处理数据...")
data = np.load(os.path.join(PROCESSED_DATA_DIR, "split_data.npz"))
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

with open(os.path.join(PROCESSED_DATA_DIR, "metadata.pkl"), "rb") as f:
    meta = pickle.load(f)
feature_names_reg = meta['feature_names']  # 35 个特征名（保证顺序）
genre_names = meta['genre_names']           # 19 个电影类型名
movies_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "movies_df.pkl"))
users_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "users_df.pkl"))
full_df = pd.read_pickle(os.path.join(PROCESSED_DATA_DIR, "full_df.pkl"))

# ============================================================
# 2. 训练回归模型（Ridge + MLP）
# ============================================================
print("[2/4] 训练回归模型...")

# MLP 用训练+验证合并训练（内部自动分 15% 做验证），Ridge 只用训练集
Xf = np.vstack([X_train, X_val])  # 纵向堆叠 [n_train+n_val, 35]
yf = np.hstack([y_train, y_val])

# Ridge 岭回归：Loss = MSE + α·Σw²，α=100.0 为 5 折交叉验证选出的最优值
ridge = Ridge(alpha=100.0, random_state=RANDOM_SEED)
ridge.fit(X_train, y_train)

# MLP 神经网络：35→64→32→16→1，ReLU+Adam+EarlyStopping
mlp = MLPRegressor(
    hidden_layer_sizes=tuple(MLP_HIDDEN_LAYERS),  # (64, 32, 16) 三层瓶颈式
    activation='relu',           # f(x)=max(0,x)，缓解梯度消失
    solver='adam',               # 自适应学习率优化器
    batch_size=MLP_BATCH_SIZE,   # 每批 64 样本更新权重
    max_iter=MLP_EPOCHS,         # 最多 100 轮
    learning_rate='adaptive',    # loss 不降时学习率自动减半
    early_stopping=True,         # 验证 loss 连续 10 轮不降则停
    validation_fraction=0.15,    # 内部 15% 做验证集
    n_iter_no_change=10,         # 早停耐心值
    random_state=RANDOM_SEED
)
mlp.fit(Xf, yf)

# 保存模型 + 元数据
joblib.dump(ridge, 'models/ridge_model.pkl')
joblib.dump(mlp, 'models/mlp_model.pkl')
joblib.dump({'feature_names': feature_names_reg, 'genre_names': genre_names},
            'models/reg_meta.pkl')
print(f"  Ridge (R²={ridge.score(X_test, y_test):.4f}) + "
      f"MLP (R²={mlp.score(X_test, y_test):.4f}) 已保存")

# ============================================================
# 3. 训练分类模型（决策树 + 随机森林）
# ============================================================
print("[3/4] 训练分类模型...")
from src.module3_classification import build_cls_data

# 构建分类专用数据集（标签 = 电影主类型，17 类）
ratings_df = full_df[['movie_id', 'user_id', 'rating', 'timestamp']].copy()
Xc, yc, class_names, feat_names_cls = build_cls_data(
    movies_df, ratings_df, genre_names)

# 移除样本数 < 3 的稀有类别（无法分层采样）
rare = {c for c, cnt in Counter(yc).items() if cnt < 3}
if rare:
    keep = np.array([yi not in rare for yi in yc])
    Xc, yc = Xc[keep], yc[keep]
    remaining = sorted(set(yc))                        # 保留类别的旧标签（升序）
    mapper = {o: n for n, o in enumerate(remaining)}
    yc = np.array([mapper[yi] for yi in yc])
    class_names = [class_names[o] for o in remaining]  # 同步更新类别名，避免与标签错位

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=RANDOM_SEED)

# CART 决策树：Gini = 1-Σpₖ²，max_depth=None 不限深度
dt = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_SEED)
dt.fit(Xc_tr, yc_tr)

# 随机森林：200 棵树 Bagging 集成，Bootstrap 采样 + 随机特征选择 + 投票
rf = RandomForestClassifier(
    n_estimators=200,      # 200 棵树
    random_state=RANDOM_SEED,
    n_jobs=-1              # 并行训练，利用所有 CPU 核心
)
rf.fit(Xc_tr, yc_tr)

joblib.dump(dt, 'models/dt_model.pkl')
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump({'feature_names': feat_names_cls, 'class_names': class_names,
             'genre_names': genre_names}, 'models/cls_meta.pkl')
print(f"  DT (acc={dt.score(Xc_te, yc_te):.4f}) + "
      f"RF (acc={rf.score(Xc_te, yc_te):.4f}) 已保存")

# ============================================================
# 4. 保存数据集参考值（供前端表单默认值使用）
# ============================================================
print("[4/4] 保存参考数据...")
refs = {
    'age_mean': float(users_df['age'].mean()),
    'age_min': int(users_df['age'].min()),
    'age_max': int(users_df['age'].max()),
    'occupations': sorted(users_df['occupation'].dropna().unique().tolist()),
    'movie_year_min': 1920,
    'movie_year_max': 2000,
    'movie_year_mean': 1990,
    'genre_list': genre_names,
    'avg_user_avg_rating': float(full_df['user_avg_rating'].mean()),
    'avg_movie_avg_rating': float(full_df['movie_avg_rating'].mean()),
}
joblib.dump(refs, 'models/refs.pkl')

print("\n全部模型已保存到 models/ 目录")
print("运行: python app.py  启动预测服务")
