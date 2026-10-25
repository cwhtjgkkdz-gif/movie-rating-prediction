"""
模块一：数据预处理与特征工程
==========================================
运行方式: python src/module1_preprocessing.py
作用: 加载 MovieLens 100K 数据集 → 数据探查 → 特征构造 →
      编码标准化 → 划分训练/验证/测试集 → 保存到 data/processed/
依赖: data_download.py（自动下载数据集）, config.py（参数）, utils.py（工具函数）
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder  # 标准化 / 标签编码
from sklearn.model_selection import train_test_split           # 数据集划分
from config import (RAW_DATA_DIR, DATASET_NAME, RANDOM_SEED,
                    TRAIN_SIZE, VAL_SIZE, TEST_SIZE, PROCESSED_DATA_DIR)
from data_download import download_movielens, generate_sample_data
from utils import print_section, print_subsection

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)  # 确保 processed 目录存在


def load_movielens_data():
    """
    加载 MovieLens 100K 数据集（首次运行自动下载）

    MovieLens 100K 文件说明:
      u.data  — 评分记录 (user_id, movie_id, rating, timestamp)  [tab分隔]
      u.item  — 电影信息 (movie_id, title, release_date, ..., 19个类型)  [|分隔]
      u.user  — 用户信息 (user_id, age, gender, occupation, zip_code)  [|分隔]

    Returns:
        ratings_df  : DataFrame, 100000条评分记录
        movies_df   : DataFrame, 1682部电影信息
        users_df    : DataFrame, 943位用户信息
        genre_names : list, 19种电影类型名称
    """
    dataset_dir = os.path.join(RAW_DATA_DIR, DATASET_NAME)

    # 尝试下载，失败则生成模拟数据（保证代码在任何环境可运行）
    if download_movielens() is None and not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(RAW_DATA_DIR, DATASET_NAME)
        generate_sample_data(dataset_dir)

    # 读取评分数据（tab分隔，无表头）
    ratings_df = pd.read_csv(os.path.join(dataset_dir, "u.data"),
                             sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    print(f"[数据] 评分: {ratings_df.shape[0]}条, "
          f"{ratings_df['user_id'].nunique()}用户, {ratings_df['movie_id'].nunique()}电影")

    # 19种电影类型（MovieLens 预定义）
    genre_names = ["unknown", "Action", "Adventure", "Animation", "Children's",
                   "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                   "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
                   "Sci-Fi", "Thriller", "War", "Western"]

    # 读取电影数据（| 分隔，latin-1 编码，前5列为基础信息，后19列为类型二值标志）
    movies_df = pd.read_csv(os.path.join(dataset_dir, "u.item"), sep='|',
                            names=['movie_id', 'title', 'release_date',
                                   'video_release_date', 'IMDb_URL'] + genre_names,
                            encoding='latin-1')

    # 读取用户数据（| 分隔）
    users_df = pd.read_csv(os.path.join(dataset_dir, "u.user"), sep='|',
                           names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

    print(f"[数据] 电影: {movies_df.shape[0]}部, 用户: {users_df.shape[0]}人")
    return ratings_df, movies_df, users_df, genre_names


def explore_data(ratings_df, movies_df, users_df):
    """
    数据探查 —— 打印基本统计信息，了解数据全貌

    探查内容:
      - 评分数量、均值、范围 → 了解评分分布特征
      - 缺失值数量 → 确认数据质量
      - 用户年龄范围、性别分布 → 了解用户画像
    """
    print_subsection("数据概况")
    print(f"  评分: {len(ratings_df)}条, 均值={ratings_df['rating'].mean():.2f}, "
          f"范围={ratings_df['rating'].min()}-{ratings_df['rating'].max()}")
    print(f"  缺失值 — ratings:{ratings_df.isnull().sum().sum()}, "
          f"movies:{movies_df.isnull().sum().sum()}, users:{users_df.isnull().sum().sum()}")
    print(f"  用户年龄: {users_df['age'].min()}~{users_df['age'].max()}, "
          f"性别 M:{sum(users_df['gender']=='M')} F:{sum(users_df['gender']=='F')}")


def build_features(ratings_df, movies_df, users_df, genre_names):
    """
    特征构造 —— 从原始字段中提取/衍生新特征（对应实验1核心内容）

    构造策略分为四类:
      1. 时间特征   — 从 timestamp 提取 星期几/是否周末/月份
      2. 统计特征   — 用户维度：平均评分/评分次数/评分标准差
                     电影维度：平均评分/评分人数/评分标准差
      3. 交互特征   — 用户评分均值 × 电影评分均值（捕捉用户-电影交互效应）
      4. 衍生特征   — 电影年龄 = 评分年份 - 上映年份

    Returns:
        df: 合并后的完整 DataFrame（41列，含所有原始列+构造列）
    """
    print_subsection("构造特征")

    # ---- 2.1 从上映日期提取年份和季节（用作电影特征） ----
    movies_df = movies_df.copy()
    # 年份: "01-Jan-1995" → 1995
    movies_df['movie_year'] = movies_df['release_date'].apply(
        lambda x: int(x.split('-')[-1]) if pd.notna(x) and x != '' else np.nan)
    movies_df['movie_year'] = movies_df['movie_year'].fillna(
        movies_df['movie_year'].median())  # 缺失用中位数填充
    # 季节: 1=春(3-5月) / 2=夏(6-8月) / 3=秋(9-11月) / 4=冬(12-2月)
    movies_df['movie_season'] = movies_df['release_date'].apply(
        lambda x: _season(x) if pd.notna(x) and x != '' else 0)

    # ---- 2.2 合并三张表 ----
    df = ratings_df.copy()
    # LEFT JOIN movies：每行评分 + 对应电影信息
    df = df.merge(movies_df[['movie_id', 'title', 'movie_year', 'movie_season'] + genre_names],
                  on='movie_id', how='left')
    # LEFT JOIN users：每行评分 + 对应用户信息
    df = df.merge(users_df[['user_id', 'age', 'gender', 'occupation']],
                  on='user_id', how='left')

    # ---- 2.3 时间特征（从评分时间戳提取） ----
    df['rating_datetime'] = pd.to_datetime(df['timestamp'], unit='s')  # Unix时间戳→日期
    df['rating_dayofweek'] = df['rating_datetime'].dt.dayofweek       # 0=周一, 6=周日
    df['rating_is_weekend'] = (df['rating_dayofweek'] >= 5).astype(int)  # 是否周末 (0/1)
    df['rating_month'] = df['rating_datetime'].dt.month               # 评分月份 (1-12)

    # ---- 2.4 用户维度统计特征（groupby + agg 聚合） ----
    us = df.groupby('user_id')['rating'].agg(['mean', 'count', 'std'])
    us.columns = ['user_avg_rating', 'user_rating_count', 'user_rating_std']
    df = df.merge(us, on='user_id', how='left')
    df['user_rating_std'] = df['user_rating_std'].fillna(0)  # 只评过1部的用户 std=NaN→0

    # ---- 2.5 电影维度统计特征 ----
    ms = df.groupby('movie_id')['rating'].agg(['mean', 'count', 'std'])
    ms.columns = ['movie_avg_rating', 'movie_rating_count', 'movie_rating_std']
    df = df.merge(ms, on='movie_id', how='left')
    df['movie_rating_std'] = df['movie_rating_std'].fillna(0)

    # ---- 2.6 交互/衍生特征 ----
    # 用户-电影评分交互：两个高均值相乘 → 高值，两个低均值相乘 → 低值
    df['user_movie_avg_product'] = df['user_avg_rating'] * df['movie_avg_rating']
    # 电影被评价时的"年龄"：越老的电影，用户口味可能越不同
    df['movie_age_at_rating'] = df['rating_datetime'].dt.year - df['movie_year']

    print(f"  构造后特征数: {df.shape[1]}")
    return df


def _season(date_str):
    """
    从 "01-Jan-1995" 格式的日期字符串提取季节
    1=春季(3-5月) / 2=夏季(6-8月) / 3=秋季(9-11月) / 4=冬季(12-2月)
    """
    try:
        m = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,  'May': 5,  'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
             }[date_str.split('-')[1]]
        return 1 if m in [3, 4, 5] else (2 if m in [6, 7, 8]
                                          else (3 if m in [9, 10, 11] else 4))
    except Exception:
        return 0  # 解析失败默认未知


def preprocess_for_modeling(df, genre_names):
    """
    编码与标准化 —— 将 DataFrame 转为模型可用的数值矩阵

    处理步骤:
      1. 分离类别特征和数值特征
      2. 类别特征编码: gender(M/F→0/1), occupation(LabelEncoding)
      3. 数值特征标准化: Z-Score = (x - mean) / std（StandardScaler）
      4. 电影类型特征: 已是二值(0/1)，无需处理
      5. 标签 y = rating 列（1-5分）

    Returns:
        X             : np.ndarray [n_samples, 35]  特征矩阵
        y             : np.ndarray [n_samples, ]     评分标签
        feature_names : list, 35个特征名称
        scaler        : StandardScaler 对象（其他模块可用）
    """
    print_subsection("编码与标准化")

    # 类别特征（需要编码）
    categorical_cols = ['gender', 'occupation', 'movie_season']
    # 数值特征（需要标准化）
    numeric_cols = ['age', 'movie_year', 'rating_dayofweek', 'rating_is_weekend',
                    'rating_month', 'user_avg_rating', 'movie_avg_rating',
                    'user_rating_count', 'movie_rating_count', 'user_rating_std',
                    'movie_rating_std', 'movie_age_at_rating', 'user_movie_avg_product']

    X_df = df[numeric_cols + categorical_cols].copy()

    # 缺失值：用中位数填充（比均值更抗异常值）
    for col in numeric_cols:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    # 类别编码
    X_df['gender'] = X_df['gender'].map({'M': 0, 'F': 1}).fillna(0).astype(int)
    X_df['occupation'] = LabelEncoder().fit_transform(X_df['occupation'].fillna('other'))
    X_df['movie_season'] = X_df['movie_season'].fillna(0).astype(int)

    # Z-Score 标准化：使不同量纲特征在相同尺度上，利于梯度下降和正则化
    scaler = StandardScaler()
    X_df[numeric_cols] = scaler.fit_transform(X_df[numeric_cols])

    # 电影类型特征（One-Hot 二值，不需要标准化）
    for g in genre_names:
        if g in df.columns:
            X_df[f'genre_{g}'] = df[g].fillna(0).astype(int)

    y = df['rating'].values  # 标签：评分 1-5
    X = X_df.values.astype(np.float32)
    print(f"  特征矩阵: {X.shape}, 标签: {y.shape}")
    return X, y, list(X_df.columns), scaler


def split_data(X, y):
    """
    数据集划分 —— 训练集 70% / 验证集 15% / 测试集 15%

    划分策略:
      - 训练集: 用于模型参数学习（梯度下降）
      - 验证集: 用于超参数调优和早停判断
      - 测试集: 用于最终评估，全程不参与训练
    """
    print_subsection("数据集划分")

    # 第一次划分：分出 70% 训练 + 30% (验证+测试)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=RANDOM_SEED)
    # 第二次划分：30% 中再按 15:15 分为验证和测试
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),  # 15/(15+15) = 0.5
        random_state=RANDOM_SEED)

    print(f"  训练:{X_train.shape[0]}  验证:{X_val.shape[0]}  测试:{X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# 主入口 —— 直接运行本文件时执行
# ============================================================
if __name__ == "__main__":
    print_section("模块一：数据预处理与特征工程")

    # Step 1-2: 加载与探查
    ratings_df, movies_df, users_df, genre_names = load_movielens_data()
    explore_data(ratings_df, movies_df, users_df)

    # Step 3: 构造特征（核心步骤）
    df = build_features(ratings_df, movies_df, users_df, genre_names)

    # Step 4: 编码与标准化
    X, y, feature_names, scaler = preprocess_for_modeling(df, genre_names)

    # Step 5: 数据集划分
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Step 6: 保存到 data/processed/（供模块二/三/四加载）
    print_subsection("保存到 data/processed/")
    movies_df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "movies_df.pkl"))
    users_df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "users_df.pkl"))
    df.to_pickle(os.path.join(PROCESSED_DATA_DIR, "full_df.pkl"))
    np.savez(os.path.join(PROCESSED_DATA_DIR, "split_data.npz"),
             X_train=X_train, X_val=X_val, X_test=X_test,
             y_train=y_train, y_val=y_val, y_test=y_test)
    with open(os.path.join(PROCESSED_DATA_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump({"feature_names": feature_names, "genre_names": genre_names,
                     "n_train": X_train.shape[0], "n_val": X_val.shape[0],
                     "n_test": X_test.shape[0]}, f)
    import joblib
    joblib.dump(scaler, os.path.join(PROCESSED_DATA_DIR, "scaler.pkl"))

    print_section("模块一完成！接下来运行:")
    print("  python src/module2_regression.py     # 回归：Ridge vs MLP")
    print("  python src/module3_classification.py  # 分类：决策树 vs 随机森林")
