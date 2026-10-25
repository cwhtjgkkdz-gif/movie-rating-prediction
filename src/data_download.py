"""
MovieLens 100K 数据集下载与解压
下载地址: https://files.grouplens.org/datasets/movielens/ml-100k.zip
"""
import os
import zipfile
import urllib.request
import shutil
from config import RAW_DATA_DIR, DATASET_NAME


def download_movielens():
    """
    下载并解压 MovieLens 100K 数据集。
    如果已存在则跳过下载。
    """
    dataset_dir = os.path.join(RAW_DATA_DIR, DATASET_NAME)
    zip_path = os.path.join(RAW_DATA_DIR, "ml-100k.zip")

    # 检查是否已存在
    required_files = ["u.data", "u.item", "u.user"]
    if all(os.path.exists(os.path.join(dataset_dir, f)) for f in required_files):
        print(f"[数据集] MovieLens 100K 已存在: {dataset_dir}")
        return dataset_dir

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # 下载
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    print(f"[数据集] 正在下载 MovieLens 100K ...")
    print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)
        print(f"  下载完成: {zip_path}")
    except Exception as e:
        print(f"  [警告] 下载失败: {e}")
        print(f"  请手动下载并解压到: {dataset_dir}")
        print(f"  下载地址: https://grouplens.org/datasets/movielens/100k/")
        return None

    # 解压
    print(f"[数据集] 正在解压...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(RAW_DATA_DIR)
    print(f"  解压完成: {dataset_dir}")

    # 清理 zip
    os.remove(zip_path)

    return dataset_dir


def generate_sample_data(output_dir):
    """
    如果无法下载 MovieLens，生成模拟数据集用于测试和验证代码流程。
    生成的数据结构和 MovieLens 100K 保持一致。

    u.data:  user_id | movie_id | rating | timestamp
    u.item:  movie_id | title | release_date | video_release_date | IMDb_URL
             | genre_0 | ... | genre_18 (19个类型)
    u.user:  user_id | age | gender | occupation | zip_code

    模拟生成 943 用户 × 1682 电影 × ~100000 评分
    """
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    n_users = 943
    n_movies = 1682
    n_ratings = 100000

    # 电影类型名称
    genres = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    # --- u.user ---
    ages = np.random.randint(18, 65, size=n_users + 1)
    ages[0] = 0
    genders = np.random.choice(['M', 'F'], size=n_users + 1)
    genders[0] = 'M'
    occupations = np.random.choice([
        'technician', 'writer', 'executive', 'administrator', 'student',
        'lawyer', 'educator', 'scientist', 'entertainment', 'programmer',
        'librarian', 'homemaker', 'artist', 'doctor', 'retired',
        'marketing', 'salesman', 'engineer', 'none', 'other'
    ], size=n_users + 1)
    occupations[0] = 'other'
    zip_codes = [f"{np.random.randint(10000, 99999)}" for _ in range(n_users + 1)]
    zip_codes[0] = '00000'

    with open(os.path.join(output_dir, 'u.user'), 'w', encoding='utf-8') as f:
        for i in range(1, n_users + 1):
            f.write(f"{i}|{ages[i]}|{genders[i]}|{occupations[i]}|{zip_codes[i]}\n")

    # --- u.item ---
    movie_genres = np.random.randint(0, 2, size=(n_movies + 1, 19))
    movie_genres[0] = 0
    # 确保每部电影至少有一个类型
    for i in range(1, n_movies + 1):
        if np.sum(movie_genres[i]) == 0:
            genre_idx = np.random.randint(1, 19)  # 不包括 unknown
            movie_genres[i, genre_idx] = 1

    years = np.random.randint(1920, 2000, size=n_movies + 1)
    years[0] = 0

    with open(os.path.join(output_dir, 'u.item'), 'w', encoding='latin-1') as f:
        for i in range(1, n_movies + 1):
            title = f"Movie {i} ({years[i]})"
            release_date = f"{np.random.randint(1,29):02d}-{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][np.random.randint(0,12)]}-{years[i]}"
            genre_str = '|'.join(map(str, movie_genres[i]))
            f.write(f"{i}|{title}|{release_date}||http://example.com/imdb/title/{i}|{genre_str}\n")

    # --- u.data ---
    rating_pairs = set()
    while len(rating_pairs) < n_ratings:
        uid = np.random.randint(1, n_users + 1)
        mid = np.random.randint(1, n_movies + 1)
        if (uid, mid) not in rating_pairs:
            rating_pairs.add((uid, mid))

    # 评分分布偏正态，均值约3.5
    base_ratings = np.random.normal(3.5, 0.8, size=n_ratings)
    base_ratings = np.clip(np.round(base_ratings), 1, 5).astype(int)

    timestamps = np.random.randint(870000000, 900000000, size=n_ratings)

    with open(os.path.join(output_dir, 'u.data'), 'w', encoding='utf-8') as f:
        for (uid, mid), rating, ts in zip(sorted(rating_pairs), base_ratings, timestamps):
            f.write(f"{uid}\t{mid}\t{rating}\t{ts}\n")

    print(f"[模拟数据] 已生成 MovieLens 格式模拟数据: {output_dir}")
    print(f"  用户数: {n_users}, 电影数: {n_movies}, 评分数: {n_ratings}")
    return output_dir
