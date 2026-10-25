"""
电影评分预测系统 — Web 服务
=============================
纯 Python 内置库实现，零额外依赖。
启动: python app.py  →  浏览器 http://localhost:5000

页面:
  /                首页（数据规模 + 三个功能入口）
  /results         训练结果（8张图表 + 指标 + 结论）
  /regression      评分预测（Ridge + MLP 实时预测）
  /classification  类型分类（决策树 + 随机森林 + Top3 置信度）

API:
  POST /api/predict/regression     回归预测
  POST /api/predict/classification 分类预测
"""
import os, json, joblib, numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ============================================================
# 加载训练好的模型和元数据（由 train_models.py 生成）
# ============================================================
ridge = joblib.load('models/ridge_model.pkl')       # Ridge 岭回归
mlp   = joblib.load('models/mlp_model.pkl')         # MLP 神经网络
dt    = joblib.load('models/dt_model.pkl')          # CART 决策树
rf    = joblib.load('models/rf_model.pkl')          # 随机森林
reg_meta = joblib.load('models/reg_meta.pkl')       # 回归元数据（特征名、类型名）
cls_meta = joblib.load('models/cls_meta.pkl')       # 分类元数据
refs = joblib.load('models/refs.pkl')               # 参考值（默认值、职业列表等）
scaler = joblib.load(os.path.join('data', 'processed', 'scaler.pkl'))  # Z-Score 标准化器

FEATURES_REG = reg_meta['feature_names']   # 35 个回归特征名（保证顺序与训练一致）
GENRE_NAMES  = reg_meta['genre_names']      # 19 个电影类型名
FEATURES_CLS = cls_meta['feature_names']    # 22 个分类特征名
CLASS_NAMES  = cls_meta['class_names']       # 17 个类别名
DISPLAY_GENRES = [g for g in GENRE_NAMES if g != 'unknown']  # 排除无意义的 unknown

# 电影类型中英文映射（页面显示中文，API 传英文 key）
GENRE_CN = {
    'Action': '动作', 'Adventure': '冒险', 'Animation': '动画',
    "Children's": '儿童', 'Comedy': '喜剧', 'Crime': '犯罪',
    'Documentary': '纪录', 'Drama': '剧情', 'Fantasy': '奇幻',
    'Film-Noir': '黑色', 'Horror': '恐怖', 'Musical': '音乐',
    'Mystery': '悬疑', 'Romance': '爱情', 'Sci-Fi': '科幻',
    'Thriller': '惊悚', 'War': '战争', 'Western': '西部',
}

# scaler 在训练时只对前 13 个数值特征做标准化
# 预测时必须用同样的 scaler 对同样的列做 transform
NUMERIC_COLS = ['age', 'movie_year', 'rating_dayofweek', 'rating_is_weekend',
                'rating_month', 'user_avg_rating', 'movie_avg_rating',
                'user_rating_count', 'movie_rating_count', 'user_rating_std',
                'movie_rating_std', 'movie_age_at_rating', 'user_movie_avg_product']


# ============================================================
# 特征构造 — 将前端表单数据转为模型可用的特征向量
# ============================================================
def build_reg_features(data):
    """
    构造 35 维回归特征向量，严格按 FEATURES_REG 顺序。
    数值特征用 scaler 做 Z-Score 标准化（与训练时一致），
    类别/类型特征保持原值（LabelEncoder 编码或 0/1 二值）。
    """
    vec = np.zeros(len(FEATURES_REG))
    for i, name in enumerate(FEATURES_REG):
        # ── 数值特征 ──
        if name == 'age':              vec[i] = float(data.get('age', 30))
        elif name == 'movie_year':     vec[i] = float(data.get('movie_year', 1990))
        elif name == 'rating_dayofweek': vec[i] = float(data.get('dayofweek', 3))
        elif name == 'rating_is_weekend': vec[i] = 1.0 if data.get('weekend') == '1' else 0.0
        elif name == 'rating_month':   vec[i] = float(data.get('month', 6))
        elif name == 'user_avg_rating': vec[i] = float(data.get('user_avg_rating', 3.53))
        elif name == 'movie_avg_rating': vec[i] = float(data.get('movie_avg_rating', 3.53))
        elif name == 'user_rating_count': vec[i] = float(data.get('user_rating_count', 100))
        elif name == 'movie_rating_count': vec[i] = float(data.get('movie_rating_count', 100))
        elif name == 'user_rating_std': vec[i] = float(data.get('user_rating_std', 1.0))
        elif name == 'movie_rating_std': vec[i] = float(data.get('movie_rating_std', 1.0))
        elif name == 'movie_age_at_rating': vec[i] = float(data.get('movie_age', 5))
        elif name == 'user_movie_avg_product':
            # 交互特征 = 用户评分均值 × 电影评分均值
            u_avg = float(data.get('user_avg_rating', 3.53))
            m_avg = float(data.get('movie_avg_rating', 3.53))
            vec[i] = u_avg * m_avg
        # ── 类别特征 ──
        elif name == 'gender':
            vec[i] = 0.0 if data.get('gender', 'M') == 'M' else 1.0  # M→0, F→1
        elif name == 'occupation':
            # 按 occupations 列表中的位置作为编码（与训练时 LabelEncoder 一致）
            occ = data.get('occupation', 'student')
            vec[i] = float(refs['occupations'].index(occ)) if occ in refs['occupations'] else 0.0
        elif name == 'movie_season':
            vec[i] = float(data.get('season', 2))  # 1=春 2=夏 3=秋 4=冬
        # ── 电影类型标志（0/1 二值） ──
        elif name.startswith('genre_'):
            g = name.replace('genre_', '')
            if g != 'unknown':
                vec[i] = 1.0 if data.get(f'genre_{g}') == '1' else 0.0

    # 只对 13 个数值特征做标准化（与训练时完全一致）
    num_idx = [i for i, name in enumerate(FEATURES_REG) if name in NUMERIC_COLS]
    raw = vec.reshape(1, -1)
    raw[0, num_idx] = scaler.transform(raw[0, num_idx].reshape(1, -1))
    return raw


def build_cls_features(data):
    """
    构造分类特征向量（22 维）。
    注意：分类特征中的类型名不带 'genre_' 前缀（如 'Action'），
    与回归特征（带 'genre_' 前缀，如 'genre_Action'）不同。
    """
    vec = np.zeros(len(FEATURES_CLS))
    for i, name in enumerate(FEATURES_CLS):
        if name == 'movie_year':     vec[i] = float(data.get('movie_year', 1990))
        elif name == 'avg_rating':   vec[i] = float(data.get('avg_rating', 3.5))
        elif name == 'num_ratings':  vec[i] = float(data.get('num_ratings', 100))
        elif name == 'std_rating':   vec[i] = float(data.get('std_rating', 1.0))
        elif name in GENRE_NAMES and name != 'unknown':
            # 分类特征名是 'Action'（不带前缀），表单提交的是 'genre_Action'（带前缀）
            vec[i] = 1.0 if data.get(f'genre_{name}') == '1' or data.get(name) == '1' else 0.0
    return vec.reshape(1, -1)


# ============================================================
# 模板渲染 — 简单字符串替换，无需 Jinja2 等模板引擎
# ============================================================
def _genres_html():
    """生成电影类型复选框 HTML（显示中文，data-genre 存英文 key）"""
    return '\n'.join(
        f'<label><input type="checkbox" class="genre" data-genre="{g}">'
        f'<span>{GENRE_CN.get(g, g)}</span></label>'
        for g in DISPLAY_GENRES)


def _occ_html():
    """生成职业下拉选项 HTML"""
    return '\n'.join(
        f'<option value="{o}" {"selected" if o=="student" else ""}>{o}</option>'
        for o in refs['occupations'])


def render_page(name, **extra):
    """读取 HTML 模板文件，替换 __PLACEHOLDER__ 标记"""
    with open(os.path.join('templates', name), 'r', encoding='utf-8') as f:
        html = f.read()
    html = html.replace('__GENRES__', _genres_html())
    html = html.replace('__YEAR_MIN__', str(refs['movie_year_min']))
    html = html.replace('__YEAR_MAX__', str(refs['movie_year_max']))
    html = html.replace('__MOVIE_YEAR__', str(refs['movie_year_mean']))
    for k, v in extra.items():
        html = html.replace(k, str(v))
    return html


# ============================================================
# HTTP 请求处理
# ============================================================
class Handler(BaseHTTPRequestHandler):

    # ── GET：页面和静态文件 ──
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._serve_file('templates/index.html')
        elif path == '/results':
            self._serve_file('templates/results.html')
        elif path == '/regression':
            self._respond(200, render_page('regression.html',
                __AGE__=str(int(refs['age_mean'])),
                __AGE_MIN__=str(refs['age_min']),
                __AGE_MAX__=str(refs['age_max']),
                __OCCUPATIONS__=_occ_html()), 'text/html')
        elif path == '/classification':
            self._respond(200, render_page('classification.html'), 'text/html')
        elif path.startswith('/static/'):
            # 静态资源（PNG 图表等）
            filepath = path.lstrip('/')
            try:
                with open(filepath, 'rb') as f:
                    content = f.read()
                ct = 'image/png' if path.endswith('.png') else 'text/plain'
                self.send_response(200)
                self.send_header('Content-Type', ct)
                self.end_headers()
                self.wfile.write(content)
            except FileNotFoundError:
                self._respond(404, 'Not Found', 'text/plain')
        else:
            self._respond(404, 'Not Found', 'text/plain')

    # ── POST：预测 API ──
    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length).decode('utf-8') if length > 0 else '{}'
        try:
            data = json.loads(body)
        except Exception:
            data = {}

        if path == '/api/predict/regression':
            # Ridge + MLP 双模型预测评分
            x = build_reg_features(data)
            rp = float(ridge.predict(x)[0])
            mp = float(mlp.predict(x)[0])
            self._respond(200, json.dumps({
                'ridge': round(rp, 2),
                'mlp': round(mp, 2),
                'ensemble': round((rp + mp) / 2, 2)
            }, ensure_ascii=False), 'application/json')

        elif path == '/api/predict/classification':
            # 决策树 + 随机森林双模型分类 + Top3 置信度
            x = build_cls_features(data)
            dp = int(dt.predict(x)[0])
            rp = int(rf.predict(x)[0])
            self._respond(200, json.dumps({
                'dt_class': CLASS_NAMES[dp],
                'rf_class': CLASS_NAMES[rp],
                'dt_top3': _top3(dt.predict_proba(x)[0]),
                'rf_top3': _top3(rf.predict_proba(x)[0]),
            }, ensure_ascii=False), 'application/json')

        else:
            self._respond(404, 'Not Found', 'text/plain')

    # ── 工具方法 ──
    def _serve_file(self, filepath):
        """读取文件内容并以 text/html 返回"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self._respond(200, f.read(), 'text/html')
        except FileNotFoundError:
            self._respond(404, 'Not Found', 'text/plain')

    def _respond(self, code, body, content_type):
        """统一响应方法"""
        self.send_response(code)
        self.send_header('Content-Type', f'{content_type}; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body.encode('utf-8') if isinstance(body, str) else body)


def _top3(proba):
    """从概率数组中提取 Top 3 类别及置信度"""
    idx = np.argsort(proba)[-3:][::-1]
    return [{'class': CLASS_NAMES[i], 'prob': round(proba[i] * 100, 1)} for i in idx]


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print('\n  ╔════════════════════════════════════╗')
    print('  ║   🎬 电影评分预测系统已启动       ║')
    print('  ║   浏览器打开: http://localhost:5000  ║')
    print('  ║   按 Ctrl+C 停止                   ║')
    print('  ╚════════════════════════════════════╝\n')
    HTTPServer(('0.0.0.0', 5000), Handler).serve_forever()
