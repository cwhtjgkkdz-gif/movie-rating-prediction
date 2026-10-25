"""
答辩素材生成脚本
=================
运行: python build_presentation.py
输出: presentation/ 目录下所有答辩素材
  - figures/  → 6 张图表
  - regression_output.txt  → 回归模块控制台输出
  - classification_output.txt → 分类模块控制台输出
  - index.html → 前端展示页面
"""
import os, sys, subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

BASE = os.path.dirname(os.path.abspath(__file__))
PRES = os.path.join(BASE, 'presentation')
FIGS = os.path.join(PRES, 'figures')
os.makedirs(FIGS, exist_ok=True)

# ============================================================
# 运行预处理（如果还没运行）
# ============================================================
print("=" * 60)
print("  Step 0: 数据预处理")
print("=" * 60)
if not os.path.exists(os.path.join(BASE, 'data', 'processed', 'split_data.npz')):
    print("  预处理数据不存在，正在运行模块一...")
    subprocess.run([sys.executable, os.path.join(BASE, 'src', 'module1_preprocessing.py')],
                   cwd=BASE)
else:
    print("  预处理数据已存在，跳过。")

# ============================================================
# 运行回归模块，捕获输出
# ============================================================
print("\n" + "=" * 60)
print("  Step 1: 运行回归模块")
print("=" * 60)

reg_result = subprocess.run(
    [sys.executable, os.path.join(BASE, 'src', 'module2_regression.py')],
    capture_output=True, text=True, cwd=BASE
)
reg_output = reg_result.stdout + reg_result.stderr
with open(os.path.join(PRES, 'regression_output.txt'), 'w', encoding='utf-8') as f:
    f.write(reg_output)
print("  控制台输出已保存到: presentation/regression_output.txt")

# ============================================================
# 运行分类模块，捕获输出
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: 运行分类模块")
print("=" * 60)

cls_result = subprocess.run(
    [sys.executable, os.path.join(BASE, 'src', 'module3_classification.py')],
    capture_output=True, text=True, cwd=BASE
)
cls_output = cls_result.stdout + cls_result.stderr
with open(os.path.join(PRES, 'classification_output.txt'), 'w', encoding='utf-8') as f:
    f.write(cls_output)
print("  控制台输出已保存到: presentation/classification_output.txt")

# ============================================================
# 检查图表
# ============================================================
print("\n" + "=" * 60)
print("  生成完毕！")
print("=" * 60)
print(f"""
  presentation/
  ├── index.html                    ← 浏览器打开这个
  ├── regression_output.txt         ← 回归控制台输出
  ├── classification_output.txt     ← 分类控制台输出
  └── figures/
      ├── p_regression_comparison.png   ← 回归指标对比
      ├── p_mlp_loss.png                ← MLP 训练曲线
      ├── p_prediction_scatter.png      ← 预测 vs 真实
      ├── p_n_estimators.png            ← n_estimators 曲线
      ├── p_feature_importance.png      ← 特征重要性
      └── p_confusion_matrix.png        ← 混淆矩阵

  下一步: 浏览器打开 presentation/index.html
""")
