"""
模块二：评分预测 — 回归任务
==========================================
运行: python src/module2_regression.py
前提: 先运行 module1_preprocessing.py
对比: Ridge 岭回归 (线性+L2正则) vs MLP 神经网络 (非线性)
评估: MAE / RMSE / R²
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import (RIDGE_ALPHAS, MLP_HIDDEN_LAYERS, MLP_EPOCHS,
                    MLP_BATCH_SIZE, RANDOM_SEED)
from utils import (print_section, print_subsection, show_plot, load_processed_data)
# 注意: 中文字体配置已在 utils.py 全局设置，无需重复


def run():
    print_section("模块二：评分预测 — Ridge vs MLP 回归对比")

    # ---- 加载模块一划分好的训练/验证/测试集 ----
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data("split")
    print(f"  训练集: {len(X_train):,}  验证集: {len(X_val):,}  测试集: {len(X_test):,}")

    # ================================================================
    # 模型一：Ridge 岭回归
    # Loss = MSE + α·Σw²，α 通过 5 折交叉验证网格搜索选出
    # ================================================================
    print_subsection("模型一：Ridge 岭回归")
    print("  原理: 线性回归 + L2 正则化 λ·||w||²，惩罚大权重防止过拟合")

    grid = GridSearchCV(Ridge(random_state=RANDOM_SEED), {'alpha': RIDGE_ALPHAS},
                        cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    best_a = grid.best_params_['alpha']
    print(f"  最优 α = {best_a}")
    for a, s in zip(RIDGE_ALPHAS, -grid.cv_results_['mean_test_score']):
        print(f"    α={a:6.2f}  →  MSE={s:.4f}{'  ★ 最优' if a == best_a else ''}")

    ridge = grid.best_estimator_
    yp_ridge = ridge.predict(X_test)
    r_mae = mean_absolute_error(y_test, yp_ridge)
    r_rmse = np.sqrt(mean_squared_error(y_test, yp_ridge))
    r_r2 = r2_score(y_test, yp_ridge)

    # ================================================================
    # 模型二：MLP 神经网络（重点模型）
    # 结构 64→32→16→1, ReLU+Adam+EarlyStopping+adaptive LR
    # ================================================================
    print_subsection("模型二：MLP 神经网络")
    print(f"  结构: 输入(35) → 隐藏层{MLP_HIDDEN_LAYERS} → 输出(1)")
    print("  激活: ReLU  |  优化器: Adam  |  早停: EarlyStopping(耐心=10)")
    print("  正则: L2(α=0.0001)  |  学习率: adaptive 自适应衰减")

    # 合并训练+验证集 → MLPRegressor 内部再分 15% 做验证
    Xf = np.vstack([X_train, X_val])   # 纵向堆叠 [n_train+n_val, 35]
    yf = np.hstack([y_train, y_val])
    mlp = MLPRegressor(
        hidden_layer_sizes=tuple(MLP_HIDDEN_LAYERS),
        activation='relu',           # f(x)=max(0,x)，缓解梯度消失
        solver='adam',               # 自适应学习率优化器
        batch_size=MLP_BATCH_SIZE,   # 每批64样本更新权重
        max_iter=MLP_EPOCHS,         # 最多100轮
        learning_rate='adaptive',    # loss不降时学习率自动减半
        early_stopping=True,         # 验证loss连续10轮不降则停止
        validation_fraction=0.15,    # 内部15%做验证集
        n_iter_no_change=10,         # 早停耐心值
        random_state=RANDOM_SEED,
        verbose=True                 # 打印每轮训练详情
    )
    mlp.fit(Xf, yf)

    yp_mlp = mlp.predict(X_test)
    m_mae = mean_absolute_error(y_test, yp_mlp)
    m_rmse = np.sqrt(mean_squared_error(y_test, yp_mlp))
    m_r2 = r2_score(y_test, yp_mlp)
    print(f"  实际迭代: {mlp.n_iter_} 轮  |  最终 Loss: {mlp.loss_:.4f}")

    # ================================================================
    # 模型对比表
    # ================================================================
    print("\n  ╔══════════════════════════════════════════════════════════╗")
    print("  ║              回归模型对比 (测试集)                         ║")
    print("  ╠══════════════╦══════════╦═══════════╦════════════════════╣")
    print("  ║ 模型          ║   MAE ↓  ║  RMSE ↓   ║   R² ↑             ║")
    print("  ╠══════════════╬══════════╬═══════════╬════════════════════╣")
    print(f"  ║ Ridge (岭回归) ║ {r_mae:.4f}   ║ {r_rmse:.4f}    ║ {r_r2:.4f}            ║")
    print(f"  ║ MLP (神经网络) ║ {m_mae:.4f}   ║ {m_rmse:.4f}    ║ {m_r2:.4f}            ║")
    print("  ╚══════════════╩══════════╩═══════════╩════════════════════╝")

    # 判定最优模型
    if m_r2 > r_r2:
        print(f"\n  ★ 最优模型: MLP 神经网络 (R²={m_r2:.4f}, MAE={m_mae:.4f})")
    else:
        print(f"\n  ★ 最优模型: Ridge 岭回归 (R²={r_r2:.4f}, MAE={r_mae:.4f})")

    print(f"""
  【MLP 最优参数】
  ├─ hidden_layer_sizes : {MLP_HIDDEN_LAYERS}
  ├─ activation         : ReLU
  ├─ solver             : Adam
  ├─ alpha (L2正则)     : 0.0001
  ├─ batch_size         : {MLP_BATCH_SIZE}
  ├─ learning_rate      : adaptive (自适应衰减)
  ├─ early_stopping     : True (耐心=10)
  ├─ 实际迭代次数        : {mlp.n_iter_}
  └─ 最终 Loss          : {mlp.loss_:.4f}
  """)

    # ================================================================
    # 图1：三指标柱状图（MAE/RMSE/R²）
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names = ['Ridge\n(岭回归)', 'MLP\n(神经网络)']
    colors = ['#4472C4', '#ED7D31']
    for ax, label, vals in zip(axes,
                                ['MAE (越小越好)', 'RMSE (越小越好)', 'R² (越大越好)'],
                                [[r_mae, m_mae], [r_rmse, m_rmse], [r_r2, m_r2]]):
        bars = ax.bar(names, vals, color=colors, edgecolor='white', width=0.45)
        ax.set_title(label, fontsize=13, fontweight='bold')
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 0.5,
                    f'{v:.4f}', ha='center', fontsize=13, fontweight='bold', color='white')
    fig.suptitle('回归模型性能对比', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('presentation/figures/p_regression_comparison.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_regression_comparison.png")
    show_plot(fig)

    # ================================================================
    # 图2：MLP 训练 Loss 曲线（红色虚线标注早停点）
    # ================================================================
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mlp.loss_curve_, color='#4472C4', linewidth=2, label='训练集 Loss')
    ax.axvline(x=mlp.n_iter_ - 1, color='red', linestyle='--', linewidth=1.5,
               label=f'早停触发点（第 {mlp.n_iter_} 轮）')
    ax.set_xlabel('迭代轮数', fontsize=12)
    ax.set_ylabel('损失值（MSE）', fontsize=12)
    ax.set_title('MLP 神经网络训练过程', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11), ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('presentation/figures/p_mlp_loss.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_mlp_loss.png")
    show_plot(fig)

    # ================================================================
    # 图3：预测值 vs 真实值散点图
    # 左：全部测试样本 | 右：10个展示样本高亮
    # ================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：全部样本（灰色半透明，展示整体分布）
    axes[0].scatter(y_test, yp_mlp, alpha=0.2, s=8, color='#4472C4')
    axes[0].plot([1, 5], [1, 5], 'r--', linewidth=1.5, label='完美预测线')
    axes[0].set_xlabel('真实评分', fontsize=12), axes[0].set_ylabel('预测评分', fontsize=12)
    axes[0].set_title(f'全部 {len(y_test):,} 条测试样本', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10), axes[0].set_xlim(0.5, 5.5), axes[0].set_ylim(0.5, 5.5)
    axes[0].grid(True, alpha=0.3)

    # 右图：10个展示样本以橙色大圆点高亮
    np.random.seed(42)
    demo_idx = np.random.choice(len(y_test), size=10, replace=False)
    axes[1].scatter(y_test, yp_mlp, alpha=0.08, s=5, color='gray')
    axes[1].scatter(y_test[demo_idx], yp_mlp[demo_idx], alpha=1, s=100,
                    color='#ED7D31', edgecolors='black', linewidths=1.5,
                    label=f'抽选的 {len(demo_idx)} 个展示样本', zorder=5)
    axes[1].plot([1, 5], [1, 5], 'r--', linewidth=1.5)
    axes[1].set_xlabel('真实评分', fontsize=12), axes[1].set_ylabel('预测评分', fontsize=12)
    axes[1].set_title('展示样本 vs 全部样本', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10), axes[1].set_xlim(0.5, 5.5), axes[1].set_ylim(0.5, 5.5)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle('MLP 预测效果：预测评分 vs 真实评分', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('presentation/figures/p_prediction_scatter.png', dpi=150, bbox_inches='tight')
    print("  [图] 已保存: p_prediction_scatter.png")
    show_plot(fig)

    print_section("模块二完成！")
    return mlp


if __name__ == "__main__":
    os.makedirs('presentation/figures', exist_ok=True)
    run()
