"""
カリフォルニア住宅価格データセット - 高速テスト版
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from double_cv_regression import DoubleCV_RegressionSelector
import time

def quick_california_housing_test():
    """高速なカリフォルニア住宅価格分析"""

    print("=" * 60)
    print("カリフォルニア住宅価格データセット - 高速分析")
    print("=" * 60)

    # データセット読み込み
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    print(f"データ形状: X={X.shape}, y={y.shape}")
    print(f"特徴量: {list(feature_names)}")
    print(f"住宅価格統計:")
    print(f"  平均: ${y.mean():.2f}万")
    print(f"  範囲: ${y.min():.2f}万 - ${y.max():.2f}万")

    # 高速設定でのモデル選択
    print(f"\n高速モデル比較を開始...")

    selector = DoubleCV_RegressionSelector(
        outer_cv=3,          # 3フォールドで高速化
        inner_cv=2,          # 2フォールドで高速化
        n_trials=20,         # 試行回数を削減
        random_state=42,
        verbose=True
    )

    # 高速モデルのみでテスト
    fast_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN']

    start_time = time.time()
    selector.fit(X, y, model_subset=fast_models)
    elapsed_time = time.time() - start_time

    print(f"\n実行時間: {elapsed_time:.1f}秒")

    # 結果表示
    selector.print_detailed_results()

    # 予測例
    print(f"\n予測例 (最初の5サンプル):")
    sample_X = X[:5]
    sample_y = y[:5]
    pred_y = selector.predict(sample_X)

    for i in range(5):
        real_price = sample_y[i]
        pred_price = pred_y[i]
        error = abs(real_price - pred_price)
        error_pct = (error / real_price) * 100

        print(f"  {i+1}. 実際: ${real_price:.2f}万, 予測: ${pred_price:.2f}万, "
              f"誤差: ${error:.2f}万 ({error_pct:.1f}%)")

    # 実用的解釈
    best_model = selector.best_model_name_
    best_rmse = selector.results_[best_model]['mean_rmse']
    best_r2 = selector.results_[best_model]['mean_r2']

    print(f"\n実用的解釈:")
    print(f"最適モデル: {best_model}")
    print(f"予測精度: R² = {best_r2:.3f} ({best_r2*100:.1f}%の分散を説明)")
    print(f"予測誤差: RMSE = {best_rmse:.3f}万ドル (平均価格の{(best_rmse/y.mean()*100):.1f}%)")

    return selector

if __name__ == "__main__":
    quick_california_housing_test()