#!/usr/bin/env python3
"""
California Housing DatasetでDCV_reg_selectorのテスト
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from DCV_reg_selector import DCV_reg_selector

def test_housing_dataset():
    """California Housing Datasetでテスト実行"""

    print("=" * 60)
    print("California Housing Dataset テスト")
    print("=" * 60)

    # California Housing Datasetの読み込み
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    print(f"データ形状: X={X.shape}, y={y.shape}")
    print(f"特徴量: {list(feature_names)}")
    print(f"目標変数（住宅価格）の範囲: {y.min():.1f} - {y.max():.1f}")

    # データサイズが大きいので一部を使用
    n_samples = 1000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    print(f"サンプリング後のデータ形状: X={X.shape}, y={y.shape}")

    print("-" * 60)

    # DCV_reg_selectorでモデル選択
    selector = DCV_reg_selector(
        outer_cv=3,      # 高速化のため3-fold
        inner_cv=3,      # 内側も3-fold
        n_trials=30,     # Optuna試行回数を30回に
        verbose=True,
        save_models=True,
        save_dir="./housing_models"
    )

    # 一部のモデルで実行（高速化のため）
    model_subset = ['LinearRegression', 'Ridge', 'Lasso', 'RandomForest', 'SVR']

    print(f"使用モデル: {model_subset}")
    print("-" * 60)

    # フィット実行
    selector.fit(X, y, model_subset=model_subset)

    # 結果表示
    print("\n" + "=" * 60)
    print("詳細結果")
    print("=" * 60)
    selector.print_detailed_results()

    # サマリー表示
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    summary_df = selector.get_results_summary()
    print(summary_df.to_string(index=False))

    # 予測例
    print(f"\n予測例（最初の5サンプル）:")
    y_pred = selector.predict(X[:5])
    print(f"真値: {y[:5]}")
    print(f"予測: {y_pred}")
    print(f"誤差: {np.abs(y[:5] - y_pred)}")

    # 保存されたモデルでの予測テスト
    print(f"\n保存されたベストモデルでの予測テスト:")
    y_pred_saved = DCV_reg_selector.predict_with_saved_model(
        "./housing_models/best_model.pkl", X[:3]
    )
    print(f"保存モデル予測: {y_pred_saved}")
    print(f"直接予測:     {selector.predict(X[:3])}")
    print(f"予測一致: {np.allclose(y_pred_saved, selector.predict(X[:3]))}")

    print("\n" + "=" * 60)
    print("テスト完了!")
    print("=" * 60)

if __name__ == "__main__":
    test_housing_dataset()