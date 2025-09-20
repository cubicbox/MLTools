"""
カリフォルニア住宅価格データセットでのダブルクロスバリデーション回帰モデル選択テスト

実際のデータセットで10種類の回帰モデルを比較し、
最適なモデルとハイパーパラメータを選択します。
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from double_cv_regression import DoubleCV_RegressionSelector
import time

def california_housing_analysis():
    """カリフォルニア住宅価格データセットでの包括的分析"""

    print("=" * 70)
    print("カリフォルニア住宅価格データセット - ダブルクロスバリデーション分析")
    print("=" * 70)

    # データセット読み込み
    print("データセットを読み込み中...")
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names

    print(f"データ形状: X={X.shape}, y={y.shape}")
    print(f"特徴量: {list(feature_names)}")
    print(f"目標変数 (住宅価格):")
    print(f"  平均: ${y.mean():.2f}万")
    print(f"  範囲: ${y.min():.2f}万 - ${y.max():.2f}万")
    print(f"  標準偏差: ${y.std():.2f}万")

    # データの基本統計
    print("\n特徴量の基本統計:")
    feature_stats = pd.DataFrame(X, columns=feature_names).describe()
    print(feature_stats.round(2))

    print("\n" + "=" * 70)
    print("ダブルクロスバリデーション開始")
    print("=" * 70)

    # モデル選択器の設定
    # カリフォルニア住宅データは20,640サンプルと大規模なため、
    # 計算時間を考慮して設定を調整
    selector = DoubleCV_RegressionSelector(
        outer_cv=5,           # 外側CV: 信頼性のため5フォールド
        inner_cv=3,           # 内側CV: 効率性のため3フォールド
        n_trials=50,          # Optuna試行回数: 大規模データのため削減
        random_state=42,
        n_jobs=-1,           # 全CPUコア使用
        verbose=True
    )

    # 計算時間を考慮してモデルを段階的にテスト
    print("Phase 1: 高速モデル群でのテスト")
    print("-" * 50)

    start_time = time.time()

    # Phase 1: 高速モデル群
    fast_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN']
    selector.fit(X, y, model_subset=fast_models)

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 完了時間: {phase1_time:.1f}秒")

    # Phase 1の結果表示
    print("\nPhase 1 結果:")
    results_df = selector.get_results_summary()
    print(results_df.round(4))

    print("\n" + "=" * 50)
    print("Phase 2: 全モデル群での詳細分析")
    print("-" * 50)

    # Phase 2: 全モデル（大規模データでは時間がかかるため注意喚起）
    print("注意: 全モデルでの分析には時間がかかります...")

    # 全モデルのリスト
    all_models = [
        'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
        'RandomForest', 'GradientBoosting', 'SVR', 'KNN',
        'DecisionTree'  # GaussianProcessは大規模データでは除外
    ]

    start_time = time.time()
    selector_full = DoubleCV_RegressionSelector(
        outer_cv=3,           # 時間短縮のため3フォールド
        inner_cv=2,           # 時間短縮のため2フォールド
        n_trials=30,          # 試行回数をさらに削減
        random_state=42,
        n_jobs=-1,
        verbose=True
    )

    selector_full.fit(X, y, model_subset=all_models)

    phase2_time = time.time() - start_time
    print(f"\nPhase 2 完了時間: {phase2_time:.1f}秒")
    print(f"総実行時間: {(phase1_time + phase2_time):.1f}秒")

    # 最終結果の詳細分析
    print("\n" + "=" * 70)
    print("最終結果 - 詳細分析")
    print("=" * 70)

    selector_full.print_detailed_results()

    # 特徴量重要度分析（利用可能な場合）
    if selector_full.feature_importance_ is not None:
        print("\n特徴量重要度分析:")
        print("-" * 30)
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': selector_full.feature_importance_
        }).sort_values('Importance', ascending=False)

        print(importance_df.round(4))

        print(f"\n最も重要な特徴量: {importance_df.iloc[0]['Feature']}")
        print(f"最も重要でない特徴量: {importance_df.iloc[-1]['Feature']}")

    # 予測例
    print("\n予測例 (最初の10サンプル):")
    print("-" * 40)
    sample_indices = np.arange(10)
    y_pred_sample = selector_full.predict(X[sample_indices])

    comparison_df = pd.DataFrame({
        'Real Price': y[sample_indices],
        'Predicted Price': y_pred_sample,
        'Absolute Error': np.abs(y[sample_indices] - y_pred_sample),
        'Relative Error (%)': np.abs(y[sample_indices] - y_pred_sample) / y[sample_indices] * 100
    })

    print(comparison_df.round(4))

    # 性能サマリー
    print(f"\n" + "=" * 70)
    print("性能サマリー")
    print("=" * 70)

    best_model = selector_full.best_model_name_
    best_rmse = selector_full.results_[best_model]['mean_rmse']
    best_r2 = selector_full.results_[best_model]['mean_r2']

    print(f"最適モデル: {best_model}")
    print(f"期待RMSE: {best_rmse:.4f} (万ドル)")
    print(f"期待R²: {best_r2:.4f}")
    print(f"平均絶対誤差割合: {(best_rmse/y.mean()*100):.2f}%")

    # 実用的な解釈
    print(f"\n実用的解釈:")
    print(f"- 平均予測誤差: 約${best_rmse:.2f}万 (約{best_rmse*10:.0f}万円)")
    print(f"- 説明可能分散: {best_r2*100:.1f}%")
    print(f"- 住宅価格の{(best_rmse/y.mean()*100):.1f}%程度の誤差で予測可能")

    return selector_full

def model_comparison_analysis():
    """より詳細なモデル比較分析"""

    print("\n" + "=" * 70)
    print("詳細モデル比較分析")
    print("=" * 70)

    # データ読み込み
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # 各モデルタイプごとの特徴を分析
    model_categories = {
        '線形モデル': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
        'アンサンブル': ['RandomForest', 'GradientBoosting'],
        'インスタンスベース': ['KNN'],
        'カーネル': ['SVR'],
        '決定木': ['DecisionTree']
    }

    results = {}

    for category, models in model_categories.items():
        print(f"\n{category}モデル群の分析...")

        selector = DoubleCV_RegressionSelector(
            outer_cv=3,
            inner_cv=2,
            n_trials=20,
            random_state=42,
            verbose=False  # 詳細出力を抑制
        )

        start_time = time.time()
        selector.fit(X, y, model_subset=models)
        elapsed_time = time.time() - start_time

        # 最良モデルの結果を保存
        best_model = selector.best_model_name_
        best_result = selector.results_[best_model]

        results[category] = {
            'best_model': best_model,
            'rmse': best_result['mean_rmse'],
            'r2': best_result['mean_r2'],
            'time': elapsed_time
        }

        print(f"  最良: {best_model} (RMSE: {best_result['mean_rmse']:.4f}, R²: {best_result['mean_r2']:.4f})")
        print(f"  実行時間: {elapsed_time:.1f}秒")

    # カテゴリ別比較結果
    print(f"\n" + "=" * 70)
    print("カテゴリ別最良モデル比較")
    print("=" * 70)

    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values('rmse')

    print(comparison_df.round(4))

    print(f"\n総合最優秀カテゴリ: {comparison_df.index[0]}")
    print(f"最速カテゴリ: {comparison_df.sort_values('time').index[0]}")
    print(f"最高精度カテゴリ: {comparison_df.sort_values('rmse').index[0]}")

if __name__ == "__main__":
    try:
        # メイン分析の実行
        final_selector = california_housing_analysis()

        # 詳細比較分析の実行
        model_comparison_analysis()

        print(f"\n" + "=" * 70)
        print("分析完了!")
        print("=" * 70)

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        print("必要な依存関係がインストールされているか確認してください:")
        print("uv pip install numpy pandas scikit-learn optuna")