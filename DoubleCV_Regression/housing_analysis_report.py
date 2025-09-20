"""
カリフォルニア住宅価格データセット分析レポート
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from double_cv_regression import DoubleCV_RegressionSelector
import time

def generate_analysis_report():
    """包括的な分析レポートを生成"""

    print("=" * 80)
    print("カリフォルニア住宅価格データセット - 包括的分析レポート")
    print("=" * 80)

    # データ読み込み
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names

    print(f"\n📊 データセット概要")
    print(f"サンプル数: {X.shape[0]:,}")
    print(f"特徴量数: {X.shape[1]}")
    print(f"特徴量: {list(feature_names)}")

    # 特徴量の詳細説明
    feature_descriptions = {
        'MedInc': '世帯収入中央値 (万ドル)',
        'HouseAge': '住宅築年数 (年)',
        'AveRooms': '平均部屋数',
        'AveBedrms': '平均寝室数',
        'Population': '人口',
        'AveOccup': '平均世帯人数',
        'Latitude': '緯度',
        'Longitude': '経度'
    }

    print(f"\n📋 特徴量の詳細:")
    for i, (feature, desc) in enumerate(feature_descriptions.items()):
        mean_val = X[:, i].mean()
        std_val = X[:, i].std()
        print(f"  {feature}: {desc} (平均: {mean_val:.2f}, 標準偏差: {std_val:.2f})")

    print(f"\n🏠 住宅価格統計:")
    print(f"  平均価格: ${y.mean():.2f}万 (約{y.mean()*100:.0f}万円)")
    print(f"  価格範囲: ${y.min():.2f}万 - ${y.max():.2f}万")
    print(f"  標準偏差: ${y.std():.2f}万")

    # モデル比較分析
    print(f"\n🤖 モデル比較分析")
    print("=" * 50)

    # 高速モデル群
    print("\nPhase 1: 高速モデル群")
    fast_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet', 'KNN']

    selector1 = DoubleCV_RegressionSelector(
        outer_cv=3, inner_cv=2, n_trials=30, random_state=42, verbose=False
    )

    start_time = time.time()
    selector1.fit(X, y, model_subset=fast_models)
    phase1_time = time.time() - start_time

    print(f"実行時間: {phase1_time:.1f}秒")
    results1 = selector1.get_results_summary()
    print(results1.round(4))

    # アンサンブルモデル群
    print(f"\nPhase 2: アンサンブルモデル群")
    ensemble_models = ['RandomForest', 'GradientBoosting']

    selector2 = DoubleCV_RegressionSelector(
        outer_cv=3, inner_cv=2, n_trials=20, random_state=42, verbose=False
    )

    start_time = time.time()
    selector2.fit(X, y, model_subset=ensemble_models)
    phase2_time = time.time() - start_time

    print(f"実行時間: {phase2_time:.1f}秒")
    results2 = selector2.get_results_summary()
    print(results2.round(4))

    # その他のモデル群
    print(f"\nPhase 3: その他のモデル群")
    other_models = ['SVR', 'DecisionTree']

    selector3 = DoubleCV_RegressionSelector(
        outer_cv=3, inner_cv=2, n_trials=15, random_state=42, verbose=False
    )

    start_time = time.time()
    selector3.fit(X, y, model_subset=other_models)
    phase3_time = time.time() - start_time

    print(f"実行時間: {phase3_time:.1f}秒")
    results3 = selector3.get_results_summary()
    print(results3.round(4))

    # 総合比較
    print(f"\n🏆 総合比較結果")
    print("=" * 50)

    # 各フェーズの最良モデルを抽出
    best_models = {
        'Phase 1 (高速)': {
            'model': selector1.best_model_name_,
            'rmse': selector1.results_[selector1.best_model_name_]['mean_rmse'],
            'r2': selector1.results_[selector1.best_model_name_]['mean_r2'],
            'time': phase1_time
        },
        'Phase 2 (アンサンブル)': {
            'model': selector2.best_model_name_,
            'rmse': selector2.results_[selector2.best_model_name_]['mean_rmse'],
            'r2': selector2.results_[selector2.best_model_name_]['mean_r2'],
            'time': phase2_time
        },
        'Phase 3 (その他)': {
            'model': selector3.best_model_name_,
            'rmse': selector3.results_[selector3.best_model_name_]['mean_rmse'],
            'r2': selector3.results_[selector3.best_model_name_]['mean_r2'],
            'time': phase3_time
        }
    }

    comparison_df = pd.DataFrame(best_models).T
    print(comparison_df.round(4))

    # 総合優勝者を決定
    all_rmse = [info['rmse'] for info in best_models.values()]
    best_phase = list(best_models.keys())[np.argmin(all_rmse)]
    champion = best_models[best_phase]

    print(f"\n🥇 総合優勝: {champion['model']} ({best_phase})")
    print(f"   RMSE: {champion['rmse']:.4f}")
    print(f"   R²: {champion['r2']:.4f}")
    print(f"   実行時間: {champion['time']:.1f}秒")

    # 実用性評価
    print(f"\n💡 実用性評価")
    print("=" * 30)

    best_rmse = champion['rmse']
    best_r2 = champion['r2']

    print(f"予測精度:")
    print(f"  - 決定係数 R²: {best_r2:.3f} ({best_r2*100:.1f}%の分散を説明)")
    print(f"  - RMSE: ${best_rmse:.3f}万 (約{best_rmse*100:.0f}万円)")
    print(f"  - 相対誤差: 平均価格の{(best_rmse/y.mean()*100):.1f}%")

    accuracy_level = ""
    if best_r2 > 0.8:
        accuracy_level = "非常に高精度"
    elif best_r2 > 0.7:
        accuracy_level = "高精度"
    elif best_r2 > 0.6:
        accuracy_level = "中程度の精度"
    else:
        accuracy_level = "低精度"

    print(f"\n総合評価: {accuracy_level}")

    # ビジネス的解釈
    print(f"\n🏢 ビジネス的解釈")
    print("-" * 20)
    print(f"• 住宅価格の約{best_r2*100:.0f}%を説明可能")
    print(f"• 平均的な予測誤差は約{best_rmse*100:.0f}万円")
    print(f"• 実用的な住宅価格推定システムとして利用可能")

    if best_rmse/y.mean() < 0.2:
        print("• 誤差率20%未満で高い実用性")
    elif best_rmse/y.mean() < 0.3:
        print("• 誤差率30%未満で実用的")
    else:
        print("• 誤差率が高く、慎重な利用が必要")

    # 特徴量重要度（可能な場合）
    best_selector = selector1 if best_phase == 'Phase 1 (高速)' else \
                   selector2 if best_phase == 'Phase 2 (アンサンブル)' else selector3

    if hasattr(best_selector.best_model_, 'feature_importances_'):
        print(f"\n📈 特徴量重要度 ({champion['model']})")
        print("-" * 30)
        importances = best_selector.best_model_.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'Description': [feature_descriptions[f] for f in feature_names]
        }).sort_values('Importance', ascending=False)

        for _, row in importance_df.iterrows():
            print(f"{row['Feature']:12} {row['Importance']:.3f} - {row['Description']}")

    total_time = phase1_time + phase2_time + phase3_time
    print(f"\n⏱️  総実行時間: {total_time:.1f}秒")
    print("=" * 80)

    return best_selector

if __name__ == "__main__":
    generate_analysis_report()