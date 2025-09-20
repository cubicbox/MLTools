# ダブルクロスバリデーション回帰モデル選択 (Optuna版)

複数の回帰モデルを公平に比較し、最適なモデルとハイパーパラメータを選択するためのダブルクロスバリデーション（ネストされたクロスバリデーション）システムです。Optunaによる効率的なハイパーパラメータ最適化を採用しています。

## 概要

このプロジェクトは、機械学習における重要な問題である「モデル選択バイアス」を回避し、真の汎化性能を推定するためのダブルクロスバリデーションを実装しています。複数の回帰アルゴリズムを同一条件で評価し、統計的に信頼性の高いモデル選択を行います。

## 主な特徴

- **バイアスのないモデル評価**: ダブルクロスバリデーションによる真の汎化性能推定
- **包括的なモデル比較**: 10種類の回帰アルゴリズムを標準サポート
- **Optuna自動最適化**: TPE（Tree-structured Parzen Estimator）による効率的なハイパーパラメータ最適化
- **統計的評価**: 複数の評価指標による性能比較
- **特徴量重要度分析**: モデル解釈のための重要度抽出
- **可視化機能**: 結果の直感的な理解のためのグラフ出力

## ダブルクロスバリデーションとは

### 通常のクロスバリデーションの問題点
通常のクロスバリデーションでハイパーパラメータを最適化し、同じデータで性能評価を行うと、**選択バイアス**により過度に楽観的な性能推定となります。

### ダブルクロスバリデーションの解決策
```
外側ループ: モデル性能の真の評価
    ├── 内側ループ: ハイパーパラメータ最適化
    └── テストセットで性能評価（バイアスなし）
```

この二重構造により、モデル選択とハイパーパラメータ最適化を行いながら、バイアスのない性能推定が可能になります。

## Optunaによるハイパーパラメータ最適化

### GridSearchとの比較

| 手法 | 利点 | 欠点 |
|------|------|------|
| **GridSearch** | 全探索による確実性 | 組み合わせ爆発、時間がかかる |
| **Optuna (TPE)** | 効率的な探索、早期収束 | 確率的手法、局所最適の可能性 |

### Optunaの利点

1. **効率的な探索**: TPE（Tree-structured Parzen Estimator）により、過去の試行結果を活用した効率的な探索
2. **動的な探索範囲**: 連続値、カテゴリカル値、条件付きパラメータに対応
3. **早期停止**: 有望でない試行の早期打ち切り（Pruning）
4. **スケーラビリティ**: 大規模なパラメータ空間でも効率的に動作

## サポートされる回帰モデル

| モデル | 説明 | ハイパーパラメータ |
|--------|------|-------------------|
| **LinearRegression** | 線形回帰 | なし |
| **Ridge** | L2正則化線形回帰 | alpha |
| **Lasso** | L1正則化線形回帰 | alpha |
| **ElasticNet** | L1+L2正則化線形回帰 | alpha, l1_ratio |
| **RandomForest** | ランダムフォレスト | n_estimators, max_depth, min_samples_split |
| **GradientBoosting** | 勾配ブースティング | n_estimators, learning_rate, max_depth |
| **SVR** | サポートベクター回帰 | C, gamma, kernel |
| **KNN** | k近傍法 | n_neighbors, weights |
| **DecisionTree** | 決定木 | max_depth, min_samples_split, min_samples_leaf |
| **GaussianProcess** | ガウス過程回帰 | kernel (RBF/Matern/RationalQuadratic), alpha, n_restarts_optimizer |

### Gaussian Process Regression の特徴

**ガウス過程回帰（GPR）**は、予測の不確実性も同時に推定できる強力な回帰手法です：

- **不確実性推定**: 各予測に対して信頼区間を提供
- **非線形関係**: カーネル関数により複雑な非線形パターンを学習
- **少数サンプル**: 小規模データでも効果的
- **カーネル選択**: RBF、Matern、Rational Quadraticカーネルから自動選択

**注意点**:
- 計算量が O(n³) で、大規模データでは時間がかかる
- ハイパーパラメータが多く、最適化に時間を要する場合がある

## インストール

### 依存関係

**uvを使用する場合（推奨）：**

```bash
# プロジェクトディレクトリで
uv init  # pyproject.toml を作成（初回のみ）
uv add numpy pandas scikit-learn matplotlib optuna
```

または、既存環境への直接インストール：
```bash
uv pip install numpy pandas scikit-learn matplotlib optuna
```

**従来のpipの場合：**
```bash
pip install numpy pandas scikit-learn matplotlib optuna
```

### uvの利点

- **高速**: pipより大幅に高速なパッケージインストール
- **信頼性**: 決定論的な依存関係解決
- **プロジェクト管理**: pyproject.tomlによる依存関係管理
- **環境分離**: プロジェクトごとの独立した環境

### 必要なライブラリ

- `numpy`: 数値計算
- `pandas`: データ処理と結果表示
- `scikit-learn`: 機械学習アルゴリズムとクロスバリデーション
- `matplotlib`: 結果の可視化（オプション）
- `optuna`: ハイパーパラメータ最適化（必須）

## 使用方法

### 基本的な使用例

```python
import numpy as np
from double_cv_regression import DoubleCV_RegressionSelector
from sklearn.datasets import make_regression

# サンプルデータ生成
X, y = make_regression(n_samples=300, n_features=10, noise=0.1, random_state=42)

# モデル選択器の作成
selector = DoubleCV_RegressionSelector(
    outer_cv=5,     # 外側クロスバリデーション
    inner_cv=3,     # 内側クロスバリデーション
    n_trials=100,   # Optuna最適化試行回数
    random_state=42,
    verbose=True
)

# モデル選択・評価の実行
selector.fit(X, y)

# 結果の表示
selector.print_detailed_results()

# 最良モデルで予測
y_pred = selector.predict(X_new)
```

### 特定モデルのみ評価

```python
# 計算時間短縮のため、一部モデルのみ評価
model_subset = ['LinearRegression', 'Ridge', 'RandomForest', 'SVR', 'GaussianProcess']
selector.fit(X, y, model_subset=model_subset)
```

### 結果の取得と可視化

```python
# 結果サマリーの取得
results_df = selector.get_results_summary()
print(results_df)

# グラフによる可視化
selector.plot_results()

# 最良モデルの詳細
print(f"最適モデル: {selector.best_model_name_}")
print(f"モデル: {selector.best_model_}")
```

## クラスリファレンス

### `DoubleCV_RegressionSelector`

#### コンストラクタ

```python
DoubleCV_RegressionSelector(
    outer_cv=5,
    inner_cv=3,
    n_trials=100,
    random_state=42,
    n_jobs=-1,
    verbose=True
)
```

**パラメータ:**
- `outer_cv` (int): 外側クロスバリデーションのフォールド数
- `inner_cv` (int): 内側クロスバリデーションのフォールド数
- `n_trials` (int): Optunaのハイパーパラメータ最適化試行回数
- `random_state` (int): 再現性のための乱数シード
- `n_jobs` (int): 並列処理のジョブ数
- `verbose` (bool): 詳細な出力の有無

#### メソッド

##### `fit(X, y, model_subset=None)`
ダブルクロスバリデーションによるモデル選択・評価を実行

**パラメータ:**
- `X` (array-like): 特徴量行列
- `y` (array-like): 目標変数
- `model_subset` (list, optional): 評価するモデル名のリスト

##### `predict(X)`
最良モデルで予測を実行

**パラメータ:**
- `X` (array-like): 予測対象の特徴量

**戻り値:**
- `y_pred` (array): 予測値

##### `get_results_summary()`
結果サマリーをDataFrameで取得

**戻り値:**
- `DataFrame`: モデル別の性能統計

##### `print_detailed_results()`
詳細な結果を標準出力に表示

##### `plot_results()`
結果をグラフで可視化（matplotlib必要）

#### 属性

- `results_` (dict): 各モデルの詳細な評価結果
- `best_model_` (object): 最良モデルのインスタンス
- `best_model_name_` (str): 最良モデル名
- `feature_importance_` (array): 特徴量重要度（利用可能な場合）

## 評価指標

### 使用される評価指標

1. **RMSE (Root Mean Squared Error)**: 予測誤差の二乗平均の平方根
2. **R² Score**: 決定係数（1に近いほど良い）
3. **MAE (Mean Absolute Error)**: 絶対誤差の平均

### 結果の解釈

- **RMSE**: 小さいほど良い（目標変数と同じ単位）
- **R²**: 1に近いほど良い（0～1、負の値も可能）
- **MAE**: 小さいほど良い（外れ値に対してRMSEより頑健）

## 出力例

```
==========================================
ダブルクロスバリデーション詳細結果
==========================================

1. Ridge
----------------------------------------
RMSE:  0.3247 ± 0.0156
R²:    0.8934 ± 0.0089
MAE:   0.2634 ± 0.0134
★ 最適モデル ★

2. RandomForest
----------------------------------------
RMSE:  0.3891 ± 0.0203
R²:    0.8456 ± 0.0156
MAE:   0.3002 ± 0.0178

3. LinearRegression
----------------------------------------
RMSE:  0.4123 ± 0.0189
R²:    0.8234 ± 0.0145
MAE:   0.3234 ± 0.0167

最終選択モデル: Ridge
期待RMSE: 0.3247
期待R²: 0.8934
```

## 計算時間の最適化

### 計算時間短縮の方法

1. **フォールド数の調整**
   ```python
   # 高速評価（精度は低下）
   selector = DoubleCV_RegressionSelector(outer_cv=3, inner_cv=2)
   ```

2. **Optuna試行回数の調整**
   ```python
   # 高速評価（精度は低下）
   selector = DoubleCV_RegressionSelector(n_trials=50)  # デフォルト100
   # 高精度評価（時間増加）
   selector = DoubleCV_RegressionSelector(n_trials=200)
   ```

3. **モデルの限定**
   ```python
   # 軽量モデルのみ
   fast_models = ['LinearRegression', 'Ridge', 'Lasso']
   selector.fit(X, y, model_subset=fast_models)
   ```

4. **並列処理の活用**
   ```python
   # CPUコア数に応じて調整
   selector = DoubleCV_RegressionSelector(n_jobs=-1)  # 全コア使用
   ```

### 推奨設定

| データサイズ | outer_cv | inner_cv | n_trials | 推奨モデル数 | 想定時間 |
|-------------|----------|----------|----------|-------------|----------|
| 小（～1000） | 5 | 3 | 100 | 全て（10種） | 5-15分 |
| 中（1000～10000） | 5 | 3 | 50-100 | 6-7種（GPR除外推奨） | 10-20分 |
| 大（10000～） | 3 | 2 | 50 | 4-5種（GPR除外必須） | 15-30分 |

## 実用例

### 1. 基本的な回帰問題

```python
# 住宅価格予測など
from sklearn.datasets import load_boston
X, y = load_boston(return_X_y=True)

selector = DoubleCV_RegressionSelector()
selector.fit(X, y)
```

### 2. 大規模データセット

```python
# 計算時間を考慮した設定
selector = DoubleCV_RegressionSelector(
    outer_cv=3,
    inner_cv=2,
    n_trials=50,  # 試行回数を削減
    n_jobs=-1
)

# 高速モデルのみ
fast_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']
selector.fit(X, y, model_subset=fast_models)
```

### 3. カスタム評価

```python
# 結果を詳細分析
results_df = selector.get_results_summary()

# 特定の指標でソート
best_r2_model = results_df.loc[results_df['Mean_R2'].idxmax(), 'Model']
best_mae_model = results_df.loc[results_df['Mean_MAE'].idxmin(), 'Model']

print(f"R²最高: {best_r2_model}")
print(f"MAE最小: {best_mae_model}")
```

## 注意事項と制限

### 統計的な考慮事項

1. **サンプルサイズ**: 小さなデータセット（n<100）では結果が不安定になる可能性
2. **特徴量数**: 特徴量がサンプル数に比して多い場合、過学習のリスク
3. **データの分布**: 極端な外れ値や不均衡なデータでは性能評価が偏る可能性

### 計算上の制限

1. **計算時間**: ダブルクロスバリデーションは単一CVより時間がかかる
2. **メモリ使用量**: 大規模データでは十分なRAMが必要
3. **並列処理**: 一部のモデル（SVR等）で並列処理の効果が限定的

### 実装上の注意

```python
# 悪い例: データリークのリスク
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 全データで標準化
selector.fit(X_scaled, y)  # データリーク発生

# 良い例: 内部で適切に処理
selector.fit(X, y)  # クラス内で適切に標準化
```

## トラブルシューティング

### よくある問題と解決策

1. **収束しないモデル**
   ```python
   # Lasso/ElasticNetのmax_iter警告
   # → models定義を変更してmax_iterを増加
   ```

2. **メモリ不足**
   ```python
   # n_jobsを削減
   selector = DoubleCV_RegressionSelector(n_jobs=1)
   ```

3. **計算時間が長すぎる**
   ```python
   # フォールド数削減 + モデル限定
   selector = DoubleCV_RegressionSelector(outer_cv=3, inner_cv=2)
   selector.fit(X, y, model_subset=['Ridge', 'RandomForest'])
   ```

## ファイル構造

```
DoubleCV_Regression/
├── double_cv_regression.py    # メイン実装
└── README.md                  # このドキュメント
```

## 今後の拡張予定

- [ ] 分類問題への対応
- [ ] カスタムモデルの追加機能
- [ ] 特徴量選択との統合
- [ ] より詳細な統計的検定機能
- [ ] 結果のエクスポート機能

## 貢献とライセンス

このプロジェクトは教育・研究目的で開発されました。機械学習のベストプラクティスに基づいた実装を提供し、モデル選択における統計的な正確性を重視しています。

## 参考文献

1. Varma, S., & Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. BMC bioinformatics, 7(1), 91.
2. Cawley, G. C., & Talbot, N. L. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. Journal of Machine Learning Research, 11(Jul), 2079-2107.