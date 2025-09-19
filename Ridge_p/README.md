# p値付きRidge回帰

各特徴量の統計的有意性検定（p値）を含むRidge回帰のスクラッチ実装です。

## 概要

このプロジェクトは、標準的なRidge回帰を拡張し、各特徴量の係数に対してp値を計算するカスタムRidge回帰実装（`RidgeRegression_p`）を提供します。これにより、モデル内の予測変数の統計的有意性を評価できます。

## 機能

- **スクラッチからのRidge回帰**: scikit-learnのRidge実装に依存しない独自実装
- **p値計算**: 各特徴量の統計的有意性検定
- **標準誤差の計算**: 係数の信頼区間を提供
- **scikit-learn互換API**: おなじみの`fit()`、`predict()`、`coef_`、`intercept_`インターフェース
- **正則化パラメータ調整**: 異なるalpha値をサポート
- **デモ関数**: 組み込まれた例と比較

## 主な利点

- **統計的推論**: 標準的なRidge回帰とは異なり、どの特徴量が統計的に有意かを判断するp値を提供
- **特徴選択**: モデルに意味のある貢献をする予測変数を特定
- **モデル解釈**: 統計的裏付けのある特徴量重要度の理解向上

## インストール

### 依存関係

```bash
pip install numpy scipy matplotlib scikit-learn
```

### 必要なライブラリ

- `numpy`: 数学演算のコア
- `scipy.stats`: p値計算のための統計分布
- `matplotlib`: 可視化（デモ関数用）
- `scikit-learn`: データ生成と前処理ユーティリティ

## 使用方法

### 基本的な使用方法

```python
import numpy as np
from ridge_regression import RidgeRegression_p

# サンプルデータの生成
X = np.random.randn(100, 3)
y = 2*X[:, 0] + 0.5*X[:, 1] + 0.1*X[:, 2] + np.random.randn(100)

# モデルの作成と学習
ridge = RidgeRegression_p(alpha=1.0)
ridge.fit(X, y)

# 予測の取得
y_pred = ridge.predict(X)

# 係数とp値へのアクセス
print("係数:", ridge.coef_)
print("切片:", ridge.intercept_)
print("p値:", ridge.get_pvalues())
print("標準誤差:", ridge.se_)
```

### 特徴量サマリー

```python
# 包括的な特徴量分析の取得
summary = ridge.get_feature_summary()
print(summary)
# 出力: {'coefficients': array([...]), 'pvalues': array([...]), 'standard_errors': array([...])}
```

### デモ関数の実行

```python
from ridge_regression import ridge_regression_demo, compare_with_linear_regression

# 異なるalpha値でのRidge回帰のデモンストレーション
ridge_regression_demo()

# Ridge回帰と線形回帰の比較
compare_with_linear_regression()
```

## クラスリファレンス

### `RidgeRegression_p`

#### コンストラクタ
```python
RidgeRegression_p(alpha=1.0)
```

**パラメータ:**
- `alpha` (float): 正則化強度。値が高いほど強い正則化を指定します。

#### メソッド

##### `fit(X, y)`
Ridge回帰モデルを学習します。

**パラメータ:**
- `X` (array-like): 形状が(n_samples, n_features)の学習特徴量
- `y` (array-like): 形状が(n_samples,)の目標値

**戻り値:**
- `self`: 学習済み推定器を返します

##### `predict(X)`
Ridge回帰モデルを使用して予測を行います。

**パラメータ:**
- `X` (array-like): 形状が(n_samples, n_features)の特徴量

**戻り値:**
- `y_pred` (array): 予測値

##### `get_pvalues()`
各特徴量係数のp値を取得します。

**戻り値:**
- `pvalues` (array): 各特徴量のp値

##### `get_feature_summary()`
特徴量統計の包括的なサマリーを取得します。

**戻り値:**
- `dict`: 係数、p値、標準誤差を含む辞書

#### 属性

- `coef_` (array): 学習後の係数重み
- `intercept_` (float): 切片項
- `se_` (array): 係数の標準誤差
- `pvalues_` (array): 全パラメータのp値（切片を含む）

## 数学的背景

### Ridge回帰の公式

Ridge回帰は以下を最小化します：
```
||y - Xβ||² + α||β||²
```

解は以下で与えられます：
```
β = (X'X + αI)⁻¹X'y
```

### p値の計算

p値はt分布を使用して計算されます：

1. **標準誤差**: `SE = √(MSE × diag((X'X + αI)⁻¹))`
2. **t統計量**: `t = β / SE`
3. **p値**: `p = 2 × (1 - CDF(|t|, df=n-p))`

ここで：
- `MSE`: 残差の平均二乗誤差
- `n`: サンプル数
- `p`: パラメータ数

## 解釈ガイド

### p値の有意水準
- **p < 0.001**: 高度に有意 (***)
- **p < 0.01**: 非常に有意 (**)
- **p < 0.05**: 有意 (*)
- **p ≥ 0.05**: 有意でない

### 係数の解釈
- **正の係数**: 特徴量が目標変数を増加させる
- **負の係数**: 特徴量が目標変数を減少させる
- **大きさ**: |係数|が大きいほど強い効果を示す（特徴量が標準化されている場合）

## 出力例

```
Ridge Regression Results:
----------------------------------------
Alpha:   0.01 | MSE:   45.23 | R²:  0.872 | Coef:    1.943 | p-value: 0.000
Alpha:   0.10 | MSE:   45.67 | R²:  0.869 | Coef:    1.891 | p-value: 0.000
Alpha:   1.00 | MSE:   48.12 | R²:  0.853 | Coef:    1.654 | p-value: 0.000

Ridge Regression coefficients:
  Feature 1: 1.943 (p-value: 0.000) ***
  Feature 2: 0.487 (p-value: 0.002) **
  Feature 3: 0.123 (p-value: 0.245)
```

## 制限事項と考慮点

1. **正則化の効果**: Ridge回帰は係数を縮小するため、p値の計算に影響を与える可能性があります
2. **多重共線性**: 特徴量間の高い相関は不安定なp値につながる可能性があります
3. **サンプルサイズ**: 小さなサンプルは信頼性の低いp値を生成する可能性があります
4. **仮定**: 線形関係と正規分布する残差を仮定しています

## ファイル構造

```
Ridge_p/
├── ridge_regression.py    # メイン実装
└── README.md             # このドキュメント
```

## 貢献

これは教育と研究目的のためのカスタム実装です。この実装はRidge回帰とp値計算の標準的な統計手法に従っています。

## ライセンス

このプロジェクトは教育および研究目的でそのまま提供されています。