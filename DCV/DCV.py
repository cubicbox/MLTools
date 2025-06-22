# %%
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from sklearn.metrics import accuracy_score
import warnings
import optuna


def double_cross_validation(
    estimator,
    X,
    y,
    outer_cv=5,
    inner_cv=3,
    param_distributions=None,
    n_trials=100,
    scoring="accuracy",
    return_estimators=False,
    verbose=False,
    optuna_sampler=None,
):
    """
    ダブルクロスバリデーション（Nested Cross-Validation）をOptunaで実装

    Parameters:
    -----------
    estimator : sklearn estimator
        使用する推定器（分類器・回帰器）
    X : array-like, shape (n_samples, n_features)
        特徴量
    y : array-like, shape (n_samples,)
        目的変数
    outer_cv : int or cross-validation generator
        外側のクロスバリデーションの分割数またはCV生成器
    inner_cv : int or cross-validation generator
        内側のクロスバリデーションの分割数またはCV生成器
    param_distributions : dict or callable or None
        ハイパーパラメータの分布定義。dictまたはObjective関数。Noneの場合はデフォルトパラメータを使用
    n_trials : int
        Optunaの試行回数
    scoring : str or callable
        評価指標
    return_estimators : bool
        学習済み推定器を返すかどうか
    verbose : bool
        詳細な出力を行うかどうか
    optuna_sampler : optuna.samplers.BaseSampler or None
        Optunaのサンプラー。Noneの場合はTPESamplerを使用

    Returns:
    --------
    scores : array
        各外側フォールドでのスコア
    mean_score : float
        平均スコア
    std_score : float
        スコアの標準偏差
    best_params_list : list
        各フォールドでの最適パラメータ
    best_values_list : list
        各フォールドでの最適値
    studies : list
        各フォールドのOptunaスタディ
    estimators : list (optional)
        学習済み推定器のリスト（return_estimators=Trueの場合）
    """

    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.utils import check_X_y
    from sklearn.base import is_classifier

    # データの検証
    X, y = check_X_y(X, y)

    # CV生成器の設定
    if isinstance(outer_cv, int):
        if is_classifier(estimator):
            outer_cv = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        else:
            outer_cv = KFold(n_splits=outer_cv, shuffle=True, random_state=42)

    if isinstance(inner_cv, int):
        if is_classifier(estimator):
            inner_cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)
        else:
            inner_cv = KFold(n_splits=inner_cv, shuffle=True, random_state=42)

    # Optunaのサンプラー設定
    if optuna_sampler is None:
        optuna_sampler = optuna.samplers.TPESampler(seed=42)

    # 結果格納用
    outer_scores = []
    best_params_list = []
    best_values_list = []
    studies = []
    estimators = [] if return_estimators else None

    if verbose:
        print("ダブルクロスバリデーション開始（Optuna使用）...")
        print(f"外側CV: {outer_cv.n_splits}分割, 内側CV: {inner_cv.n_splits}分割")
        print(f"Optuna試行回数: {n_trials}")

    # Optunaのログレベル設定（冗長な出力を抑制）
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    # 外側ループ
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        if verbose:
            print(f"\n=== 外側フォールド {fold_idx + 1}/{outer_cv.n_splits} ===")

        # データ分割
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # ハイパーパラメータ調整（内側ループ）
        if param_distributions is not None:
            if verbose:
                print("Optunaでハイパーパラメータ調整中...")

            # Objective関数の定義
            def objective(trial):
                # パラメータ分布の定義
                if callable(param_distributions):
                    # 関数として定義されている場合
                    params = param_distributions(trial)
                else:
                    # 辞書として定義されている場合
                    params = {}
                    for param_name, param_config in param_distributions.items():
                        if isinstance(param_config, dict):
                            if param_config["type"] == "int":
                                params[param_name] = trial.suggest_int(
                                    param_name,
                                    param_config["low"],
                                    param_config["high"],
                                )
                            elif param_config["type"] == "float":
                                if param_config.get("log", False):
                                    params[param_name] = trial.suggest_float(
                                        param_name,
                                        param_config["low"],
                                        param_config["high"],
                                        log=True,
                                    )
                                else:
                                    params[param_name] = trial.suggest_float(
                                        param_name,
                                        param_config["low"],
                                        param_config["high"],
                                    )
                            elif param_config["type"] == "categorical":
                                params[param_name] = trial.suggest_categorical(
                                    param_name, param_config["choices"]
                                )
                        else:
                            # 直接リストが指定された場合（カテゴリカル扱い）
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_config
                            )

                # 推定器にパラメータを設定
                estimator_clone = clone(estimator)
                estimator_clone.set_params(**params)

                # 内側クロスバリデーション
                scores = cross_val_score(
                    estimator_clone,
                    X_train_outer,
                    y_train_outer,
                    cv=inner_cv,
                    scoring=scoring,
                    n_jobs=-1,
                )

                return scores.mean()

            # Optunaスタディの作成と最適化
            study = optuna.create_study(
                direction=(
                    "maximize"
                    if scoring in ["accuracy", "f1", "roc_auc", "precision", "recall"]
                    else "minimize"
                ),
                sampler=optuna_sampler,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

            best_params = study.best_params
            best_value = study.best_value
            best_params_list.append(best_params)
            best_values_list.append(best_value)
            studies.append(study)

            # 最適パラメータで推定器を学習
            estimator_clone = clone(estimator)
            estimator_clone.set_params(**best_params)
            estimator_clone.fit(X_train_outer, y_train_outer)
            best_estimator = estimator_clone

            if verbose:
                print(f"最適パラメータ: {best_params}")
                print(f"内側CV最良スコア: {best_value:.4f}")
        else:
            # パラメータ分布が指定されていない場合はデフォルトパラメータを使用
            estimator_clone = clone(estimator)
            estimator_clone.fit(X_train_outer, y_train_outer)
            best_estimator = estimator_clone
            best_params_list.append({})
            best_values_list.append(None)
            studies.append(None)

        # テストデータで評価
        if hasattr(best_estimator, "predict_proba") and scoring == "roc_auc":
            # ROC-AUCの場合は確率予測を使用
            y_pred_proba = best_estimator.predict_proba(X_test_outer)[:, 1]
            from sklearn.metrics import roc_auc_score

            score = roc_auc_score(y_test_outer, y_pred_proba)
        else:
            # その他の指標
            y_pred = best_estimator.predict(X_test_outer)
            if scoring == "accuracy":
                score = accuracy_score(y_test_outer, y_pred)
            elif scoring == "neg_mean_squared_error":
                from sklearn.metrics import mean_squared_error

                score = -mean_squared_error(y_test_outer, y_pred)
            else:
                # sklearn.metricsから対応する関数を取得
                from sklearn.metrics import get_scorer

                scorer = get_scorer(scoring)
                score = scorer(best_estimator, X_test_outer, y_test_outer)

        outer_scores.append(score)

        if return_estimators:
            estimators.append(best_estimator)

        if verbose:
            print(f"外側テストスコア: {score:.4f}")

    # 結果の集計
    outer_scores = np.array(outer_scores)
    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)

    if verbose:
        print(f"\n=== 最終結果 ===")
        print(f"各フォールドスコア: {outer_scores}")
        print(f"平均スコア: {mean_score:.4f} ± {std_score:.4f}")

    # 結果の構築
    results = {
        "scores": outer_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "best_params_list": best_params_list,
        "best_values_list": best_values_list,
        "studies": studies,
    }

    if return_estimators:
        results["estimators"] = estimators

    return results


def create_param_distributions_rf():
    """RandomForestClassifier用のパラメータ分布定義例"""
    return {
        "n_estimators": {"type": "int", "low": 10, "high": 200},
        "max_depth": {"type": "int", "low": 3, "high": 20},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
        "max_features": {"type": "categorical", "choices": ["sqrt", "log2", None]},
    }


def create_param_distributions_svm():
    """SVC用のパラメータ分布定義例"""
    return {
        "C": {"type": "float", "low": 0.01, "high": 100, "log": True},
        "gamma": {"type": "float", "low": 1e-6, "high": 1e-1, "log": True},
        "kernel": {"type": "categorical", "choices": ["rbf", "linear", "poly"]},
    }


def create_param_distributions_xgb():
    """XGBoost用のパラメータ分布定義例（関数形式）"""

    def objective_func(trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    return objective_func


# 使用例とテスト用のコード
if __name__ == "__main__":
    from sklearn.datasets import make_classification, make_regression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import StratifiedKFold, KFold

    print("=== RandomForestClassifierでのOptuna最適化例 ===")
    # 分類データの生成
    X_clf, y_clf = make_classification(
        n_samples=300, n_features=10, n_informative=5, n_redundant=2, random_state=42
    )

    # RandomForestClassifierでのダブルCV（辞書形式）
    rf_clf = RandomForestClassifier(random_state=42)
    rf_param_dist = create_param_distributions_rf()

    rf_results = double_cross_validation(
        estimator=rf_clf,
        X=X_clf,
        y=y_clf,
        outer_cv=5,
        inner_cv=3,
        param_distributions=rf_param_dist,
        n_trials=50,
        scoring="accuracy",
        verbose=True,
    )

    print(
        f"\nRandomForest最終結果: {rf_results['mean_score']:.4f} ± {rf_results['std_score']:.4f}"
    )

    print("\n=== SVCでのOptuna最適化例 ===")
    # SVCでのダブルCV
    svm_clf = SVC(random_state=42)
    svm_param_dist = create_param_distributions_svm()

    svm_results = double_cross_validation(
        estimator=svm_clf,
        X=X_clf,
        y=y_clf,
        outer_cv=3,  # 計算時間短縮のため3分割
        inner_cv=3,
        param_distributions=svm_param_dist,
        n_trials=30,
        scoring="accuracy",
        verbose=True,
    )

    print(
        f"\nSVM最終結果: {svm_results['mean_score']:.4f} ± {svm_results['std_score']:.4f}"
    )

    print("\n=== カスタムサンプラーの例 ===")
    # カスタムサンプラーを使用
    from optuna.samplers import RandomSampler

    custom_sampler = RandomSampler(seed=123)

    # 簡単なパラメータ分布（リスト形式）
    simple_param_dist = {
        "n_estimators": [50, 100, 150, 200],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    simple_results = double_cross_validation(
        estimator=RandomForestClassifier(random_state=42),
        X=X_clf,
        y=y_clf,
        outer_cv=3,
        inner_cv=3,
        param_distributions=simple_param_dist,
        n_trials=20,
        scoring="accuracy",
        optuna_sampler=custom_sampler,
        return_estimators=True,
        verbose=True,
    )

    print(
        f"\nSimple設定結果: {simple_results['mean_score']:.4f} ± {simple_results['std_score']:.4f}"
    )
    print(f"学習済みモデル数: {len(simple_results['estimators'])}")

    print("\n=== 回帰タスクの例 ===")
    # 回帰データの生成
    X_reg, y_reg = make_regression(
        n_samples=200, n_features=10, noise=0.1, random_state=42
    )

    # RidgeでのダブルCV
    ridge = Ridge()
    ridge_param_dist = {
        "alpha": {"type": "float", "low": 0.001, "high": 100, "log": True}
    }

    ridge_results = double_cross_validation(
        estimator=ridge,
        X=X_reg,
        y=y_reg,
        outer_cv=5,
        inner_cv=3,
        param_distributions=ridge_param_dist,
        n_trials=30,
        scoring="neg_mean_squared_error",
        verbose=True,
    )

    print(
        f"\nRidge最終結果: {ridge_results['mean_score']:.4f} ± {ridge_results['std_score']:.4f}"
    )

    print("\n=== 最適化履歴の分析例 ===")
    # 最初のフォールドの最適化履歴を表示
    if rf_results["studies"][0] is not None:
        study = rf_results["studies"][0]
        print(f"最適化試行回数: {len(study.trials)}")
        print(f"最良値: {study.best_value:.4f}")
        print(f"最良パラメータ: {study.best_params}")

        # 最適化履歴の可視化（optunaの可視化機能を使用）
        try:
            import optuna.visualization as vis

            # 注意: この部分は実際の環境では動作しますが、
            # ここではコメントアウトしています
            # fig = vis.plot_optimization_history(study)
            # fig.show()
        except ImportError:
            print("可視化にはoptuna[visualization]が必要です")
# %%
