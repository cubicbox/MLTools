"""
ダブルクロスバリデーションによる回帰モデル選択・評価システム

複数の回帰モデルを公平に比較し、最適なモデルとハイパーパラメータを選択します。
ダブルクロスバリデーション（ネストされたクロスバリデーション）により、
モデル選択バイアスを回避し、真の汎化性能を推定します。
"""

import numpy as np
import pandas as pd
import warnings
import pickle
import os
from typing import List, Optional
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
import time

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Please install with: uv add optuna (or pip install optuna)")

warnings.filterwarnings('ignore')


class DoubleCV_RegressionSelector:
    """
    ダブルクロスバリデーションによる回帰モデル選択器

    主な機能:
    - 複数の回帰モデルの比較
    - ハイパーパラメータ最適化
    - ネストされたクロスバリデーション
    - モデル性能の統計的比較
    - 特徴量重要度分析
    """

    def __init__(self,
                 outer_cv: int = 5,
                 inner_cv: int = 3,
                 n_trials: int = 100,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 verbose: bool = True,
                 save_models: bool = True,
                 save_dir: str = "./models"):
        """
        Parameters:
        -----------
        outer_cv : int, default=5
            外側クロスバリデーションのフォールド数（真の性能評価用）
        inner_cv : int, default=3
            内側クロスバリデーションのフォールド数（ハイパーパラメータ最適化用）
        n_trials : int, default=100
            Optunaのハイパーパラメータ最適化試行回数
        random_state : int, default=42
            再現性のための乱数シード
        n_jobs : int, default=-1
            並列処理のジョブ数
        verbose : bool, default=True
            詳細な出力を行うか
        save_models : bool, default=True
            訓練済みモデルをpklファイルとして保存するか
        save_dir : str, default="./models"
            モデル保存ディレクトリ
        """
        self.outer_cv = outer_cv
        self.inner_cv = inner_cv
        self.n_trials = n_trials
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.save_models = save_models
        self.save_dir = save_dir

        # 結果保存用
        self.results_ = {}
        self.best_model_ = None
        self.best_model_name_ = None
        self.feature_importance_ = None
        self.trained_models_ = {}  # 全ての訓練済みモデルを保存

        # 保存ディレクトリの作成
        if self.save_models:
            os.makedirs(self.save_dir, exist_ok=True)

        # モデル定義
        self._define_models()

    def _define_models(self):
        """使用する回帰モデルとOptunaパラメータ最適化関数を定義"""

        def suggest_ridge_params(trial):
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True)
            }

        def suggest_lasso_params(trial):
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 100.0, log=True),
                'max_iter': 2000
            }

        def suggest_elasticnet_params(trial):
            return {
                'alpha': trial.suggest_float('alpha', 0.01, 10.0, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.1, 0.9),
                'max_iter': 2000
            }

        def suggest_randomforest_params(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            # max_depthの条件分岐
            if trial.suggest_categorical('max_depth_type', ['limited', 'unlimited']) == 'limited':
                params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            # unlimitedの場合はmax_depth=Noneなので、パラメータに含めない
            return params

        def suggest_gradientboosting_params(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }

        def suggest_svr_params(trial):
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_float('gamma', 1e-5, 1.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            }

        def suggest_knn_params(trial):
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
            }

        def suggest_decisiontree_params(trial):
            params = {
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            # max_depthの条件分岐
            if trial.suggest_categorical('max_depth_type', ['limited', 'unlimited']) == 'limited':
                params['max_depth'] = trial.suggest_int('max_depth', 3, 20)

            # max_featuresの条件分岐
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 'none'])
            if max_features != 'none':
                params['max_features'] = max_features

            return params

        def suggest_gpr_params(trial):
            # カーネル選択
            kernel_type = trial.suggest_categorical('kernel_type', ['rbf', 'matern', 'rational_quadratic'])

            if kernel_type == 'rbf':
                # RBFカーネル
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
                kernel = ConstantKernel(1.0) * RBF(length_scale=length_scale) + WhiteKernel(1e-5)

            elif kernel_type == 'matern':
                # Maternカーネル
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
                nu = trial.suggest_categorical('nu', [0.5, 1.5, 2.5])
                kernel = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=nu) + WhiteKernel(1e-5)

            else:  # rational_quadratic
                # Rational Quadraticカーネル
                length_scale = trial.suggest_float('length_scale', 0.1, 10.0, log=True)
                rq_alpha = trial.suggest_float('rq_alpha', 0.1, 10.0)
                kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=length_scale, alpha=rq_alpha) + WhiteKernel(1e-5)

            return {
                'kernel': kernel,
                'alpha': trial.suggest_float('gpr_alpha', 1e-12, 1e-6, log=True),
                'n_restarts_optimizer': trial.suggest_int('n_restarts_optimizer', 0, 5)
            }

        self.models = {
            'LinearRegression': {
                'model_class': LinearRegression,
                'param_suggest_func': None  # パラメータ最適化なし
            },

            'Ridge': {
                'model_class': Ridge,
                'param_suggest_func': suggest_ridge_params
            },

            'Lasso': {
                'model_class': Lasso,
                'param_suggest_func': suggest_lasso_params
            },

            'ElasticNet': {
                'model_class': ElasticNet,
                'param_suggest_func': suggest_elasticnet_params
            },

            'RandomForest': {
                'model_class': RandomForestRegressor,
                'param_suggest_func': suggest_randomforest_params
            },

            'GradientBoosting': {
                'model_class': GradientBoostingRegressor,
                'param_suggest_func': suggest_gradientboosting_params
            },

            'SVR': {
                'model_class': SVR,
                'param_suggest_func': suggest_svr_params
            },

            'KNN': {
                'model_class': KNeighborsRegressor,
                'param_suggest_func': suggest_knn_params
            },

            'DecisionTree': {
                'model_class': DecisionTreeRegressor,
                'param_suggest_func': suggest_decisiontree_params
            },

            'GaussianProcess': {
                'model_class': GaussianProcessRegressor,
                'param_suggest_func': suggest_gpr_params
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray,
            model_subset: Optional[List[str]] = None) -> 'DoubleCV_RegressionSelector':
        """
        ダブルクロスバリデーションによるモデル選択・評価を実行

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            特徴量
        y : array-like of shape (n_samples,)
            目標変数
        model_subset : list of str, optional
            評価するモデル名のリスト。Noneの場合は全モデルを評価

        Returns:
        --------
        self : DoubleCV_RegressionSelector
        """

        if self.verbose:
            print("=" * 60)
            print("ダブルクロスバリデーション回帰モデル選択を開始")
            print("=" * 60)
            print(f"データ形状: X={X.shape}, y={y.shape}")
            print(f"外側CV: {self.outer_cv}フォールド, 内側CV: {self.inner_cv}フォールド")

        # データの前処理
        X = np.array(X)
        y = np.array(y)

        # 使用するモデルを決定
        if model_subset is None:
            models_to_evaluate = self.models
        else:
            models_to_evaluate = {k: v for k, v in self.models.items()
                                if k in model_subset}

        if self.verbose:
            print(f"評価モデル数: {len(models_to_evaluate)}")
            print(f"モデル: {list(models_to_evaluate.keys())}")
            print("-" * 60)

        # 外側クロスバリデーション
        outer_kf = KFold(n_splits=self.outer_cv,
                        shuffle=True,
                        random_state=self.random_state)

        # 各モデルの結果を保存
        model_results = {}

        for model_name, model_config in models_to_evaluate.items():
            if self.verbose:
                print(f"評価中: {model_name}")

            start_time = time.time()

            # 外側CVの各フォールドでの性能を保存
            outer_scores = {
                'mse': [],
                'rmse': [],
                'mae': [],
                'r2': []
            }

            best_params_per_fold = []

            for fold_idx, (train_idx, test_idx) in enumerate(outer_kf.split(X)):
                # データ分割
                X_train_outer, X_test_outer = X[train_idx], X[test_idx]
                y_train_outer, y_test_outer = y[train_idx], y[test_idx]

                # 標準化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_outer)
                X_test_scaled = scaler.transform(X_test_outer)

                # Optunaでハイパーパラメータ最適化
                if model_config['param_suggest_func'] is not None and OPTUNA_AVAILABLE:
                    # Optuna最適化関数を定義
                    def objective(trial):
                        params = model_config['param_suggest_func'](trial)

                        # random_stateの追加（対応モデルのみ）
                        try:
                            model = model_config['model_class'](**params, random_state=self.random_state)
                        except TypeError:
                            model = model_config['model_class'](**params)

                        # 内側クロスバリデーション
                        inner_kf = KFold(n_splits=self.inner_cv,
                                       shuffle=True,
                                       random_state=self.random_state + fold_idx)

                        scores = cross_val_score(model, X_train_scaled, y_train_outer,
                                               cv=inner_kf, scoring='neg_mean_squared_error',
                                               n_jobs=1)  # Optunaは並列化するためn_jobs=1
                        return -scores.mean()  # MSEを最小化（負の値を最大化）

                    # Optuna study作成・実行
                    study = optuna.create_study(
                        direction='minimize',
                        sampler=optuna.samplers.TPESampler(seed=self.random_state + fold_idx)
                    )

                    # ログ出力を抑制
                    optuna.logging.set_verbosity(optuna.logging.WARNING)

                    study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

                    # 最良パラメータでモデル作成
                    best_params = study.best_params
                    # モデル実際のパラメータのみを再生成
                    final_params = model_config['param_suggest_func'](study.best_trial)

                    try:
                        best_model = model_config['model_class'](**final_params, random_state=self.random_state)
                    except TypeError:
                        best_model = model_config['model_class'](**final_params)

                    best_model.fit(X_train_scaled, y_train_outer)
                    best_params_per_fold.append(best_params)

                else:
                    # パラメータ最適化が不要なモデルまたはOptuna未インストール
                    try:
                        best_model = model_config['model_class'](random_state=self.random_state)
                    except TypeError:
                        best_model = model_config['model_class']()

                    best_model.fit(X_train_scaled, y_train_outer)
                    best_params_per_fold.append({})

                # テストセットで評価
                y_pred = best_model.predict(X_test_scaled)

                # 評価指標計算
                mse = mean_squared_error(y_test_outer, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_outer, y_pred)
                r2 = r2_score(y_test_outer, y_pred)

                outer_scores['mse'].append(mse)
                outer_scores['rmse'].append(rmse)
                outer_scores['mae'].append(mae)
                outer_scores['r2'].append(r2)

            # 結果の統計量計算
            model_results[model_name] = {
                'mean_mse': np.mean(outer_scores['mse']),
                'std_mse': np.std(outer_scores['mse']),
                'mean_rmse': np.mean(outer_scores['rmse']),
                'std_rmse': np.std(outer_scores['rmse']),
                'mean_mae': np.mean(outer_scores['mae']),
                'std_mae': np.std(outer_scores['mae']),
                'mean_r2': np.mean(outer_scores['r2']),
                'std_r2': np.std(outer_scores['r2']),
                'best_params_per_fold': best_params_per_fold,
                'raw_scores': outer_scores
            }

            elapsed_time = time.time() - start_time

            if self.verbose:
                print(f"  完了 (時間: {elapsed_time:.1f}秒)")
                print(f"  RMSE: {model_results[model_name]['mean_rmse']:.4f} "
                     f"(±{model_results[model_name]['std_rmse']:.4f})")
                print(f"  R²: {model_results[model_name]['mean_r2']:.4f} "
                     f"(±{model_results[model_name]['std_r2']:.4f})")

        self.results_ = model_results

        # 最良モデルの選択（RMSE基準）
        best_model_name = min(model_results.keys(),
                             key=lambda x: model_results[x]['mean_rmse'])
        self.best_model_name_ = best_model_name

        # 最良モデルを全データで再学習
        self._train_best_model(X, y)

        # 全モデルを再学習して保存
        if self.save_models:
            self._train_and_save_all_models(X, y, models_to_evaluate)

        if self.verbose:
            print("-" * 60)
            print(f"最適モデル: {best_model_name}")
            if self.save_models:
                print(f"モデル保存ディレクトリ: {self.save_dir}")
            print("=" * 60)

        return self

    def _train_best_model(self, X: np.ndarray, y: np.ndarray):
        """最良モデルを全データで再学習"""

        # 最良パラメータの決定（最頻値または平均）
        best_params_list = self.results_[self.best_model_name_]['best_params_per_fold']

        if best_params_list and best_params_list[0]:  # パラメータがある場合
            # 各パラメータの最頻値を取得
            best_params = {}
            for param_name in best_params_list[0].keys():
                param_values = [params[param_name] for params in best_params_list]
                # 最頻値を取得
                unique_values, counts = np.unique(param_values, return_counts=True)
                best_params[param_name] = unique_values[np.argmax(counts)]
        else:
            best_params = {}

        # 最良モデルの設定
        model_config = self.models[self.best_model_name_]

        # random_stateを受け取らないモデルへの対応
        try:
            self.best_model_ = model_config['model_class'](**best_params,
                                                         random_state=self.random_state)
        except TypeError:
            # random_stateを受け取らないモデル（LinearRegression等）
            self.best_model_ = model_config['model_class'](**best_params)

        # 全データで学習
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.best_model_.fit(X_scaled, y)
        self.scaler_ = scaler

        # 特徴量重要度の取得（可能な場合）
        self._extract_feature_importance()

        # 最良モデルの保存
        if self.save_models:
            self._save_best_model()

    def _extract_feature_importance(self):
        """特徴量重要度を抽出"""
        try:
            if hasattr(self.best_model_, 'feature_importances_'):
                self.feature_importance_ = self.best_model_.feature_importances_
            elif hasattr(self.best_model_, 'coef_'):
                self.feature_importance_ = np.abs(self.best_model_.coef_)
            else:
                self.feature_importance_ = None
        except:
            self.feature_importance_ = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """最良モデルで予測"""
        if self.best_model_ is None:
            raise ValueError("モデルが学習されていません。先にfit()を実行してください。")

        X_scaled = self.scaler_.transform(X)
        return self.best_model_.predict(X_scaled)

    def get_results_summary(self) -> pd.DataFrame:
        """結果のサマリーをDataFrameで返す"""
        if not self.results_:
            raise ValueError("モデルが学習されていません。先にfit()を実行してください。")

        summary_data = []
        for model_name, results in self.results_.items():
            summary_data.append({
                'Model': model_name,
                'Mean_RMSE': results['mean_rmse'],
                'Std_RMSE': results['std_rmse'],
                'Mean_R2': results['mean_r2'],
                'Std_R2': results['std_r2'],
                'Mean_MAE': results['mean_mae'],
                'Std_MAE': results['std_mae']
            })

        df = pd.DataFrame(summary_data)
        return df.sort_values('Mean_RMSE').reset_index(drop=True)

    def plot_results(self):
        """結果の可視化（matplotlib必要）"""
        try:
            import matplotlib.pyplot as plt

            df = self.get_results_summary()

            _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # RMSE
            ax1.barh(df['Model'], df['Mean_RMSE'], xerr=df['Std_RMSE'])
            ax1.set_xlabel('RMSE')
            ax1.set_title('Root Mean Squared Error')

            # R²
            ax2.barh(df['Model'], df['Mean_R2'], xerr=df['Std_R2'])
            ax2.set_xlabel('R²')
            ax2.set_title('R² Score')

            # MAE
            ax3.barh(df['Model'], df['Mean_MAE'], xerr=df['Std_MAE'])
            ax3.set_xlabel('MAE')
            ax3.set_title('Mean Absolute Error')

            # 特徴量重要度
            if self.feature_importance_ is not None:
                feature_names = [f'Feature_{i}' for i in range(len(self.feature_importance_))]
                ax4.bar(feature_names, self.feature_importance_)
                ax4.set_xlabel('Features')
                ax4.set_ylabel('Importance')
                ax4.set_title(f'Feature Importance ({self.best_model_name_})')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'Feature importance\nnot available',
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Feature Importance')

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("matplotlib がインストールされていないため、グラフを表示できません。")
            print("インストール: uv add matplotlib (または pip install matplotlib)")

    def _train_and_save_all_models(self, X: np.ndarray, y: np.ndarray, models_to_evaluate: dict):
        """全モデルを全データで再学習し、pklファイルとして保存"""

        if self.verbose:
            print("\n全モデルを再学習・保存中...")

        # データの標準化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for model_name, model_config in models_to_evaluate.items():
            # 各モデルの最良パラメータを取得
            best_params_list = self.results_[model_name]['best_params_per_fold']

            if best_params_list and best_params_list[0]:  # パラメータがある場合
                # 最適なパラメータを再生成（内部変数を除外）
                # 最も良いfoldのパラメータを使用
                fold_scores = []
                for i in range(len(best_params_list)):
                    fold_scores.append(self.results_[model_name]['raw_scores']['rmse'][i])
                best_fold_idx = np.argmin(fold_scores)

                # そのfoldのベストパラメータからモデル用パラメータを再構築
                if model_config['param_suggest_func'] is not None:
                    # ダミーのTrialクラスを作成してパラメータを再生成
                    class DummyTrial:
                        def __init__(self, params):
                            self.params = params

                        def suggest_float(self, name, low, high, log=False):
                            return self.params.get(name, (low + high) / 2)

                        def suggest_int(self, name, low, high):
                            return self.params.get(name, int((low + high) / 2))

                        def suggest_categorical(self, name, choices):
                            return self.params.get(name, choices[0])

                    dummy_trial = DummyTrial(best_params_list[best_fold_idx])
                    best_params = model_config['param_suggest_func'](dummy_trial)
                else:
                    best_params = {}
            else:
                best_params = {}

            # モデルを作成
            try:
                model = model_config['model_class'](**best_params, random_state=self.random_state)
            except TypeError:
                model = model_config['model_class'](**best_params)

            # 全データで学習
            model.fit(X_scaled, y)

            # モデルとスケーラーをセットで保存
            model_data = {
                'model': model,
                'scaler': scaler,
                'best_params': best_params,
                'model_name': model_name,
                'performance': {
                    'mean_rmse': self.results_[model_name]['mean_rmse'],
                    'mean_r2': self.results_[model_name]['mean_r2'],
                    'mean_mae': self.results_[model_name]['mean_mae']
                }
            }

            # 訓練済みモデルを保存
            self.trained_models_[model_name] = model_data

            # pklファイルとして保存
            model_path = os.path.join(self.save_dir, f"{model_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)

            if self.verbose:
                print(f"  {model_name} → {model_path}")

    def _save_best_model(self):
        """最良モデルを別途保存"""
        best_model_data = {
            'model': self.best_model_,
            'scaler': self.scaler_,
            'model_name': self.best_model_name_,
            'feature_importance': self.feature_importance_,
            'performance': {
                'mean_rmse': self.results_[self.best_model_name_]['mean_rmse'],
                'mean_r2': self.results_[self.best_model_name_]['mean_r2'],
                'mean_mae': self.results_[self.best_model_name_]['mean_mae']
            }
        }

        best_model_path = os.path.join(self.save_dir, "best_model.pkl")
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model_data, f)

        if self.verbose:
            print(f"最良モデル → {best_model_path}")

    @classmethod
    def load_model(cls, model_path: str):
        """保存されたモデルを読み込み

        Parameters:
        -----------
        model_path : str
            読み込むpklファイルのパス

        Returns:
        --------
        dict : モデルデータ（model, scaler, 性能指標等を含む）
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

    @classmethod
    def predict_with_saved_model(cls, model_path: str, X: np.ndarray) -> np.ndarray:
        """保存されたモデルで予測

        Parameters:
        -----------
        model_path : str
            読み込むpklファイルのパス
        X : array-like
            予測対象の特徴量

        Returns:
        --------
        np.ndarray : 予測結果
        """
        model_data = cls.load_model(model_path)
        X_scaled = model_data['scaler'].transform(X)
        return model_data['model'].predict(X_scaled)

    def print_detailed_results(self):
        """詳細な結果を表示"""
        if not self.results_:
            raise ValueError("モデルが学習されていません。先にfit()を実行してください。")

        print("\n" + "=" * 80)
        print("ダブルクロスバリデーション詳細結果")
        print("=" * 80)

        df = self.get_results_summary()

        for i, row in df.iterrows():
            print(f"\n{i+1}. {row['Model']}")
            print("-" * 40)
            print(f"RMSE:  {row['Mean_RMSE']:.4f} ± {row['Std_RMSE']:.4f}")
            print(f"R²:    {row['Mean_R2']:.4f} ± {row['Std_R2']:.4f}")
            print(f"MAE:   {row['Mean_MAE']:.4f} ± {row['Std_MAE']:.4f}")

            if row['Model'] == self.best_model_name_:
                print("★ 最適モデル ★")

        print(f"\n最終選択モデル: {self.best_model_name_}")
        print(f"期待RMSE: {self.results_[self.best_model_name_]['mean_rmse']:.4f}")
        print(f"期待R²: {self.results_[self.best_model_name_]['mean_r2']:.4f}")


def demo_regression_selection():
    """デモンストレーション関数"""
    from sklearn.datasets import make_regression

    print("ダブルクロスバリデーション回帰モデル選択 デモ")
    print("=" * 60)

    # サンプルデータ生成
    X, y = make_regression(n_samples=200, n_features=10, noise=0.1,
                          random_state=42)

    print(f"生成データ: X={X.shape}, y={y.shape}")

    # モデル選択器の作成と実行
    selector = DoubleCV_RegressionSelector(
        outer_cv=3,  # デモのため少なく設定
        inner_cv=2,
        n_trials=20,  # デモのため少なく設定
        verbose=True
    )

    # 一部のモデルのみで実行（デモのため）
    model_subset = ['LinearRegression', 'Ridge', 'RandomForest', 'SVR', 'GaussianProcess']

    selector.fit(X, y, model_subset=model_subset)

    # 結果表示
    selector.print_detailed_results()

    # 予測例
    y_pred = selector.predict(X[:5])
    print(f"\n予測例（最初の5サンプル）:")
    print(f"真値: {y[:5]}")
    print(f"予測: {y_pred}")


if __name__ == "__main__":
    demo_regression_selection()