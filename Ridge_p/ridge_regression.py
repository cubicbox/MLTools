import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats


class RidgeRegression_p:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.X_train_ = None
        self.y_train_ = None
        self.pvalues_ = None
        self.se_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X_train_ = X.copy()
        self.y_train_ = y.copy()

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

        I = np.eye(X_with_intercept.shape[1])
        I[0, 0] = 0

        theta = np.linalg.inv(X_with_intercept.T @ X_with_intercept + self.alpha * I) @ X_with_intercept.T @ y

        self.intercept_ = theta[0]
        self.coef_ = theta[1:]

        self._calculate_pvalues(X_with_intercept, y)

        return self

    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X @ self.coef_ + self.intercept_

    def _calculate_pvalues(self, X_with_intercept, y):
        y_pred = X_with_intercept @ np.concatenate([[self.intercept_], self.coef_])
        residuals = y - y_pred
        n = X_with_intercept.shape[0]
        p = X_with_intercept.shape[1]

        mse = np.sum(residuals**2) / (n - p)

        XTX_reg = X_with_intercept.T @ X_with_intercept
        I = np.eye(XTX_reg.shape[0])
        I[0, 0] = 0

        try:
            cov_matrix = mse * np.linalg.inv(XTX_reg + self.alpha * I)
            se_all = np.sqrt(np.diag(cov_matrix))

            theta_all = np.concatenate([[self.intercept_], self.coef_])
            t_stats = theta_all / se_all

            self.pvalues_ = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-p))
            self.se_ = se_all[1:]

        except np.linalg.LinAlgError:
            self.pvalues_ = np.full(len(self.coef_), np.nan)
            self.se_ = np.full(len(self.coef_), np.nan)

    def get_pvalues(self):
        if self.pvalues_ is None:
            raise ValueError("Model must be fitted before calculating p-values")
        return self.pvalues_[1:]

    def get_feature_summary(self):
        if self.coef_ is None:
            raise ValueError("Model must be fitted first")

        return {
            'coefficients': self.coef_,
            'pvalues': self.get_pvalues(),
            'standard_errors': self.se_
        }


def ridge_regression_demo():
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    plt.figure(figsize=(15, 10))

    for i, alpha in enumerate(alphas):
        ridge = RidgeRegression_p(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)

        y_pred = ridge.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        plt.subplot(2, 3, i+1)

        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_plot_scaled = scaler.transform(X_plot)
        y_plot = ridge.predict(X_plot_scaled)

        plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
        plt.plot(X_plot, y_plot, color='red', linewidth=2, label=f'Ridge (α={alpha})')
        plt.title(f'Ridge Regression (α={alpha})\nMSE: {mse:.2f}, R²: {r2:.3f}')
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("Ridge Regression Results:")
    print("-" * 40)
    for alpha in alphas:
        ridge = RidgeRegression_p(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred = ridge.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pval = ridge.get_pvalues()[0]
        print(f"Alpha: {alpha:6.2f} | MSE: {mse:8.2f} | R²: {r2:6.3f} | Coef: {ridge.coef_[0]:8.3f} | p-value: {pval:.3f}")


def compare_with_linear_regression():
    from sklearn.linear_model import LinearRegression

    X, y = make_regression(n_samples=50, n_features=5, noise=30, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    linear_reg = LinearRegression()
    ridge_reg = RidgeRegression_p(alpha=1.0)

    linear_reg.fit(X_train, y_train)
    ridge_reg.fit(X_train, y_train)

    linear_pred = linear_reg.predict(X_test)
    ridge_pred = ridge_reg.predict(X_test)

    linear_mse = mean_squared_error(y_test, linear_pred)
    ridge_mse = mean_squared_error(y_test, ridge_pred)

    linear_r2 = r2_score(y_test, linear_pred)
    ridge_r2 = r2_score(y_test, ridge_pred)

    print("\nComparison: Linear vs Ridge Regression")
    print("=" * 45)
    print(f"Linear Regression - MSE: {linear_mse:.2f}, R²: {linear_r2:.3f}")
    print(f"Ridge Regression  - MSE: {ridge_mse:.2f}, R²: {ridge_r2:.3f}")

    print("\nCoefficients comparison:")
    print("Linear Regression coefficients:")
    for i, coef in enumerate(linear_reg.coef_):
        print(f"  Feature {i+1}: {coef:.3f}")

    print("Ridge Regression coefficients:")
    ridge_pvalues = ridge_reg.get_pvalues()
    for i, (coef, pval) in enumerate(zip(ridge_reg.coef_, ridge_pvalues)):
        print(f"  Feature {i+1}: {coef:.3f} (p-value: {pval:.3f})")


if __name__ == "__main__":
    print("Ridge Regression Demonstration")
    print("=" * 50)

    ridge_regression_demo()
    compare_with_linear_regression()