import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import (
    LinearRegression,
    HuberRegressor,
    RANSACRegressor,
    TheilSenRegressor
)


# -------------------------------------------------
# Question 1: Dataset generation and visualization
# -------------------------------------------------

def generate_clean_data(
    n_samples=500,
    noise=20,
    random_state=42
):
    """
    Generate a clean synthetic regression dataset.

    Return:
        X, y, true_coef

    Requirements:
    - n_samples = 500 by default
    - n_features = 1
    - n_informative = 1
    - noise = 20
    - coef = True
    - random_state = 42
    """
    X, y, true_coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=noise,
        coef=True,
        random_state=random_state
    )

    return X, y, true_coef


def add_outliers(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Add artificial outliers to the first n_outliers observations.

    Use:
        X[:n_outliers] = 10 + 0.75 * random_normal_values
        y[:n_outliers] = -15 + 20 * random_normal_values

    Return:
        X_out, y_out

    Important:
    Do not modify the original X and y directly.
    Make copies first.
    """
    X_out = X.copy()
    y_out = y.copy()

    rng = np.random.RandomState(random_state)

    X_out[:n_outliers] = (
        10 + 0.75 * rng.normal(size=(n_outliers, X.shape[1]))
    )

    y_out[:n_outliers] = (
        -15 + 20 * rng.normal(size=n_outliers)
    )

    return X_out, y_out


def plot_dataset_with_outliers(
    X,
    y,
    n_outliers=25
):
    """
    Plot the dataset and highlight the first n_outliers observations.

    Return:
        matplotlib Figure object

    Requirements:
    - normal observations and artificial outliers should be visually different
    - include title
    - include x-label
    - include y-label
    - include legend
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        X[n_outliers:],
        y[n_outliers:],
        label="Normal observations",
        alpha=0.7
    )

    ax.scatter(
        X[:n_outliers],
        y[:n_outliers],
        label="Artificial outliers",
        marker="x",
        s=60
    )

    ax.set_title("Dataset with Artificial Outliers")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


# -------------------------------------------------
# Question 2: Fit regression models
# -------------------------------------------------

def fit_linear_regression(X, y):
    """
    Fit ordinary Linear Regression.

    Return:
        fitted coefficient as a float
    """
    model = LinearRegression()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_huber_regression(X, y):
    """
    Fit Huber Regression.

    Return:
        fitted coefficient as a float
    """
    model = HuberRegressor()
    model.fit(X, y)

    return float(model.coef_[0])


def fit_ransac_regression(X, y, random_state=42):
    """
    Fit RANSAC Regression.

    Return:
        fitted coefficient as a float

    Hint:
    RANSAC stores the final linear model in estimator_.
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.estimator_.coef_[0])


def fit_theilsen_regression(X, y, random_state=42):
    """
    Fit Theil-Sen Regression.

    Return:
        fitted coefficient as a float
    """
    model = TheilSenRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    return float(model.coef_[0])


def coefficient_errors(coef_dict, true_coef):
    """
    Given a dictionary of coefficients and the true coefficient,
    return a dictionary of absolute coefficient errors.

    Example input:
        {
            "linear_regression": 8.7,
            "huber_regression": 37.5,
            "ransac_regression": 62.8,
            "theilsen_regression": 59.4
        }

    Return:
        {
            "linear_regression": abs(...),
            ...
        }
    """
    return {
        model_name: abs(coef - true_coef)
        for model_name, coef in coef_dict.items()
    }


def best_robust_model(errors):
    """
    Return the name of the robust model with the smallest error.

    Only compare:
        huber_regression
        ransac_regression
        theilsen_regression

    Do not include ordinary linear_regression in this comparison.
    """
    robust_errors = {
        "huber_regression": errors["huber_regression"],
        "ransac_regression": errors["ransac_regression"],
        "theilsen_regression": errors["theilsen_regression"]
    }

    return min(robust_errors, key=robust_errors.get)


def ransac_outlier_summary(
    X,
    y,
    n_outliers=25,
    random_state=42
):
    """
    Fit RANSAC and return:

        total_outliers_detected, added_outliers_detected

    total_outliers_detected:
        total number of samples classified as outliers by RANSAC

    added_outliers_detected:
        number of artificial outliers among the first n_outliers
        that RANSAC classified as outliers
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    total_outliers_detected = int(np.sum(outlier_mask))
    added_outliers_detected = int(
        np.sum(outlier_mask[:n_outliers])
    )

    return (
        total_outliers_detected,
        added_outliers_detected
    )


# -------------------------------------------------
# Question 2: Visualization functions
# -------------------------------------------------

def plot_regression_fits(
    X,
    y,
    random_state=42
):
    """
    Plot fitted regression lines for:
    - Linear Regression
    - Huber Regression
    - RANSAC Regression
    - Theil-Sen Regression

    Return:
        matplotlib Figure object

    Requirements:
    - scatter plot of data
    - fitted line for each model
    - title
    - x-label
    - y-label
    - legend
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        X,
        y,
        alpha=0.6,
        label="Data"
    )

    line_X = np.linspace(
        X.min(),
        X.max(),
        500
    ).reshape(-1, 1)

    models = {
        "Linear Regression":
            LinearRegression(),

        "Huber Regression":
            HuberRegressor(),

        "RANSAC Regression":
            RANSACRegressor(
                random_state=random_state
            ),

        "Theil-Sen Regression":
            TheilSenRegressor(
                random_state=random_state
            )
    }

    for name, model in models.items():
        model.fit(X, y)

        predictions = model.predict(line_X)

        ax.plot(
            line_X,
            predictions,
            label=name
        )

    ax.set_title("Regression Model Comparison")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig


def plot_ransac_inliers_outliers(
    X,
    y,
    random_state=42
):
    """
    Fit RANSAC and visualize inliers vs outliers.

    Return:
        matplotlib Figure object

    Requirements:
    - inliers and outliers should be visually different
    - title
    - x-label
    - y-label
    - legend
    """
    model = RANSACRegressor(
        random_state=random_state
    )

    model.fit(X, y)

    inlier_mask = model.inlier_mask_
    outlier_mask = ~inlier_mask

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        X[inlier_mask],
        y[inlier_mask],
        label="Inliers",
        alpha=0.7
    )

    ax.scatter(
        X[outlier_mask],
        y[outlier_mask],
        label="Outliers",
        marker="x",
        s=60
    )

    ax.set_title("RANSAC Inlier vs Outlier Detection")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.legend()

    return fig
