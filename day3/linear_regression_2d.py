"""
第3回：1 次元入力 x・1 次元出力 y の線形回帰（最小二乗）用データ生成・可視化。

「2D」は (x, y) 平面上の 2 次元プロットの意味。
行列形では X: (m, 1), y: (m,), w: (1,) で y ≈ X @ w（切片なし）。
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = [
    "apply_plot_style",
    "make_regression_2d_data",
    "plot_linear_regression_2d",
]


def apply_plot_style() -> None:
    """論文・スライド向けに、軸まわりの文字を大きめにそろえる。"""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titlecolor": "#111111",
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "font.size": 15,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        }
    )


def _as_design_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[1] != 1:
        raise ValueError("X は shape (m,) または (m, 1) である必要があります（1 次元入力）")
    return X


def _as_slope(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float).reshape(-1)
    if w.size != 1:
        raise ValueError("w はスカラまたは shape (1,) である必要があります")
    return float(w[0])


def make_regression_2d_data(
    n_samples: int = 50,
    w_true: float | np.ndarray | None = None,
    noise_std: float = 0.35,
    x_range: tuple[float, float] = (-2.2, 2.2),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    スカラ入力量 x_i に対し y_i = w_true * x_i + ε_i を生成する。

    Parameters
    ----------
    n_samples : int
        サンプル数（既定 50）
    w_true : float / (1,) ndarray / None
        真の斜め。None なら乱数で決める。
    noise_std : float
        ガウスノイズの標準偏差
    x_range : (low, high)
        x の一様分布の範囲
    rng : Generator または None
        再現用乱数。None なら非固定。

    Returns
    -------
    X : ndarray, shape (n_samples, 1)
    y : ndarray, shape (n_samples,)
    w_true : ndarray, shape (1,)
    """
    if rng is None:
        rng = np.random.default_rng()
    if w_true is None:
        w_scalar = float(rng.normal(0.8, 0.9))
    else:
        w_scalar = _as_slope(np.asarray(w_true, dtype=float))

    w_arr = np.array([w_scalar], dtype=float)
    low, high = x_range
    X = rng.uniform(low, high, size=(n_samples, 1))
    eps = rng.normal(0.0, noise_std, size=n_samples)
    y = (X @ w_arr).ravel() + eps
    return X, y, w_arr


def plot_linear_regression_2d(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    w_true: np.ndarray | None = None,
    title: str | None = None,
    point_color: str = "#1b6ca8",
    line_color: str = "#f37121",
    true_line_color: str = "#0f4c75",
    figsize: tuple[float, float] = (9.0, 6.5),
) -> plt.Figure:
    """
    (x, y) 平面上に散布図と回帰直線 y = w x を描く（切片なし）。

    Parameters
    ----------
    X : (m,) または (m, 1)
    y : (m,)
    w : 推定斜め（スカラ相当）
    w_true : (1,) または None
        指定時は破線で真の直線を重ねる。
    """
    apply_plot_style()

    Xd = _as_design_matrix(X)
    yv = np.asarray(y, dtype=float).ravel()
    w_hat = _as_slope(w)

    if Xd.shape[0] != yv.shape[0]:
        raise ValueError("X の行数と y の長さが一致しません")

    xv = Xd[:, 0]
    x_min, x_max = xv.min(), xv.max()
    span = max(x_max - x_min, 1e-6)
    pad = 0.08 * span
    xs = np.linspace(x_min - pad, x_max + pad, 200)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.scatter(
        xv,
        yv,
        c=point_color,
        s=58,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.85,
        zorder=3,
    )
    ax.plot(xs, w_hat * xs, color=line_color, linewidth=2.8, zorder=2)

    legend_handles = [
        Line2D(
            [0],
            [0],
            linestyle="",
            marker="o",
            markersize=11,
            markerfacecolor=point_color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=r"データ $(x_i, y_i)$",
        ),
        Line2D([0], [0], color=line_color, lw=2.8, label=r"回帰直線 ($\hat w$)"),
    ]

    if w_true is not None:
        wt = _as_slope(w_true)
        ax.plot(
            xs,
            wt * xs,
            color=true_line_color,
            linewidth=2.2,
            linestyle=(0, (5, 4)),
            zorder=1,
            alpha=0.9,
            label=r"真の直線 ($w_{\mathrm{true}}$)",
        )
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=true_line_color,
                lw=2.2,
                linestyle=(0, (5, 4)),
                label=r"真の直線 ($w_{\mathrm{true}}$)",
            )
        )

    ax.set_xlabel(r"$x$", fontsize=20, labelpad=10)
    ax.set_ylabel(r"$y$", fontsize=20, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=16, pad=4)

    if title:
        ax.set_title(title, fontsize=22, pad=14)

    ax.legend(
        handles=legend_handles,
        loc="best",
        framealpha=0.94,
        fancybox=True,
        edgecolor="#cccccc",
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, y, w_true = make_regression_2d_data(n_samples=50, rng=rng)
    w_demo, *_ = np.linalg.lstsq(X, y, rcond=None)

    fig = plot_linear_regression_2d(
        X,
        y,
        w_demo,
        w_true=w_true,
        title=r"1 入力の線形回帰：$(x,y)$ 平面上の散布図と直線 $y \approx w x$",
    )
    plt.show()
