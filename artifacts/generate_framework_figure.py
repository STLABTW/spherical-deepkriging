"""
Generate a combined framework figure:
  Left  — spherical MRTS basis functions φ₂–φ₇ on Mollweide globe maps
  Right — hidden_block.png (MLP block diagram)
  Theme — spatial prediction on S²
"""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.special import spence

# ── helpers ────────────────────────────────────────────────────────────────────


def li2(y):
    """Dilogarithm Li_2(y) = spence(1 - y)."""
    y = np.asarray(y, dtype=float)
    return spence(1.0 - np.clip(y, 1e-15, 1.0 - 1e-15))


def spherical_tps_kernel(cos_gamma):
    """Φ(s, s*) = Li_2(0.5 + cos(γ)/2) + 1 - π/6  (vectorised)."""
    return li2(0.5 + cos_gamma / 2.0) + 1.0 - np.pi / 6.0


# ── 1. knots uniformly on S² ───────────────────────────────────────────────────

np.random.seed(42)
m = 150
u_k = np.random.uniform(-1, 1, m)
phi_k = np.random.uniform(0, 2 * np.pi, m)
lat_k = np.arcsin(u_k)
lon_k = phi_k - np.pi

knot_cart = np.stack(
    [
        np.cos(lat_k) * np.cos(lon_k),
        np.cos(lat_k) * np.sin(lon_k),
        np.sin(lat_k),
    ],
    axis=1,
)  # (m, 3)

# ── 2. kernel matrix and Q K Q ─────────────────────────────────────────────────

cos_K = np.clip(knot_cart @ knot_cart.T, -1, 1)
K = spherical_tps_kernel(cos_K)

row_mean = K.mean(axis=1, keepdims=True)
col_mean = K.mean(axis=0, keepdims=True)
grand_mean = K.mean()
QKQ = K - row_mean - col_mean + grand_mean

# ── 3. eigendecomposition (descending) ─────────────────────────────────────────

eigvals, eigvecs = eigh(QKQ)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# ── 4. evaluate φ₂ … φ₇ on lat/lon grid ───────────────────────────────────────

n_lon, n_lat = 360, 180
lons_g = np.linspace(-np.pi, np.pi, n_lon)
lats_g = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
lon_g, lat_g = np.meshgrid(lons_g, lats_g)

grid_cart = np.stack(
    [
        np.cos(lat_g.ravel()) * np.cos(lon_g.ravel()),
        np.cos(lat_g.ravel()) * np.sin(lon_g.ravel()),
        np.sin(lat_g.ravel()),
    ],
    axis=1,
)  # (N, 3)

cos_ks = np.clip(grid_cart @ knot_cart.T, -1, 1)
k_vec = spherical_tps_kernel(cos_ks)  # (N, m)
Konev = K.mean(axis=1)  # (m,)
diff = k_vec - Konev[None, :]  # (N, m)

# compute φ₁ … φ₆  (φ₁ = constant, captures global mean / linearity)
num_show = 6
basis_maps = np.zeros((n_lat, n_lon, num_show))
basis_maps[:, :, 0] = 1.0 / np.sqrt(m)  # φ₁: constant basis

for i in range(1, num_show):
    lam = eigvals[i - 1]
    if abs(lam) < 1e-12:
        continue
    vals = (diff @ eigvecs[:, i - 1]) / lam
    basis_maps[:, :, i] = vals.reshape(n_lat, n_lon)

# ── 5. build figure  (zen style) ───────────────────────────────────────────────

ARTIFACTS = Path(__file__).parent
hidden_img = mpimg.imread(ARTIFACTS / "hidden_block.png")

# muted diverging palette
from matplotlib.colors import LinearSegmentedColormap

zen_colors = ["#4a7fb5", "#f7f4f0", "#c0504d"]  # steel-blue · off-white · muted-red
zen_cmap = LinearSegmentedColormap.from_list("zen", zen_colors, N=256)

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.color": "#2e2e2e",
        "axes.titlesize": 11,
    }
)

fig = plt.figure(figsize=(14, 6.0))
fig.patch.set_facecolor("#fafaf8")  # warm off-white

outer = gridspec.GridSpec(
    1,
    2,
    figure=fig,
    width_ratios=[1.6, 1],
    wspace=0.0,
    left=0.02,
    right=0.97,
    top=0.86,
    bottom=0.12,
)

# ── left: 2×3 Mollweide maps ───────────────────────────────────────────────────
inner = gridspec.GridSpecFromSubplotSpec(
    2,
    3,
    subplot_spec=outer[0],
    hspace=0.28,
    wspace=0.04,
)

labels = [rf"$\phi_{i+1}$" for i in range(num_show)]

for i in range(num_show):
    ax = fig.add_subplot(inner[i // 3, i % 3], projection="mollweide")
    ax.set_facecolor("#fafaf8")
    bmap = basis_maps[:, :, i]
    vmax = np.percentile(np.abs(bmap), 99) or 1.0
    ax.pcolormesh(
        lons_g,
        lats_g,
        bmap,
        cmap=zen_cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )
    ax.set_title(labels[i], fontsize=12, pad=4, color="#2e2e2e")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

# label below basis panel
fig.text(
    0.275,
    0.04,
    r"Spherical MRTS Basis Features  $\boldsymbol{\phi}(\mathbf{s})$",
    ha="center",
    fontsize=11,
    style="italic",
    color="#555555",
)

# ── right: MLP hidden block ────────────────────────────────────────────────────
ax_nn = fig.add_subplot(outer[1])
ax_nn.set_facecolor("#fafaf8")
ax_nn.imshow(hidden_img)
ax_nn.axis("off")

fig.text(
    0.775,
    0.04,
    r"Spatial Prediction  $\hat{y}(\mathbf{s}_0)$",
    ha="center",
    fontsize=11,
    style="italic",
    color="#555555",
)

# ── arrow + input label ────────────────────────────────────────────────────────
fig.add_artist(
    mpatches.FancyArrowPatch(
        (0.538, 0.47),
        (0.576, 0.47),
        transform=fig.transFigure,
        arrowstyle="-|>",
        color="#888888",
        lw=1.4,
        mutation_scale=14,
    )
)
fig.text(
    0.557,
    0.525,
    r"$\mathbf{u}(\mathbf{s}) = (\boldsymbol{\phi}(\mathbf{s})^\top,\ \mathbf{x}(\mathbf{s})^\top)^\top$",
    ha="center",
    fontsize=10,
    color="#2e2e2e",
)

# thin vertical divider
fig.add_artist(
    plt.matplotlib.lines.Line2D(
        [0.557, 0.557],
        [0.12, 0.88],
        transform=fig.transFigure,
        color="#cccccc",
        lw=0.8,
        linestyle="--",
    )
)

out_path = ARTIFACTS / "framework_overview.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out_path}")
