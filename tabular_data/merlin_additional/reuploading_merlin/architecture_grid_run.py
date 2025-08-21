"""
architecture_grid_run.py – minimal benchmark of 9 re‑uploading designs
=====================================================================
Train **MerlinReuploadingClassifier** for every (data‑block, var‑block) pair
in {A,B,C}² and plot a 3×3 grid per circuit depth ℓ.

Goals
-----
1. **Keep code tiny & readable** – one file, no fancy abstractions.
2. **Fresh figure per depth** – avoid artefacts; memory‑safe via `plt.close()`.
3. **Self‑contained** – DATASET + HYPERPARAMS declared at top, ready to tweak.

Usage
-----
```bash
python architecture_grid_run.py           # saves architecture_grid_1.png … _9.png
```
Set `DEPTHS = [1,2,3]` for a subset, etc.
"""

from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from paper_datasets import CirclesDataset          # swap for MoonsDataset, …
from reuploading_experiment import MerlinReuploadingClassifier

# ── global knobs ───────────────────────────────────────────────────────────────

DEPTHS = range(1, 10)               # ℓ = 1 … 9  (set to [3] for a quick test)
DATASET = CirclesDataset(n_train=400, n_test=100, noise=0.05)
HYPERPARAMS = dict(
    max_epochs=10_000,
    learning_rate=1e-3,
    batch_size=400,
    patience=1_000,
    tau=1.0,
    track_history=False,
)

# ── little helpers ─────────────────────────────────────────────────────────────

def _classification_map(ax, model, X_tr, y_tr, X_te, y_te):
    res = 220
    all_pts = np.vstack([X_tr, X_te]) if X_te is not None else X_tr
    margin = 0.15 * (all_pts.max(0) - all_pts.min(0))
    xlim = (all_pts[:, 0].min()-margin[0], all_pts[:, 0].max()+margin[0])
    ylim = (all_pts[:, 1].min()-margin[1], all_pts[:, 1].max()+margin[1])

    xx, yy = np.meshgrid(np.linspace(*xlim, res), np.linspace(*ylim, res))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    prob = model.predict_proba(grid)
    diff = (prob[:, 1] - prob[:, 0]).reshape(xx.shape)

    cmap = LinearSegmentedColormap.from_list('rwb', ['#d62728', '#ffffff', '#1f77b4'])
    ax.contourf(xx, yy, diff, levels=np.linspace(-1, 1, 21), cmap=cmap, alpha=.75, extend='both')
    ax.contour(xx, yy, diff, levels=[0], colors='k', linewidths=2)

    ax.scatter(*X_tr[y_tr==0].T, c='darkred',  s=30, lw=.6, ec='w')
    ax.scatter(*X_tr[y_tr==1].T, c='darkblue', s=30, lw=.6, ec='w')
    if X_te is not None:
        ax.scatter(*X_te[y_te==0].T, c='darkred',  s=40, marker='s', lw=1, ec='w')
        ax.scatter(*X_te[y_te==1].T, c='darkblue', s=40, marker='s', lw=1, ec='w')

    ax.set(xlim=xlim, ylim=ylim, xticks=[], yticks=[])


def _probability_axis(ax, model, X_tr, y_tr, X_te, y_te):
    feat_tr = model.get_quantum_features(X_tr)[:, 0]
    off = .12
    ax.scatter(feat_tr[y_tr==0], -off*np.ones_like(feat_tr[y_tr==0]), c='red',  s=22, alpha=.85)
    ax.scatter(feat_tr[y_tr==1],  off*np.ones_like(feat_tr[y_tr==1]), c='blue', s=22, alpha=.85)
    if X_te is not None:
        feat_te = model.get_quantum_features(X_te)[:, 0]
        ax.scatter(feat_te[y_te==0], -2*off*np.ones_like(feat_te[y_te==0]), c='red',  s=28, marker='s', alpha=.85)
        ax.scatter(feat_te[y_te==1],  2*off*np.ones_like(feat_te[y_te==1]), c='blue', s=28, marker='s', alpha=.85)
    ax.set_xlim(0, 1);
    ax.axis('off')


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    X_tr, y_tr = DATASET.train
    X_te, y_te = DATASET.test

    designs = [a+b for a, b in itertools.product('ABC', repeat=2)]

    for L in DEPTHS:
        # fresh figure per depth → guaranteed reset
        fig, axes = plt.subplots(6, 3, figsize=(12, 18), dpi=120,
                                 gridspec_kw={'height_ratios': [4,1]*3,
                                              'hspace': .25, 'wspace': .15})

        for idx, design in enumerate(designs):
            row, col = divmod(idx, 3)
            mdl = MerlinReuploadingClassifier(dimension=X_tr.shape[1], num_layers=L, design=design)
            mdl.fit(X_tr, y_tr, **HYPERPARAMS)

            ax_map  = axes[2*row,   col]
            ax_prob = axes[2*row+1, col]
            _classification_map(ax_map,  mdl, X_tr, y_tr, X_te, y_te)
            _probability_axis(ax_prob, mdl, X_tr, y_tr, X_te, y_te)

            if col==0:
                ax_map.set_ylabel(f'Data {design[0]}', rotation=0, labelpad=40, fontsize=12, va='center')
        for col,label in enumerate('ABC'):
            axes[0, col].set_title(f'Train {label}', fontsize=13, pad=18)

        fig.suptitle(f'Re‑uploading circuits – {L} layer(s)', fontsize=16, y=.92)
        plt.tight_layout(rect=[0,0,1,.95])
        out = f'architecture_grid_{L}.png'
        fig.savefig(out, dpi=300)
        plt.close(fig)             # <‑‑ free memory, reset canvas
        print(f'✓ saved {out}')
