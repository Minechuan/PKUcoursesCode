# Python code to visualize a 21x21 policy table and save an image.
# The function `visualize_policy` accepts a 21x21 numpy array (policy) of integers
# and draws boundary lines between adjacent cells where the policy differs.
# It saves the image to the specified path and also returns the matplotlib figure.

import json, numpy as np
import os
import matplotlib.pyplot as plt
import math

from matplotlib.colors import ListedColormap, BoundaryNorm

def visualize_policies_grid(all_policy,
                            outpath="./visualize/policies_grid.png",
                            ncols=2,
                            figsize_per_subplot=(6,6),
                            cmap_name="tab20",
                            alpha=0.6,
                            draw_boundaries=True,
                            show_axis=True,
                            show_cell_labels=False,
                            dpi=200):
    """
    all_policy: array-like, shape (n_iters, 21, 21), dtype int
    每个 policy[i] 是 21x21 的整数 action 网格（action 可以为负数）
    每行 ncols 个子图（默认 2），总行数自动计算。
    """
    all_policy = np.asarray(all_policy, dtype=int)
    assert all_policy.ndim == 3 and all_policy.shape[1:] == (21,21)

    n = all_policy.shape[0]
    nrows = math.ceil(n / ncols)
    fig_w = ncols * figsize_per_subplot[0]
    fig_h = nrows * figsize_per_subplot[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    plt.subplots_adjust(wspace=0.6, hspace=0.2)  # 默认大约 0.2，值越大间距越大
    axes = np.array(axes).reshape(-1)  # flatten to 1D list for easy indexing

    # determine global action range so color mapping is consistent
    min_action = int(all_policy.min())
    max_action = int(all_policy.max())
    actions = np.arange(min_action, max_action + 1)
    n_actions = len(actions)

    # build discrete colormap with n_actions colors
    base_cmap = plt.get_cmap(cmap_name)
    colors = base_cmap(np.linspace(0, 1, n_actions))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(min_action - 0.5, max_action + 1.5), n_actions)

    for idx in range(nrows * ncols):
        ax = axes[idx]
        if idx < n:
            policy = all_policy[idx]

            # show discrete heatmap of actions (semi-transparent)
            im = ax.imshow(policy,
                           origin='lower',
                           extent=[0, 21, 0, 21],
                           interpolation='nearest',
                           cmap=cmap,
                           norm=norm,
                           alpha=alpha)

            # optional: draw dashed boundaries where adjacent cells have different actions
            if draw_boundaries:
                # vertical boundaries between columns j and j+1
                for i in range(21):
                    for j in range(20):
                        if policy[i, j] != policy[i, j+1]:
                            x = j + 1.0
                            ax.plot([x, x], [i, i+1], linewidth=1.0, color="black", linestyle="--")
                # horizontal boundaries between rows i and i+1
                for i in range(20):
                    for j in range(21):
                        if policy[i, j] != policy[i+1, j]:
                            y = i + 1.0
                            ax.plot([j, j+1], [y, y], linewidth=1.0, color="black", linestyle="--")

            # optional: draw action number in each cell (may be cluttered)
            if show_cell_labels:
                for i in range(21):
                    for j in range(21):
                        ax.text(j + 0.5, i + 0.5, str(int(policy[i, j])),
                                ha='center', va='center', fontsize=8, color='black')

            # axis labels / ticks
            if show_axis:
                ax.set_xticks(range(0,21,5))
                ax.set_yticks(range(0,21,5))
                ax.set_xlabel("#Cars at second location")
                ax.set_ylabel("#Cars at first location")
            else:
                ax.axis('off')

            ax.set_title(f"Policy iter {idx}")
        else:
            ax.axis('off')  # empty subplot

    # overall colorbar (map colors -> actions)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # adjust as needed
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, ticks=actions)
    cb.set_label('action')

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])  # leave space on right for colorbar
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath

def visualize_value_function(V,
                             outpath="./visualize/value_function.png",
                             figsize=(6,6),
                             cmap_name="viridis",
                             dpi=200):
    """
    V: array-like, shape (21, 21), dtype float
    """
    V = np.asarray(V, dtype=float)
    assert V.shape == (21, 21)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(V,
                   origin='lower',
                   extent=[0, 21, 0, 21],
                   interpolation='nearest',
                   cmap=cmap_name)

    ax.set_xticks(range(0,21,5))
    ax.set_yticks(range(0,21,5))
    ax.set_xlabel("#Cars at second location")
    ax.set_ylabel("#Cars at first location")
    ax.set_title("Value Function")

    # colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # adjust as needed
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Value')

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])  # leave space on right for colorbar
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    return outpath


with open("saved_policy.json") as f:
    data = json.load(f)

policy_num_iter = data["iteration"]
all_policy = np.array(data["policy"], dtype=int)
with open("saved_value.json") as f:
    data = json.load(f)
V = np.array(data["value"], dtype=float)
# Create sample and visualize
print(f"Visualizing policy for {policy_num_iter} iterations...")
out = visualize_policies_grid(all_policy, outpath="./visualize/all_policies_grid.png",
                              ncols=2, alpha=0.55, draw_boundaries=True, show_cell_labels=False)

print("Visualize value function...")
out = visualize_value_function(V, outpath="./visualize/value_function.png")
print("Saved to", out)
