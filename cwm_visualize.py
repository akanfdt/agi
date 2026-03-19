from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from cwm_core import CWMCore


DEFAULT_MODEL_CANDIDATES = [
    "cwm_model.pt",
    "tmp_fresh_longrun.pt",
    "tmp_cwm_eval_model.pt",
    "tmp_repro_after_content_only.pt",
    "tmp_repro_3k.pt",
]


def resolve_model_path(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"model not found: {path}")
        return path
    for name in DEFAULT_MODEL_CANDIDATES:
        path = Path(name)
        if path.exists():
            return path
    raise FileNotFoundError("no CWM model file found")


def safe_token_label(token: str) -> str:
    if token == "▁":
        return "<WORD_START>"
    return token.replace("▁", "")


def pick_display_tokens(core: CWMCore, include_internal: bool) -> list[str]:
    tokens = []
    for token in core.anchors.keys():
        if include_internal or core.is_emittable_token(token):
            tokens.append(token)
    return tokens


def top_gravity_edges(core: CWMCore, allowed: set[str], limit: int) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for src, mapping in core.gravity.forward_gravity.items():
        if src not in allowed:
            continue
        for dst, value in mapping.items():
            if dst not in allowed or value <= 0.0:
                continue
            rows.append((src, dst, float(value)))
    rows.sort(key=lambda item: item[2], reverse=True)
    return rows[:limit]


def build_plot(
    core: CWMCore,
    output_path: Path,
    max_points: int,
    label_top_n: int,
    edge_top_n: int,
    include_internal: bool,
    random_seed: int,
) -> None:
    display_tokens = pick_display_tokens(core, include_internal=include_internal)
    if not display_tokens:
        raise RuntimeError("no tokens available to visualize")

    display_tokens.sort(key=lambda tok: core.anchors[tok].importance, reverse=True)
    display_tokens = display_tokens[:max_points]

    vectors = np.stack([core.anchors[tok].vec.detach().cpu().numpy() for tok in display_tokens], axis=0)
    importance = np.array([float(core.anchors[tok].importance) for tok in display_tokens], dtype=np.float32)

    pca = PCA(n_components=2, random_state=random_seed)
    coords = pca.fit_transform(vectors)

    n_clusters = max(2, min(12, len(display_tokens) // 20))
    if len(display_tokens) < n_clusters:
        n_clusters = max(1, len(display_tokens))
    if n_clusters > 1:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_ids = kmeans.fit_predict(vectors)
    else:
        cluster_ids = np.zeros(len(display_tokens), dtype=np.int32)

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(14, 10))

    size_scale = np.clip(importance, a_min=np.percentile(importance, 10), a_max=np.percentile(importance, 95))
    size_scale = 18.0 + 120.0 * (size_scale - size_scale.min()) / max(1e-8, size_scale.max() - size_scale.min())

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=cluster_ids,
        s=size_scale,
        alpha=0.72,
        cmap="tab20",
        linewidths=0.35,
        edgecolors="white",
    )

    token_to_index = {tok: i for i, tok in enumerate(display_tokens)}
    top_tokens = display_tokens[:label_top_n]
    for tok in top_tokens:
        idx = token_to_index[tok]
        ax.text(
            coords[idx, 0],
            coords[idx, 1],
            safe_token_label(tok),
            fontsize=9,
            alpha=0.9,
        )

    edge_tokens = set(display_tokens[: max(label_top_n * 3, 80)])
    for src, dst, weight in top_gravity_edges(core, edge_tokens, edge_top_n):
        src_idx = token_to_index.get(src)
        dst_idx = token_to_index.get(dst)
        if src_idx is None or dst_idx is None:
            continue
        ax.plot(
            [coords[src_idx, 0], coords[dst_idx, 0]],
            [coords[src_idx, 1], coords[dst_idx, 1]],
            color="#3b82f6",
            alpha=min(0.5, 0.12 + weight * 0.22),
            linewidth=0.4 + weight * 1.4,
        )

    ax.set_title(
        f"CWM Point Universe | anchors={len(display_tokens)} shown / {len(core.anchors)} total | step={core.step}",
        fontsize=14,
    )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.grid(alpha=0.12)

    importance_note = "\n".join(
        f"{i+1}. {safe_token_label(tok)} ({core.anchors[tok].importance:.2f})"
        for i, tok in enumerate(top_tokens[:12])
    )
    ax.text(
        1.02,
        0.98,
        "Top Importance\n" + importance_note,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
    )

    fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02, label="Cluster")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CWM anchors as a 2D universe map.")
    parser.add_argument("--model", type=str, default=None, help="Path to a CWM .pt model file")
    parser.add_argument("--output", type=str, default="cwm_universe.png", help="PNG output path")
    parser.add_argument("--max-points", type=int, default=1200, help="Maximum number of points to render")
    parser.add_argument("--label-top-n", type=int, default=80, help="How many high-importance tokens to annotate")
    parser.add_argument("--edge-top-n", type=int, default=120, help="How many strong forward-gravity edges to draw")
    parser.add_argument("--include-internal", action="store_true", help="Include internal/non-emittable tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for clustering/PCA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.model)
    core = CWMCore.load(str(model_path))
    output_path = Path(args.output)
    build_plot(
        core=core,
        output_path=output_path,
        max_points=args.max_points,
        label_top_n=args.label_top_n,
        edge_top_n=args.edge_top_n,
        include_internal=args.include_internal,
        random_seed=args.seed,
    )
    print(f"model   : {model_path}")
    print(f"output  : {output_path}")
    print(f"anchors : {len(core.anchors)}")
    print(f"step    : {core.step}")


if __name__ == "__main__":
    main()
