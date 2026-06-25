"""
CristalVisualizer — visualización del espacio interno de LLMs.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional


PHASE_COLORS = {
    "chaos":           "#e74c3c",
    "semantic":        "#f39c12",
    "crystallization": "#2ecc71",
}

CRISTAL_CMAP = LinearSegmentedColormap.from_list(
    "cristal", ["#1a1a2e", "#e74c3c", "#f39c12", "#2ecc71"]
)


class CristalVisualizer:

    @staticmethod
    def heatmap(result, ax=None, show_answer: bool = True):
        """Mapa de calor por capa: rank del token correcto."""
        profiles = result.profiles
        n = len(profiles)
        ranks = np.array([p.logitlens_rank for p in profiles], dtype=float)

        # Normalizar: rank 0 = verde, rank alto = rojo
        scores = 1.0 - np.clip(ranks / 1000, 0, 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 3))

        im = ax.imshow(scores.reshape(1, -1), aspect="auto", cmap=CRISTAL_CMAP,
                       vmin=0, vmax=1)

        # Marcar cristalización
        if result.crystal_layer is not None:
            ax.axvline(result.crystal_layer, color="white", linewidth=2, linestyle="--")
            ax.text(result.crystal_layer + 0.3, 0,
                    f"L{result.crystal_layer}\n↑crystal",
                    color="white", fontsize=7, va="center")

        ax.set_yticks([])
        ax.set_xticks(range(0, n, max(1, n // 10)))
        ax.set_xticklabels([f"L{i}" for i in range(0, n, max(1, n // 10))], fontsize=7)

        title = f"{result.model_id.split('/')[-1]}"
        if show_answer:
            title += f' | "{result.prompt[:40]}..." → "{result.answer}"'
        title += f" | crystal={result.crystal_layer} ({result.crystal_type})"
        ax.set_title(title, fontsize=9)

        return ax

    @staticmethod
    def rank_curve(result, ax=None, log_scale: bool = True):
        """Curva de rank del token correcto por capa."""
        profiles = result.profiles
        layers = [p.layer for p in profiles]
        ranks  = [p.logitlens_rank for p in profiles]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        # Colorear por fase
        for p in profiles[:-1]:
            color = PHASE_COLORS.get(result.phases.get(p.layer, "semantic"), "#888")
            ax.fill_betweenx(
                [1, max(ranks) * 1.1],
                p.layer, p.layer + 1,
                alpha=0.15, color=color
            )

        ax.plot(layers, ranks, color="white", linewidth=1.5, zorder=5)
        ax.scatter(layers, ranks, c=[
            PHASE_COLORS.get(result.phases.get(l, "semantic"), "#888") for l in layers
        ], s=20, zorder=6)

        if result.crystal_layer is not None:
            ax.axvline(result.crystal_layer, color="#2ecc71", linewidth=1.5,
                       linestyle="--", label=f"crystal L{result.crystal_layer}")

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Layer", fontsize=9)
        ax.set_ylabel("Rank of correct token", fontsize=9)
        ax.set_title(f'{result.model_id.split("/")[-1]} — "{result.answer}"', fontsize=9)
        ax.invert_yaxis()
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#444")

        return ax

    @staticmethod
    def trajectory_2d(result, ax=None, method: str = "pca"):
        """Trayectoria de activaciones en 2D (PCA o UMAP)."""
        if result.hidden_states is None:
            raise ValueError("store_hidden=True requerido en CristalAnalyzer.analyze()")

        hs = result.hidden_states  # (n_layers, dim)

        if method == "pca":
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2).fit_transform(hs)
        elif method == "umap":
            import umap
            coords = umap.UMAP(n_components=2, random_state=42).fit_transform(hs)
        else:
            raise ValueError(f"method debe ser 'pca' o 'umap'")

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        n = len(coords)
        colors = [PHASE_COLORS.get(result.phases.get(l, "semantic"), "#888")
                  for l in range(n)]

        ax.plot(coords[:, 0], coords[:, 1], color="#444", linewidth=0.8, zorder=1)
        ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=40, zorder=2)

        # Marcar inicio, cristalización y final
        ax.scatter(*coords[0], color="white", s=80, marker="o", zorder=3, label="L0")
        ax.scatter(*coords[-1], color="#3498db", s=80, marker="*", zorder=3, label=f"L{n-1}")
        if result.crystal_layer is not None:
            ax.scatter(*coords[result.crystal_layer], color="#2ecc71",
                       s=120, marker="D", zorder=4, label=f"crystal L{result.crystal_layer}")

        # Anotar algunas capas
        for i in range(0, n, max(1, n // 6)):
            ax.annotate(f"L{i}", coords[i], fontsize=6, color="white",
                        xytext=(3, 3), textcoords="offset points")

        ax.set_title(f'Trayectoria {method.upper()} — {result.model_id.split("/")[-1]}', fontsize=9)
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white", labelsize=7)
        ax.legend(fontsize=7, facecolor="#2a2a3e", labelcolor="white")

        return ax

    @staticmethod
    def compare_models(results: list, figsize=(14, 5)):
        """Panel comparativo: un heatmap por modelo."""
        n = len(results)
        fig, axes = plt.subplots(n, 1, figsize=figsize)
        fig.patch.set_facecolor("#0d0d1a")

        if n == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            ax.set_facecolor("#1a1a2e")
            CristalVisualizer.heatmap(result, ax=ax)
            ax.tick_params(colors="white", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#333")

        fig.suptitle("Cristaloscope — comparación cross-model", fontsize=11,
                     color="white", y=1.01)
        plt.tight_layout()
        return fig

    @staticmethod
    def full_report(result, store_hidden: bool = True, figsize=(14, 10)):
        """Panel completo: heatmap + rank curve + trayectoria 2D + stats."""
        fig = plt.figure(figsize=figsize, facecolor="#0d0d1a")
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

        ax_heat  = fig.add_subplot(gs[0, :])
        ax_rank  = fig.add_subplot(gs[1, 0])
        ax_traj  = fig.add_subplot(gs[1, 1])

        ax_heat.set_facecolor("#1a1a2e")
        ax_rank.set_facecolor("#1a1a2e")
        ax_traj.set_facecolor("#1a1a2e")

        CristalVisualizer.heatmap(result, ax=ax_heat)
        CristalVisualizer.rank_curve(result, ax=ax_rank)

        if result.hidden_states is not None:
            CristalVisualizer.trajectory_2d(result, ax=ax_traj)
        else:
            ax_traj.text(0.5, 0.5, "store_hidden=True\nrequired for trajectory",
                         ha="center", va="center", color="white", transform=ax_traj.transAxes)

        for ax in [ax_heat, ax_rank, ax_traj]:
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("#333")

        fig.suptitle(
            f'Cristaloscope — {result.model_id.split("/")[-1]}\n'
            f'"{result.prompt}" → "{result.answer}" | '
            f'crystal=L{result.crystal_layer} ({result.crystal_type})',
            fontsize=10, color="white"
        )
        return fig
