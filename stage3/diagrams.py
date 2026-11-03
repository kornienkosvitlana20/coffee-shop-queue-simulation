"""Схема багатоканальної СМО і граф станів  для c станцій і черги m."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

OUT = "report/diagrams"


def scheme(c=3, m=6):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    arrow = dict(arrowstyle="-|>", mutation_scale=16, color="#34495e", lw=1.6)
    ax.add_patch(FancyArrowPatch((0.2, 2.3), (1.6, 2.3), **arrow))
    ax.text(0.9, 2.55, "Потік клієнтів\nλ (Пуассон)", ha="center", fontsize=10)
    # черга
    for k in range(m):
        ax.add_patch(FancyBboxPatch((1.7 + k * 0.5, 2.0), 0.42, 0.6, boxstyle="round,pad=0.02",
                                    fc="#eef4fb", ec="#34495e"))
    ax.text(1.7 + m * 0.25, 2.8, f"Спільна черга (не більше m = {m} клієнтів)", ha="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((1.7 + m * 0.5 + 0.05, 2.3), (5.6, 2.3), **arrow))
    # станції
    ys = [3.6 - i * (3.0 / max(c - 1, 1)) for i in range(c)] if c > 1 else [2.3]
    ys = [3.7, 2.3, 0.9][:c] if c == 3 else ys
    for i, y in enumerate(ys):
        ax.add_patch(FancyBboxPatch((5.9, y - 0.35), 1.7, 0.7, boxstyle="round,pad=0.03",
                                    fc="#d5efd9", ec="#34495e"))
        ax.text(6.75, y, f"Станція {i + 1}\n(μ)", ha="center", va="center", fontsize=10)
        ax.add_patch(FancyArrowPatch((5.6, 2.3), (5.9, y), **arrow))
        ax.add_patch(FancyArrowPatch((7.6, y), (8.7, 2.3), **arrow))
    ax.text(9.7, 2.3, "Обслужені\nклієнти\n(виручка)", ha="center", va="center", fontsize=10)
    # відмова
    ax.add_patch(FancyArrowPatch((1.8, 2.0), (1.8, 0.7), color="#c0392b", arrowstyle="-|>",
                                 mutation_scale=16, lw=1.6, ls="--"))
    ax.text(1.95, 0.4, f"Черга повна (m = {m}):\nклієнт іде до конкурентів", fontsize=10, color="#c0392b")
    ax.set_title(f"Схема багатоканальної СМО з обмеженою чергою (c = {c}, m = {m})", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/smo_scheme.png", dpi=170); plt.close(fig)


def state_graph(c=3, m=6):
    n = c + m + 1
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(-0.8, n * 1.5 - 0.2); ax.set_ylim(-1.6, 1.9); ax.axis("off")
    arrow = dict(arrowstyle="-|>", mutation_scale=14, lw=1.4)
    for k in range(n):
        x = k * 1.5
        col = "#d5efd9" if k <= c else "#fbe0dc" if k == n - 1 else "#fdf3d0"
        ax.add_patch(Circle((x, 0), 0.42, fc=col, ec="#34495e", lw=1.5))
        ax.text(x, 0, f"S{k}", ha="center", va="center", fontsize=11)
        if k < n - 1:
            ax.add_patch(FancyArrowPatch((x + 0.42, 0.22), (x + 1.5 - 0.42, 0.22), color="#2874a6", **arrow))
            ax.text(x + 0.75, 0.5, "λ", ha="center", fontsize=11, color="#2874a6")
            mu = f"{min(k + 1, c)}μ" if min(k + 1, c) > 1 else "μ"
            ax.add_patch(FancyArrowPatch((x + 1.5 - 0.42, -0.22), (x + 0.42, -0.22), color="#b9770e", **arrow))
            ax.text(x + 0.75, -0.62, mu, ha="center", fontsize=11, color="#b9770e")
    ax.text(c * 1.5 / 2, 1.1, f"Усі станції не зайняті повністю\n(k = 0…{c}: у черзі нікого)", ha="center", fontsize=9)
    ax.text((c + 1 + m / 2 - 0.5) * 1.5, 1.1, f"Зайняті всі {c} станцій, у черзі 1…{m} клієнтів", ha="center", fontsize=9)
    ax.text((n - 1) * 1.5, -1.2, "відмова\n(черга повна)", ha="center", fontsize=9, color="#c0392b")
    ax.set_title(f"Граф станів СМО (c = {c}, m = {m}): Sk — у системі k клієнтів, {n} станів", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/state_graph.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    c = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    scheme(c); state_graph(c)