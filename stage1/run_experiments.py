"""Один прогін (графіки) і 100 прогонів (усереднені показники) для одного каналу."""
import os

import matplotlib.pyplot as plt
import pandas as pd

import params
from stage1.single_channel import simulate_day, SERVED, LOST

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def plot_one_run(seed=1):
    summary, steps, clients = simulate_day(seed)
    clients.to_csv(f"{RESULTS}/stage1_log_clients.csv", index=False)
    steps.to_csv(f"{RESULTS}/stage1_log_steps.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].bar(steps.time, steps.queue_len, color="steelblue")
    axes[0].set_title("Довжина черги наприкінці кожного кроку (10 хв)")
    axes[0].set_xlabel("Час")
    axes[0].set_ylabel("Клієнтів у черзі")
    axes[0].set_xticks(range(0, len(steps), 6))
    axes[0].set_xticklabels(steps.time[::6], rotation=45)

    axes[1].bar(steps.time, steps.lost, color="indianred")
    axes[1].set_title("Втрачені клієнти за кроками")
    axes[1].set_xlabel("Час")
    axes[1].set_ylabel("Клієнтів")
    axes[1].set_xticks(range(0, len(steps), 6))
    axes[1].set_xticklabels(steps.time[::6], rotation=45)

    waits = clients[clients.status == SERVED].wait
    axes[2].hist(waits, bins=20, color="seagreen", edgecolor="white")
    axes[2].set_title("Час очікування обслужених клієнтів")
    axes[2].set_xlabel("хв")
    axes[2].set_ylabel("Клієнтів")

    fig.tight_layout()
    fig.savefig(f"{RESULTS}/stage1_one_run.png", dpi=150)
    plt.close(fig)
    print("Один прогін збережено у results/")


def run_many(n=params.N_RUNS):
    rows, lost_by_period = [], []
    for seed in range(n):
        summary, _, clients = simulate_day(seed)
        rows.append(summary)
        by_period = clients.groupby("period").status.apply(
            lambda s: 100 * (s == LOST).mean())
        lost_by_period.append(by_period)

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS}/stage1_100_runs.csv", index=False)

    mean_table = pd.DataFrame({"середнє": df.mean(), "стд. відхилення": df.std()}).round(2)
    mean_table.to_csv(f"{RESULTS}/stage1_summary.csv")
    print(f"Середні показники за {n} прогонів:")
    print(mean_table)

    period_table = pd.DataFrame(lost_by_period).mean().round(1)
    period_table.to_csv(f"{RESULTS}/stage1_lost_by_period.csv")
    print("\nСередній % втрат по періодах:")
    print(period_table)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(df.profit, bins=15, color="steelblue", edgecolor="white")
    axes[0].set_title("Розподіл прибутку за добу")
    axes[0].set_xlabel("грн")
    axes[1].hist(df.lost_pct, bins=15, color="indianred", edgecolor="white")
    axes[1].set_title("Розподіл частки втрачених клієнтів")
    axes[1].set_xlabel("%")
    period_table.plot.bar(ax=axes[2], color="darkorange")
    axes[2].set_title("Середній % втрат по періодах доби")
    axes[2].set_xlabel("Період (години)")
    axes[2].set_ylabel("%")
    for ax in axes:
        ax.set_ylabel(ax.get_ylabel() or "Кількість прогонів")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/stage1_100_runs.png", dpi=150)
    plt.close(fig)
    print("Графіки збережено у results/")


if __name__ == "__main__":
    plot_one_run()
    run_many()