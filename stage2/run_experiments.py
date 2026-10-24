"""Експерименти для багатоканальної моделі: прогін, 100 прогонів, тести адекватності, оптимальний розклад."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import params
from stage1.tests import patched
from stage1.single_channel import LOST
from stage2.multi_channel import make_day, run_many, simulate_day

R = "results"
OPT = [3, 2, 1, 2, 1]        # максимум прибутку (за перебором)
REC = [3, 2, 2, 2, 1]        # рекомендований: втрати < 1 %
PLABELS = [f"{s:02d}-{e:02d}" for s, e, _ in params.PERIODS]
DAYS = [make_day(s) for s in range(params.N_RUNS)]


def by_period(schedule, days=DAYS):
    """Середній % втрат, очікування і виручка по періодах (за 100 прогонів)."""
    lost, wait, rev = [], [], []
    for cl in days:
        _, _, log = simulate_day(schedule, clients=cl, detail=True)
        g = log.groupby("period")
        lost.append(g.status.apply(lambda s: 100 * (s == LOST).mean()))
        wait.append(g.wait.mean())
        rev.append(log[log.status != LOST].groupby("period").price.sum())
    return pd.DataFrame({"lost_pct": pd.DataFrame(lost).mean(), "avg_wait": pd.DataFrame(wait).mean(),
                         "revenue": pd.DataFrame(rev).mean()}).reindex(PLABELS)


def one_run():
    summary, steps, log = simulate_day(OPT, seed=1, detail=True)
    log.to_csv(f"{R}/stage2_log_clients.csv", index=False)
    steps.to_csv(f"{R}/stage2_log_steps.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x = range(len(steps))
    ax[0].bar(x, steps.queue_len, color="steelblue", label="Черга")
    ax[0].step(x, steps.open_stations, where="mid", color="black", label="Відкрито станцій")
    ax[0].set_title(f"Черга і відкриті станції, розклад {OPT}")
    ax[0].set_xlabel("Крок (10 хв від 07:00)"); ax[0].set_ylabel("Клієнтів / станцій"); ax[0].legend()
    ax[1].bar(x, steps.busy_stations, color="seagreen")
    ax[1].set_title("Зайняті станції наприкінці кроку")
    ax[1].set_xlabel("Крок (10 хв від 07:00)"); ax[1].set_ylabel("Станцій")
    w = log[log.status != LOST].wait
    ax[2].hist(w, bins=20, color="darkorange", edgecolor="white")
    ax[2].set_title("Час очікування обслужених клієнтів"); ax[2].set_xlabel("хв"); ax[2].set_ylabel("Клієнтів")
    fig.tight_layout(); fig.savefig(f"{R}/stage2_one_run.png", dpi=150); plt.close(fig)
    return summary


def hundred_runs():
    rows = {}
    for name, sch in [("1 станція весь день", [1] * 5), ("2 станції", [2] * 5), ("3 станції", [3] * 5),
                      ("5 станцій", [5] * 5), (f"Рекомендований {REC}", REC), (f"Оптимальний {OPT}", OPT)]:
        d = run_many(sch, days=DAYS)
        rows[name] = d.mean()
        if sch == OPT:
            d.to_csv(f"{R}/stage2_100_runs_opt.csv", index=False)
            dopt = d
    tab = pd.DataFrame(rows).T[["lost_pct", "revenue", "cost", "profit", "avg_wait", "max_wait", "utilization"]]
    tab.to_csv(f"{R}/stage2_compare_schedules.csv")
    sd = dopt[["lost_pct", "profit", "avg_wait", "utilization"]].agg(["mean", "std"]).T
    sd.to_csv(f"{R}/stage2_opt_summary.csv")
    bp = by_period(OPT); bp.to_csv(f"{R}/stage2_opt_by_period.csv")
    bpr = by_period(REC); bpr.to_csv(f"{R}/stage2_rec_by_period.csv")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].hist(dopt.profit, bins=15, color="steelblue", edgecolor="white")
    ax[0].set_title(f"Розподіл прибутку за добу, {OPT}"); ax[0].set_xlabel("грн"); ax[0].set_ylabel("Кількість прогонів")
    ax[1].hist(dopt.lost_pct, bins=15, color="indianred", edgecolor="white")
    ax[1].set_title("Розподіл частки втрачених клієнтів"); ax[1].set_xlabel("%"); ax[1].set_ylabel("Кількість прогонів")
    bp.lost_pct.plot.bar(ax=ax[2], color="darkorange")
    ax[2].set_title("Середній % втрат по періодах"); ax[2].set_xlabel("Період"); ax[2].set_ylabel("%")
    fig.tight_layout(); fig.savefig(f"{R}/stage2_100_runs.png", dpi=150); plt.close(fig)
    return tab, sd, bp, bpr, dopt


def adequacy():
    out = []
    # 1) c постійне: втрати падають, витрати ростуть
    rows = []
    for c in range(1, 6):
        m = run_many([c] * 5, days=DAYS).mean()
        rows.append(dict(c=c, lost_pct=m.lost_pct, avg_wait=m.avg_wait, utilization=m.utilization,
                         cost=m.cost, profit=m.profit))
    t1 = pd.DataFrame(rows).set_index("c"); t1.to_csv(f"{R}/stage2_test_c.csv")
    out.append(("c↑: втрати спадають", bool((t1.lost_pct.diff().dropna() <= 0.3).all())))
    out.append(("c↑: очікування спадає", bool((t1.avg_wait.diff().dropna() <= 0.05).all())))
    out.append(("c↑: витрати ростуть", bool((t1.cost.diff().dropna() > 0).all())))
    out.append(("c↑: завантаженість станції спадає", bool((t1.utilization.diff().dropna() <= 0).all())))
    # 2) λ
    rows = []
    for f in [0.5, 1, 1.5, 2]:
        with patched(PERIODS=[(s, e, l * f) for s, e, l in params.PERIODS]):
            days = [make_day(s) for s in range(params.N_RUNS)]
            m = run_many(OPT, days=days).mean()
        rows.append(dict(factor=f, clients=m.clients, lost_pct=m.lost_pct, avg_wait=m.avg_wait, profit=m.profit))
    t2 = pd.DataFrame(rows).set_index("factor"); t2.to_csv(f"{R}/stage2_test_lambda.csv")
    out.append(("λ↑: втрати зростають", bool((t2.lost_pct.diff().dropna() >= -0.3).all())))
    # 3) MAX_QUEUE
    rows = []
    for q in [0, 3, 6, 12]:
        m = run_many(OPT, max_queue=q, days=DAYS).mean()
        rows.append(dict(max_queue=q, lost_pct=m.lost_pct, avg_wait=m.avg_wait, profit=m.profit))
    t3 = pd.DataFrame(rows).set_index("max_queue"); t3.to_csv(f"{R}/stage2_test_queue.csv")
    out.append(("MAX_QUEUE↑: втрати спадають", bool((t3.lost_pct.diff().dropna() <= 0.3).all())))
    out.append(("MAX_QUEUE↑: очікування зростає", bool((t3.avg_wait.diff().dropna() >= -0.05).all())))
    # 4) баланс
    s, _, log = simulate_day(OPT, seed=3, detail=True)
    out.append(("обслужені + втрачені = усі клієнти", s["served"] + s["lost"] == s["clients"]))
    out.append(("завантаженість ≤ 1", s["utilization"] <= 1))
    out.append(("початок обслуговування ≥ прибуття", bool((log.dropna(subset=["start"]).eval("start >= arrival - 1e-9")).all())))
    # одночасно ніхто не обслуговується більше ніж станцій відкрито (перевірка накладання)
    ok = True
    for st, g in log[log.status != LOST].groupby("station"):
        g = g.sort_values("start")
        ok &= bool((g.start.values[1:] >= g.finish.values[:-1] - 1e-9).all())
    out.append(("станція не обслуговує двох клієнтів одночасно", ok))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(t1.index, t1.lost_pct, "o-", color="indianred", label="Втрати, %")
    ax2 = ax[0].twinx(); ax2.plot(t1.index, t1.profit, "s-", color="steelblue", label="Прибуток, грн")
    ax[0].set_xlabel("Кількість станцій (постійна)"); ax[0].set_ylabel("Втрати, %"); ax2.set_ylabel("Прибуток, грн")
    ax[0].set_title("Тест 1: кількість станцій")
    ax[1].plot(t2.index, t2.lost_pct, "o-", color="indianred")
    ax[1].set_title(f"Тест 2: множник λ (розклад {OPT})"); ax[1].set_xlabel("Множник λ"); ax[1].set_ylabel("Втрати, %")
    ax[2].plot(t3.index, t3.lost_pct, "o-", color="indianred", label="Втрати, %")
    ax[2].plot(t3.index, t3.avg_wait, "s-", color="seagreen", label="Очікування, хв")
    ax[2].set_title("Тест 3: максимальна черга"); ax[2].set_xlabel("MAX_QUEUE"); ax[2].legend()
    for a in ax: a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{R}/stage2_tests.png", dpi=150); plt.close(fig)
    return t1, t2, t3, out


def grid_plots():
    g = pd.read_csv(f"{R}/stage2_grid.csv")
    top = g.sort_values("profit", ascending=False).head(10)
    top.round(2).to_csv(f"{R}/stage2_top10.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax[0].scatter(g.lost_pct, g.profit, c=g.cost, cmap="viridis", s=8)
    ax[0].scatter(g.loc[g.profit.idxmax(), "lost_pct"], g.profit.max(), c="red", s=60, marker="*", label="Максимум прибутку")
    ax[0].set_xlabel("Втрачені клієнти, %"); ax[0].set_ylabel("Прибуток, грн"); ax[0].set_xlim(-1, 30)
    ax[0].set_title("Усі 3125 розкладів: прибуток і втрати"); ax[0].legend()
    fig.colorbar(sc, ax=ax[0], label="Витрати на станції, грн")
    # найкращий прибуток по кількості станцій у кожному періоді
    best = [g.groupby(f"c{i + 1}").profit.max() for i in range(5)]
    for i, b in enumerate(best):
        ax[1].plot(b.index, b.values, "o-", label=PLABELS[i])
    ax[1].set_xlabel("Кількість станцій у періоді"); ax[1].set_ylabel("Найкращий прибуток серед розкладів, грн")
    ax[1].set_title("Внесок кожного періоду"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{R}/stage2_grid.png", dpi=150); plt.close(fig)
    return top


if __name__ == "__main__":
    print("ONE", one_run())
    tab, sd, bp, bpr, dopt = hundred_runs()
    print(tab.round(2)); print(sd.round(2)); print(bp.round(2)); print(bpr.round(2))
    t1, t2, t3, out = adequacy()
    print(t1.round(2)); print(t2.round(2)); print(t3.round(2))
    for n, ok in out: print("OK  " if ok else "FAIL", n)
    print(grid_plots()[["c1", "c2", "c3", "c4", "c5", "lost_pct", "profit"]].round(2))
    # парне порівняння OPT і REC
    a = run_many(OPT, days=DAYS).profit; b = run_many(REC, days=DAYS).profit
    d = a - b
    print("Різниця прибутку OPT-REC: середнє %.1f, ст.помилка %.1f" % (d.mean(), d.std() / len(d) ** .5))
