"""Етап 3: розрахунки за аналітичною моделлю і порівняння з імітацією (100 прогонів)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import params
from stage2.multi_channel import make_day, run_many
from stage2.run_experiments import by_period, OPT, REC, PLABELS, DAYS
from stage3.analytical import day_summary, mean_order, period_table, smo

R = "results"


def analytic_optimum():
    """Для кожного періоду окремо: c, що максимізує аналітичний прибуток."""
    rows, best = [], []
    for c_idx, (s, e, lam) in enumerate(params.PERIODS):
        prof = {}
        for c in range(1, params.MAX_STATIONS + 1):
            sch = [1] * 5; sch[c_idx] = c
            prof[c] = period_table(sch).profit[c_idx]
            rows.append(dict(period=PLABELS[c_idx], c=c, profit=prof[c],
                             p_rej=period_table(sch).p_rej[c_idx]))
        best.append(max(prof, key=prof.get))
    return best, pd.DataFrame(rows)


if __name__ == "__main__":
    n, t, price = mean_order()
    mu = 60 / t
    par = pd.DataFrame([dict(period=PLABELS[i], lam=l, hours=e - s, mu=round(mu, 3), r=round(l / mu, 3),
                             rho_c1=round(l / mu, 3)) for i, (s, e, l) in enumerate(params.PERIODS)])
    par.to_csv(f"{R}/stage3_params.csv", index=False)
    print(f"Напоїв {n:.3f}; час {t:.3f} хв; μ={mu:.3f}; сума {price:.1f} грн"); print(par)

    best, tab_c = analytic_optimum()
    tab_c.to_csv(f"{R}/stage3_period_profit_by_c.csv", index=False)
    print("Аналітичний оптимум по періодах:", best)

    scheds = {"1 станція": [1] * 5, "2 станції": [2] * 5, "3 станції": [3] * 5, "5 станцій": [5] * 5,
              f"Рекомендований {REC}": REC, f"Оптимальний {OPT}": OPT, f"Аналітичний опт. {best}": best}
    rows = []
    for name, sch in scheds.items():
        a = day_summary(sch)
        s = run_many(sch, days=DAYS).mean()
        rows.append(dict(розклад=name, втрати_ан=a["lost_pct"], втрати_імітація=s.lost_pct,
                         виручка_ан=a["revenue"], виручка_імітація=s.revenue,
                         прибуток_ан=a["profit"], прибуток_імітація=s.profit))
    cmp_ = pd.DataFrame(rows).set_index("розклад")
    cmp_["відхилення_прибутку_%"] = 100 * (cmp_.прибуток_ан - cmp_.прибуток_імітація) / cmp_.прибуток_імітація
    cmp_.round(2).to_csv(f"{R}/stage3_compare_day.csv"); print(cmp_.round(2).to_string())

    # по періодах для розкладу REC і OPT
    allp = []
    for name, sch in [("REC", REC), ("OPT", OPT)]:
        a = period_table(sch); s = by_period(sch)
        d = pd.DataFrame({"розклад": name, "період": PLABELS, "c": sch,
                          "втрати_ан_%": 100 * a.p_rej.values, "втрати_імітація_%": s.lost_pct.values,
                          "очікування_ан_хв": a.Wq_min.values, "очікування_імітація_хв": s.avg_wait.values,
                          "виручка_ан": a.revenue.values, "виручка_імітація": s.revenue.values})
        allp.append(d)
    perp = pd.concat(allp); perp.round(2).to_csv(f"{R}/stage3_compare_periods.csv", index=False)
    print(perp.round(2).to_string())

    # таблиця аналітичних характеристик для піку (07-10) залежно від c
    rows = []
    for c in range(1, 6):
        r = smo(params.PERIODS[0][2], mu, c, params.MAX_QUEUE)
        rows.append(dict(c=c, p_rej=r["p_rej"], p_queue=r["p_queue"], A=r["A"], k_busy=r["k_busy"],
                         Lq=r["Lq"], Wq_min=r["Wq_min"], D=price * r["A"] - params.STATION_COST_PER_HOUR * c))
    peak = pd.DataFrame(rows).set_index("c"); peak.round(4).to_csv(f"{R}/stage3_peak.csv"); print(peak.round(3))

    # графіки
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, p in enumerate(params.PERIODS):
        d = tab_c[tab_c.period == PLABELS[i]]
        ax[0].plot(d.c, d.profit / (p[1] - p[0]), "o-", label=f"{PLABELS[i]} (λ={p[2]})")
    ax[0].set_xlabel("Кількість станцій c"); ax[0].set_ylabel("Прибуток D, грн/год")
    ax[0].set_title("Аналітична модель: прибуток за годину"); ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_xticks(range(1, 6)); ax[0].grid(alpha=.3)
    peak.p_rej.mul(100).plot(ax=ax[1], marker="o", color="indianred", label="Відмова, %")
    peak.Wq_min.plot(ax=ax[1], marker="s", color="seagreen", label="Очікування в черзі, хв")
    ax[1].set_title("Пік 07-10: відмова й очікування (аналітика)"); ax[1].set_xlabel("c"); ax[1].legend(); ax[1].set_xticks(range(1, 6)); ax[1].grid(alpha=.3)
    x = np.arange(len(cmp_))
    ax[2].bar(x - .2, cmp_.прибуток_ан, .4, label="Аналітика", color="steelblue")
    ax[2].bar(x + .2, cmp_.прибуток_імітація, .4, label="Імітація (100 прогонів)", color="darkorange")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["c=1", "c=2", "c=3", "c=5", "REC", "OPT", "АН"], fontsize=9)
    ax[2].set_ylabel("Прибуток за добу, грн"); ax[2].set_title("Прибуток: аналітика і імітація"); ax[2].legend()
    fig.tight_layout(); fig.savefig(f"{R}/stage3_analytics.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for j, name in enumerate(["REC", "OPT"]):
        d = perp[perp.розклад == name]; xx = np.arange(5)
        ax[j].bar(xx - .2, d["втрати_ан_%"], .4, label="Аналітика", color="steelblue")
        ax[j].bar(xx + .2, d["втрати_імітація_%"], .4, label="Імітація", color="darkorange")
        ax[j].set_xticks(xx); ax[j].set_xticklabels(PLABELS)
        ax[j].set_title(f"Втрати по періодах, розклад {REC if name == 'REC' else OPT}")
        ax[j].set_ylabel("%"); ax[j].set_xlabel("Період"); ax[j].legend()
    fig.tight_layout(); fig.savefig(f"{R}/stage3_compare_periods.png", dpi=150); plt.close(fig)
