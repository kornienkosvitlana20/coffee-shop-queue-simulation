"""Аналітична модель: багатоканальна СМО з обмеженою чергою (M/M/c/m)."""
from math import factorial

import pandas as pd

import params


def mean_order():
    """Середні показники одного замовлення: напоїв, час (хв), сума (грн)."""
    n = 1 / params.P_GEOM                                  # математичне сподівання геом. розподілу
    t = n * sum(p * tm for _, _, tm, p in params.MENU)
    price = n * sum(p * pr for _, pr, _, p in params.MENU)
    return n, t, price


def smo(lam, mu, c, m):
    """Характеристики СМО: lam, mu — 1/год; c — канали; m — місця в черзі."""
    r = lam / mu                      # навантаження
    rho = r / c                       # приведена завантаженість
    head = sum(r ** i / factorial(i) for i in range(c))
    geo = (m + 1) if abs(rho - 1) < 1e-12 else (1 - rho ** (m + 1)) / (1 - rho)
    q0 = 1 / (head + r ** c / factorial(c) * geo)
    q = [r ** k / factorial(k) * q0 for k in range(c + 1)]
    q += [r ** c / factorial(c) * rho ** j * q0 for j in range(1, m + 1)]
    p_rej = q[c + m]
    Q = 1 - p_rej
    A = lam * Q
    lq = sum(j * q[c + j] for j in range(1, m + 1))
    k_busy = r * Q
    return dict(q0=q0, p_queue=sum(q[c:c + m]), p_rej=p_rej, Q=Q, A=A,
                k_busy=k_busy, Lq=lq, Ls=lq + k_busy,
                Wq_min=60 * lq / lam, Ws_min=60 * (lq + k_busy) / lam,
                load=k_busy / c, q=q)


def period_table(schedule, m=None):
    """Показники по кожному періоду для розкладу (кількість станцій у періоді)."""
    m = params.MAX_QUEUE if m is None else m
    _, t_serv, price = mean_order()
    mu = 60 / t_serv
    rows = []
    for c, (s, e, lam) in zip(schedule, params.PERIODS):
        res = smo(lam, mu, c, m)
        hours = e - s
        revenue = price * res["A"] * hours
        cost = params.STATION_COST_PER_HOUR * c * hours
        rows.append(dict(period=f"{s:02d}-{e:02d}", lam=lam, c=c, hours=hours,
                         clients=lam * hours, lost=lam * hours * res["p_rej"],
                         p_rej=res["p_rej"], A=res["A"], Lq=res["Lq"],
                         Wq_min=res["Wq_min"], load=res["load"],
                         revenue=revenue, cost=cost, profit=revenue - cost))
    return pd.DataFrame(rows)


def day_summary(schedule, m=None):
    t = period_table(schedule, m)
    clients, lost = t.clients.sum(), t.lost.sum()
    return dict(clients=clients, lost=lost, lost_pct=100 * lost / clients,
                revenue=t.revenue.sum(), cost=t.cost.sum(), profit=t.profit.sum())


if __name__ == "__main__":
    n, t, price = mean_order()
    print(f"Напоїв {n:.3f}, час {t:.3f} хв, сума {price:.1f} грн, mu = {60 / t:.3f} 1/год")
    # перевірка за прикладом з методички: lam=7, tобсл=0,75 год, c=3, m=4
    ex = smo(7, 4 / 3, 3, 4)
    print({k: round(v, 6) for k, v in ex.items() if k != "q"})
    print(period_table([3, 2, 2, 2, 1]).round(3).to_string())