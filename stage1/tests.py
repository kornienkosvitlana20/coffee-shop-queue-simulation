import copy
from contextlib import contextmanager

import matplotlib.pyplot as plt
import pandas as pd

import params
from stage1.single_channel import simulate_day

RESULTS = "results"


@contextmanager
def patched(**changes):
    """Тимчасово змінює параметри в params і повертає їх назад."""
    old = {k: copy.deepcopy(getattr(params, k)) for k in changes}
    try:
        for k, v in changes.items():
            setattr(params, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(params, k, v)


def mean_summary(n=100, **sim_kwargs):
    """Середні показники за n прогонів."""
    rows = [simulate_day(seed, **sim_kwargs)[0] for seed in range(n)]
    return pd.DataFrame(rows).mean()


def check(df, column, direction, title):
    """Перевірка монотонності показника за зміною параметра."""
    diffs = df[column].diff().dropna()
    ok = (diffs >= -0.5).all() if direction == "up" else (diffs <= 0.5).all()
    word = "зростає" if direction == "up" else "спадає"
    print(f"{'OK  ' if ok else 'FAIL'} {title}: очікуємо, що {column} {word}")
    return ok


def test_lambda():
    """Більше клієнтів, тобто більше втрат, довша черга, більше завантаження."""
    rows = []
    for f in [0.25, 0.5, 1.0, 1.5, 2.0]:
        periods = [(s, e, lam * f) for s, e, lam in params.PERIODS]
        with patched(PERIODS=periods):
            m = mean_summary()
        m["значення"] = f
        rows.append(m)
    df = pd.DataFrame(rows).set_index("значення")
    df.to_csv(f"{RESULTS}/stage1_test_lambda.csv")
    print("\nТест 1: множник інтенсивності потоку λ")
    print(df[["clients", "lost_pct", "avg_queue", "avg_wait", "utilization", "profit"]].round(2))
    return df, [check(df, "lost_pct", "up", "λ↑"),
                check(df, "avg_queue", "up", "λ↑"),
                check(df, "utilization", "up", "λ↑")]


def test_queue_limit():
    """Довша допустима черга: менше втрат, але довше очікування."""
    rows = []
    for q in [0, 2, 6, 12, 30]:
        m = mean_summary(max_queue=q)
        m["значення"] = q
        rows.append(m)
    df = pd.DataFrame(rows).set_index("значення")
    df.to_csv(f"{RESULTS}/stage1_test_queue.csv")
    print("\nТест 2: максимальна довжина черги")
    print(df[["lost_pct", "avg_queue", "avg_wait", "utilization", "profit"]].round(2))
    return df, [check(df, "lost_pct", "down", "MAX_QUEUE↑"),
                check(df, "avg_wait", "up", "MAX_QUEUE↑")]


def test_service_time():
    """Повільніше приготування: довша черга і більше втрат."""
    rows = []
    for f in [0.5, 0.75, 1.0, 1.5, 2.0]:
        menu = [(n, price, t * f, p) for n, price, t, p in params.MENU]
        with patched(MENU=menu):
            m = mean_summary()
        m["значення"] = f
        rows.append(m)
    df = pd.DataFrame(rows).set_index("значення")
    df.to_csv(f"{RESULTS}/stage1_test_service.csv")
    print("\nТест 3: множник часу приготування")
    print(df[["lost_pct", "avg_queue", "avg_wait", "utilization", "profit"]].round(2))
    return df, [check(df, "lost_pct", "up", "час приготування↑"),
                check(df, "avg_wait", "up", "час приготування↑")]


def test_sanity():
    """Крайні випадки і баланс клієнтів."""
    results = []
    low = [(s, e, lam * 0.05) for s, e, lam in params.PERIODS]
    with patched(PERIODS=low):
        m = mean_summary()
    results.append(m.lost_pct < 1)
    print(f"\n{'OK  ' if results[-1] else 'FAIL'} Дуже мало клієнтів: втрати {m.lost_pct:.2f}% (очікуємо ≈ 0)")

    m = mean_summary(max_queue=0)
    summary, _, clients = simulate_day(1)
    results.append(summary["served"] + summary["lost"] == summary["clients"])
    print(f"{'OK  ' if results[-1] else 'FAIL'} Баланс: обслужені + втрачені = усі клієнти")

    results.append(summary["utilization"] <= 1.0)
    print(f"{'OK  ' if results[-1] else 'FAIL'} Завантаженість ≤ 1 ({summary['utilization']:.2f})")
    return results


def plot(df_l, df_q, df_s):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(df_l.index, df_l.lost_pct, marker="o", color="indianred")
    axes[0].set_title("Втрати залежно від множника λ")
    axes[0].set_xlabel("Множник інтенсивності потоку")
    axes[0].set_ylabel("Втрачені клієнти, %")
    axes[1].plot(df_q.index, df_q.lost_pct, marker="o", color="steelblue", label="Втрати, %")
    axes[1].plot(df_q.index, df_q.avg_wait, marker="s", color="seagreen", label="Очікування, хв")
    axes[1].set_title("Вплив обмеження черги")
    axes[1].set_xlabel("MAX_QUEUE")
    axes[1].legend()
    axes[2].plot(df_s.index, df_s.avg_queue, marker="o", color="darkorange")
    axes[2].set_title("Довжина черги залежно від часу приготування")
    axes[2].set_xlabel("Множник часу приготування")
    axes[2].set_ylabel("Середня довжина черги")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/stage1_tests.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df_l, r1 = test_lambda()
    df_q, r2 = test_queue_limit()
    df_s, r3 = test_service_time()
    r4 = test_sanity()
    plot(df_l, df_q, df_s)
    total = r1 + r2 + r3 + r4
    print(f"\nПройдено {sum(total)} з {len(total)} перевірок. Графіки: results/stage1_tests.png")