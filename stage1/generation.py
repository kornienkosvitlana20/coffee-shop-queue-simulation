"""Генерація клієнтів і замовлень для імітаційної моделі кав'ярні."""
from dataclasses import dataclass

import numpy as np

import params


@dataclass
class Client:
    arrival: float        # момент прибуття, хв від відкриття (07:00 = 0)
    n_drinks: int         # кількість напоїв у замовленні
    drinks: list          # назви напоїв
    price: float          # сума замовлення, грн
    service_time: float   # час приготування всього замовлення, хв


def lambda_per_hour(t_min):
    """Інтенсивність потоку (клієнтів/год) у момент t_min (хв від відкриття)."""
    hour = params.OPEN_HOUR + t_min / 60
    for start, end, lam in params.PERIODS:
        if start <= hour < end:
            return lam
    raise ValueError(f"Час {hour:.2f} год поза робочим днем")


def generate_order(rng, arrival):
    """Створює клієнта із замовленням.

    Кількість напоїв: геометричний розподіл (мінімум 1).
    Вид кожного напою: за ймовірностями з меню.
    """
    n_drinks = int(rng.geometric(params.P_GEOM))
    probs = [item[3] for item in params.MENU]
    idx = rng.choice(len(params.MENU), size=n_drinks, p=probs)
    return Client(
        arrival=float(arrival),
        n_drinks=n_drinks,
        drinks=[params.MENU[i][0] for i in idx],
        price=sum(params.MENU[i][1] for i in idx),
        service_time=sum(params.MENU[i][2] for i in idx),
    )


def generate_arrivals(rng, step_start):
    """Клієнти, що прийшли за один крок (step_start ... step_start + STEP_MIN).

    Кількість: розподіл Пуассона з λ·(крок/60).
    Момент прибуття кожного: рівномірно в межах кроку, за зростанням.
    """
    lam = lambda_per_hour(step_start)
    n = rng.poisson(lam * params.STEP_MIN / 60)
    times = np.sort(rng.uniform(step_start, step_start + params.STEP_MIN, n))
    return [generate_order(rng, t) for t in times]


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    clients = []
    for step in range(params.SIM_MINUTES // params.STEP_MIN):
        clients += generate_arrivals(rng, step * params.STEP_MIN)

    print("Клієнтів за добу:", len(clients))
    print("Середня к-сть напоїв:", round(np.mean([c.n_drinks for c in clients]), 2))
    print("Середня сума замовлення:", round(np.mean([c.price for c in clients]), 1))
    print("Середній час обслуговування, хв:",
          round(np.mean([c.service_time for c in clients]), 2))
    print("Перші 3 клієнти:")
    for c in clients[:3]:
        print(c)