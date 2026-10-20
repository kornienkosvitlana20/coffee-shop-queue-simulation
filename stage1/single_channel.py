"""Імітаційна модель кав'ярні з одним каналом (однією станцією)."""
from collections import deque

import numpy as np
import pandas as pd

import params
from stage1.generation import generate_arrivals

SERVED, LOST = "обслужений", "втрачений"


def period_label(arrival):
    """Назва періоду доби, до якого належить момент прибуття."""
    hour = params.OPEN_HOUR + arrival / 60
    for start, end, _ in params.PERIODS:
        if start <= hour < end:
            return f"{start:02d}-{end:02d}"


class Station:
    """Одна станція + спільна обмежена черга."""

    def __init__(self, max_queue):
        self.max_queue = max_queue
        self.busy_until = 0.0      # коли станція звільниться, хв
        self.queue = deque()
        self.busy_time = 0.0       # робочий час у межах доби, хв
        self.lost = 0
        self.log = []              # лог по клієнтах

    def _start(self, client, start):
        finish = start + client.service_time
        self.busy_until = finish
        day = params.SIM_MINUTES
        self.busy_time += max(0.0, min(finish, day) - min(start, day))
        self.log.append(dict(
            arrival=client.arrival, start=start, finish=finish,
            wait=start - client.arrival, n_drinks=client.n_drinks,
            price=client.price, service_time=client.service_time,
            status=SERVED, period=period_label(client.arrival)))

    def advance(self, t):
        """Станція бере з черги клієнтів, поки встигає до моменту t."""
        while self.queue and self.busy_until <= t:
            self._start(self.queue.popleft(), self.busy_until)

    def arrive(self, client):
        self.advance(client.arrival)
        if self.busy_until <= client.arrival and not self.queue:
            self._start(client, client.arrival)
        elif len(self.queue) < self.max_queue:
            self.queue.append(client)
        else:
            self.lost += 1
            self.log.append(dict(
                arrival=client.arrival, start=np.nan, finish=np.nan,
                wait=np.nan, n_drinks=client.n_drinks, price=client.price,
                service_time=client.service_time, status=LOST,
                period=period_label(client.arrival)))


def simulate_day(seed=None, max_queue=None):
    """Один прогін моделі (1 доба). Повертає (підсумки, лог кроків, лог клієнтів)."""
    rng = np.random.default_rng(seed)
    st = Station(params.MAX_QUEUE if max_queue is None else max_queue)
    steps = []

    for k in range(params.SIM_MINUTES // params.STEP_MIN):
        t0 = k * params.STEP_MIN
        t1 = t0 + params.STEP_MIN
        arrivals = generate_arrivals(rng, t0)
        lost_before = st.lost
        for c in arrivals:
            st.arrive(c)
        st.advance(t1)
        hh = params.OPEN_HOUR + t0 // 60
        steps.append(dict(
            step=k, time=f"{hh:02d}:{t0 % 60:02d}", arrived=len(arrivals),
            lost=st.lost - lost_before, queue_len=len(st.queue),
            busy=st.busy_until > t1))

    st.advance(float("inf"))   # дообслужити тих, хто лишився в черзі після закриття

    steps = pd.DataFrame(steps)
    clients = pd.DataFrame(st.log).sort_values("arrival").reset_index(drop=True)
    served = clients[clients.status == SERVED]

    revenue = served.price.sum()
    cost = params.STATION_COST_PER_HOUR * (params.CLOSE_HOUR - params.OPEN_HOUR)
    summary = dict(
        clients=len(clients),
        served=len(served),
        lost=st.lost,
        lost_pct=100 * st.lost / len(clients),
        revenue=revenue,
        cost=cost,
        profit=revenue - cost,
        avg_wait=served.wait.mean(),
        max_wait=served.wait.max(),
        avg_queue=steps.queue_len.mean(),
        max_queue=steps.queue_len.max(),
        utilization=st.busy_time / params.SIM_MINUTES,
    )
    return summary, steps, clients


if __name__ == "__main__":
    summary, steps, clients = simulate_day(seed=1)
    print("Підсумки доби:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
    print("\nФрагмент логу по клієнтах:")
    print(clients.head(15).round(2).to_string())
    print("\nФрагмент логу по кроках:")
    print(steps.head(15).to_string())