
from collections import deque

import numpy as np
import pandas as pd

import params
from stage1.generation import generate_arrivals
from stage1.single_channel import SERVED, LOST, period_label

N_STEPS = params.SIM_MINUTES // params.STEP_MIN
INF = float("inf")


def period_bounds():
    """Межі періодів у хвилинах від відкриття."""
    return [((s - params.OPEN_HOUR) * 60, (e - params.OPEN_HOUR) * 60)
            for s, e, _ in params.PERIODS]


def make_day(seed):
    """Усі клієнти доби для даного seed (не залежать від розкладу станцій)."""
    rng = np.random.default_rng(seed)
    clients = []
    for k in range(N_STEPS):
        clients += generate_arrivals(rng, k * params.STEP_MIN)
    return clients


class Cafe:
    def __init__(self, schedule, max_queue=None):
        self.schedule = list(schedule)
        self.max_queue = params.MAX_QUEUE if max_queue is None else max_queue
        self.bounds = period_bounds()
        self.busy_until = [0.0] * params.MAX_STATIONS
        self.busy_time = [0.0] * params.MAX_STATIONS
        self.queue = deque()
        self.lost = 0
        self.log = []

    # --- розклад станцій ---
    def _period(self, t):
        for p, (a, b) in enumerate(self.bounds):
            if a <= t < b:
                return p
        return len(self.bounds) - 1          # після закриття: як в останньому періоді

    def _next_open(self, i, t):
        """Найближчий момент >= t, коли станція i відкрита."""
        p = self._period(t)
        while True:
            if i < self.schedule[p]:
                return t
            if p == len(self.bounds) - 1:
                return INF
            p += 1
            t = self.bounds[p][0]

    def _best_station(self, t):
        """Станція, яка може почати обслуговування найраніше, і цей момент."""
        best_i, best_s = None, INF
        for i in range(params.MAX_STATIONS):
            s = self._next_open(i, max(self.busy_until[i], t))
            if s < best_s:
                best_i, best_s = i, s
        return best_i, best_s

    # --- обслуговування ---
    def _start(self, client, i, start):
        finish = start + client.service_time
        self.busy_until[i] = finish
        day = params.SIM_MINUTES
        self.busy_time[i] += max(0.0, min(finish, day) - min(start, day))
        self.log.append(dict(
            arrival=client.arrival, start=start, finish=finish,
            wait=start - client.arrival, station=i + 1,
            n_drinks=client.n_drinks, price=client.price,
            service_time=client.service_time, status=SERVED,
            period=period_label(client.arrival)))

    def advance(self, t):
        """Клієнти з черги займають станції, що звільнилися до моменту t."""
        while self.queue:
            head = self.queue[0]
            i, s = self._best_station(head.arrival)
            if s > t:
                break
            self.queue.popleft()
            self._start(head, i, s)

    def arrive(self, client):
        self.advance(client.arrival)
        i, s = self._best_station(client.arrival)
        if not self.queue and s <= client.arrival:
            self._start(client, i, client.arrival)
        elif len(self.queue) < self.max_queue:
            self.queue.append(client)
        else:
            self.lost += 1
            self.log.append(dict(
                arrival=client.arrival, start=np.nan, finish=np.nan,
                wait=np.nan, station=0, n_drinks=client.n_drinks,
                price=client.price, service_time=client.service_time,
                status=LOST, period=period_label(client.arrival)))

    def busy_now(self, t):
        return sum(b > t for b in self.busy_until)


def station_cost(schedule):
    """Витрати на станції: вартість години × кількість станцій × тривалість періоду."""
    return sum(params.STATION_COST_PER_HOUR * c * (e - s)
               for c, (s, e, _) in zip(schedule, params.PERIODS))


def simulate_day(schedule, seed=None, max_queue=None, clients=None, detail=False):
    """Один прогін (1 доба). Повертає словник підсумків (і логи, якщо detail)."""
    if clients is None:
        clients = make_day(seed)
    cafe = Cafe(schedule, max_queue)
    steps = []
    idx = 0
    for k in range(N_STEPS):
        t0, t1 = k * params.STEP_MIN, (k + 1) * params.STEP_MIN
        lost_before, n_arr = cafe.lost, 0
        while idx < len(clients) and clients[idx].arrival < t1:
            cafe.arrive(clients[idx])
            idx += 1
            n_arr += 1
        cafe.advance(t1)
        if detail:
            hh = params.OPEN_HOUR + t0 // 60
            steps.append(dict(
                step=k, time=f"{hh:02d}:{t0 % 60:02d}", arrived=n_arr,
                lost=cafe.lost - lost_before, queue_len=len(cafe.queue),
                busy_stations=cafe.busy_now(t1),
                open_stations=schedule[cafe._period(t0)]))
    cafe.advance(INF)

    log = pd.DataFrame(cafe.log)
    served = log[log.status == SERVED]
    revenue = served.price.sum()
    cost = station_cost(schedule)
    open_minutes = sum(c * (e - s) * 60 for c, (s, e, _) in zip(schedule, params.PERIODS))
    summary = dict(
        clients=len(log), served=len(served), lost=cafe.lost,
        lost_pct=100 * cafe.lost / len(log),
        revenue=revenue, cost=cost, profit=revenue - cost,
        avg_wait=served.wait.mean(), max_wait=served.wait.max(),
        utilization=sum(cafe.busy_time) / open_minutes,
    )
    if not detail:
        return summary
    steps = pd.DataFrame(steps)
    summary["avg_queue"] = steps.queue_len.mean()
    summary["max_queue"] = steps.queue_len.max()
    return summary, steps, log.sort_values("arrival").reset_index(drop=True)


def run_many(schedule, n=params.N_RUNS, max_queue=None, days=None):
    """n незалежних прогонів. days — заздалегідь згенеровані доби (спільні випадкові числа)."""
    rows = []
    for seed in range(n):
        cl = days[seed] if days is not None else make_day(seed)
        rows.append(simulate_day(schedule, clients=cl, max_queue=max_queue))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary, steps, log = simulate_day([3, 2, 2, 2, 1], seed=1, detail=True)
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}")
    print(log.head(12).round(2).to_string())
    print(steps.iloc[:12].to_string())