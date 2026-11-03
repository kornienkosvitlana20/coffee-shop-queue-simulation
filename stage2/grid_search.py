import itertools
from multiprocessing import Pool

import pandas as pd

import params
from stage2.multi_channel import make_day, run_many

N = params.N_RUNS
DAYS = None


def init():
    global DAYS
    DAYS = [make_day(s) for s in range(N)]


def evaluate(schedule):
    m = run_many(schedule, N, days=DAYS).mean()
    row = {f"c{i + 1}": c for i, c in enumerate(schedule)}
    row.update(m.to_dict())
    return row


if __name__ == "__main__":
    schedules = list(itertools.product(range(1, params.MAX_STATIONS + 1),
                                       repeat=len(params.PERIODS)))
    with Pool(initializer=init) as pool:
        rows = pool.map(evaluate, schedules, chunksize=20)
    df = pd.DataFrame(rows)
    df.to_csv("results/stage2_grid.csv", index=False)
    print("Готово:", len(df), "розкладів")