import time

import pandas as pd

from vangja_simple.components import FourierSeasonality, LinearTrend

# Fetch data
data = pd.read_csv(
    "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
)

for k in range(5):
    model = LinearTrend() + FourierSeasonality(365.25, 10) + FourierSeasonality(7, 3)
    start_time = time.time()
    # .fit call find_MAP
    model.fit(data, progressbar=False)
    print(f"MAP {k}: {time.time() - start_time}s")
