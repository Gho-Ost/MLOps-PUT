import bentoml
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":

    DATA_PATH = Path("../project1/data/TS0277.csv.gz")

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ['time', 'value']
    df['time'] = pd.to_datetime(df['time'])

    df = df.drop_duplicates(subset='time', keep='last')

    start_time = pd.Timestamp('2001-01-01 00:00:00')
    end_time = df['time'].max()

    expected_index = pd.date_range(start=start_time, end=end_time, freq='15min')
    actual_times = pd.DatetimeIndex(df['time'])
    missing_times = expected_index.difference(actual_times)
    df = df.set_index("time").reindex(expected_index)

    df.index.name = "time"
    df['value'] = df['value'].ffill()

    avg_consumption = df['value'].mean()
    std_consumption = df['value'].std()
    df['scaled_value'] = (df['value'] - avg_consumption) / std_consumption

    df = df.sort_index()

    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        data = df["scaled_value"].values[:96*7].tolist()
        result = client.day_forecast(
            data = data
        )
        result = np.array(eval(result)) # Extremely safe 
        print(f"Result: {result.shape}")

        result = result[0].reshape(-1).tolist()
        
        split = len(data)

        y = data
        y.extend(result)

        x = range(len(y))
        plt.plot(x[:split], y[:split], color='blue', label="data")
        plt.plot(x[split:], y[split:], color='red', label="pred")
        plt.legend()
        plt.show()