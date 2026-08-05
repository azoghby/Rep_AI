import csv
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found in the data folder.")

latest_file = csv_files[-1]

times = []
values = []

with open(latest_file, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        time_ms = int(row["time_ms"])
        value_key = "signal_value" if "signal_value" in row else "emg_value"
        signal_value = int(row[value_key])

        times.append(time_ms / 1000)
        values.append(signal_value)

average_value = sum(values) / len(values)
min_value = min(values)
max_value = max(values)
threshold = (min_value + max_value) / 2
high_samples = [value for value in values if value > threshold]
percent_high = len(high_samples) / len(values) * 100

print(f"Analyzing: {latest_file.name}")
print(f"Samples: {len(values)}")
print(f"Average signal: {average_value:.2f}")
print(f"Minimum signal: {min_value}")
print(f"Maximum signal: {max_value}")
print(f"Percent high/flexed: {percent_high:.1f}%")

plt.figure()
plt.plot(times, values)
plt.title(f"RepAI CSV Analysis: {latest_file.name}")
plt.xlabel("Time (seconds)")
plt.ylabel("Signal Value")
plt.grid(True)

output_graph = GRAPHS_DIR / f"{latest_file.stem}_plot.png"
plt.savefig(output_graph)
plt.show()

print(f"Saved graph to: {output_graph}")
