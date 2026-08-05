import serial
import matplotlib.pyplot as plt
from collections import deque

from serial_port import find_arduino_port


PORT = find_arduino_port()
BAUD_RATE = 115200
MAX_POINTS = 500

times = deque(maxlen=MAX_POINTS)
values = deque(maxlen=MAX_POINTS)

ser = serial.Serial(PORT, BAUD_RATE)

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])

ax.set_title("Live EMG Signal")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Sensor Value")

while True:
    raw_line = ser.readline().decode("utf-8").strip()

    try:
        time_text, value_text = raw_line.split(",")
        time_ms = int(time_text)
        emg_value = int(value_text)

        times.append(time_ms / 1000)
        values.append(emg_value)

        line.set_xdata(times)
        line.set_ydata(values)

        ax.relim()
        ax.autoscale_view()

        plt.pause(0.01)

    except ValueError:
        pass
