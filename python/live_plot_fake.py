from collections import deque
import time

import matplotlib.pyplot as plt
import serial

from serial_port import find_arduino_port


BAUD_RATE = 115200
WINDOW_SECONDS = 8
MAX_POINTS = 2000
PLOT_UPDATE_EVERY_N_SAMPLES = 10
Y_MIN = 0
Y_MAX = 1023


def parse_serial_line(raw_line):
    time_text, value_text = raw_line.split(",")
    return int(time_text), int(value_text)


def trim_old_points(times, values, latest_time):
    cutoff = latest_time - WINDOW_SECONDS

    while times and times[0] < cutoff:
        times.popleft()
        values.popleft()


def main():
    port = find_arduino_port()
    print(f"Selected port: {port}")

    times = deque(maxlen=MAX_POINTS)
    values = deque(maxlen=MAX_POINTS)
    first_sample_time_ms = None
    sample_count = 0

    with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
        time.sleep(2)
        ser.reset_input_buffer()

        print("Reading live EMG data.")
        print("Press Ctrl+C to stop.")

        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], linewidth=1.5)

        ax.set_title("Live MyoWare EMG Signal")
        ax.set_xlabel("Time since start (seconds)")
        ax.set_ylabel("EMG Value")
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.grid(True, alpha=0.25)

        try:
            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()

                if not raw_line:
                    plt.pause(0.001)
                    continue

                try:
                    time_ms, emg_value = parse_serial_line(raw_line)
                except ValueError:
                    continue

                if first_sample_time_ms is None:
                    first_sample_time_ms = time_ms

                time_seconds = (time_ms - first_sample_time_ms) / 1000
                times.append(time_seconds)
                values.append(emg_value)
                trim_old_points(times, values, time_seconds)

                sample_count += 1

                if sample_count % PLOT_UPDATE_EVERY_N_SAMPLES != 0:
                    continue

                line.set_xdata(times)
                line.set_ydata(values)

                if times:
                    right_edge = max(WINDOW_SECONDS, times[-1])
                    ax.set_xlim(right_edge - WINDOW_SECONDS, right_edge)

                fig.canvas.draw_idle()
                plt.pause(0.001)

        except KeyboardInterrupt:
            print("\nLive plot stopped.")


if __name__ == "__main__":
    main()
