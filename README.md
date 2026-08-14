# RepAI

RepAI is an experimental EMG workout-analysis prototype that records MyoWare-style muscle signals, detects repetitions, and summarizes set-level activation patterns.

Prototype status: this is a student engineering project, not a medical device, clinical tool, or validated strength assessment system. The included replay data is synthetic so reviewers can run the workflow without owning the hardware.

## Key Features

- Streamlit workflow for signal checks, timed recordings, replay-mode demos, and post-recording analysis.
- Wired Arduino/MyoWare acquisition path using serial rows formatted as `time_ms,emg_value`.
- Replay acquisition path for CSV files with `time_ms` plus `emg_value` or `signal_value`.
- Legacy threshold-based repetition detector.
- Experimental hybrid contraction-cycle detector for harder contraction shapes.
- Optional calibration utilities for baseline and max-flex normalization.
- Generated summaries, rep plots, and static HTML reports.

## System Architecture

```text
MyoWare-style EMG sensor
        |
Arduino analog input
        |
Serial stream: time_ms,emg_value
        |
Python acquisition layer
        |
CSV + JSON metadata
        |
Rep detection, calibration, comparison, quality checks
        |
Streamlit app, PNG graphs, text summaries, HTML reports
```

## Hardware Chain

The wired prototype uses a MyoWare-style EMG sensor on the target muscle, an Arduino-compatible board reading the analog signal, and a USB serial connection at `115200` baud. The Arduino reader sketch is in `arduino/myoware_reader/`. A synthetic signal-generator sketch used during development is in `arduino/fake_emg_generator/`.

## Software Pipeline

Recordings are CSV files with `time_ms` and signal values. Optional JSON sidecars store exercise metadata such as exercise name, muscle, side, weight, expected reps, data type, and test type. Analysis scripts smooth the signal, estimate a baseline, detect reps, compute set summaries, and save generated outputs under `graphs/`, `summaries/`, and `reports/`.

## Detectors

The legacy detector in `python/detect_reps.py` uses smoothed-signal threshold crossings, a low-percentile baseline estimate, a start threshold, an end threshold, minimum rep duration, and cooldown timing.

The hybrid detector in the same file is experimental. It starts with broad activity-region detection, generates valley candidates inside sustained regions, evaluates contraction evidence, estimates cadence, performs global sequence selection, and is intended to better handle sustained tension and complex contraction shapes. It is not a replacement for independent validation.

The hybrid detector passed an internal annotated regression corpus during private prototype testing, but that corpus was small, recorded under controlled prototype conditions, and contains interpretive boundaries. This is not an independent accuracy benchmark. Additional participants, exercises, electrode placements, hardware states, and sessions are needed before making stronger accuracy claims.

## Streamlit Workflows

Run the app to use:

- Overview of available recordings, summaries, calibration status, and detector comparison.
- Workout Session with hardware preflight, serial or replay source selection, signal check, timed recording, analysis, and report refresh.
- Structured side-comparison and weight-ladder protocol views that store local JSON manifests in `app/protocol_sessions/`.

## Installation

```bash
git clone https://github.com/azoghby/Rep_AI.git
cd RepAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run With Synthetic Replay Data

Launch the app:

```bash
python -m streamlit run app/streamlit_app.py
```

In the app, open `Workout Session`, confirm the preflight checklist, select `Replay CSV`, and choose:

```text
examples/synthetic/synthetic_bicep_curl_6_reps.csv
```

Then run the signal check or start a short timed recording. Replay mode writes a new local synthetic recording into `data/` and runs the same analysis path as the hardware workflow. No physical EMG sensor is required for this demonstration.

You can also run a command-line smoke test:

```bash
python tests/smoke_public.py
```

## Run With Wired MyoWare Hardware

1. Upload `arduino/myoware_reader/mayoware_reader.ino` to an Arduino-compatible board.
2. Connect the MyoWare-style sensor output to the configured Arduino analog input.
3. Use a battery-powered or properly isolated setup.
4. Launch the app:

```bash
python -m streamlit run app/streamlit_app.py
```

Choose `Serial hardware`, select the detected serial port, run the signal check, then record a timed workout set.

## Repository Structure

```text
app/                  Streamlit UI, protocol helpers, quality checks.
app/protocol_sessions Empty local manifest directory placeholder.
arduino/              Arduino sketches for wired and generated signal input.
docs/                 Additional engineering notes.
examples/synthetic/   Public synthetic replay CSV and metadata.
python/               Acquisition, calibration, analysis, detector, and report scripts.
tests/                Public smoke validation.
data/                 Local runtime recordings, ignored by Git.
graphs/               Local generated plots, ignored by Git.
summaries/            Local generated summaries, ignored by Git.
reports/              Local generated HTML reports, ignored by Git.
```

## Example Output

The synthetic replay smoke test is expected to load the public CSV, read replay samples, and detect six synthetic curl-like contractions with both the legacy and experimental hybrid detectors. Generated graphs and summaries are written locally when analysis scripts or the app are run.

## Electrical Safety

Use battery-powered or properly isolated hardware for body-connected sensor work, follow the sensor manufacturer's documentation, avoid unsafe mains-powered setups, and stop immediately if anything feels painful, hot, unstable, or abnormal. RepAI is an experimental personal prototype, not medical equipment.

## Limitations

- Not clinically validated and not medical-grade.
- Repetition boundaries can be interpretive, especially with sustained tension or partial relaxation.
- Single-sensor side comparisons are sequential and can be affected by electrode repositioning, fatigue, timing, and skin contact.
- Normalized activation depends heavily on calibration quality and placement consistency.
- No definitive muscular-imbalance conclusions should be drawn from the current prototype.
- BLE and multi-sensor collection are not implemented in this public release.

## Roadmap

- Broaden synthetic and publishable validation examples.
- Add more robust replay-first demos and automated tests.
- Improve detector diagnostics and calibration guidance.
- Explore wireless acquisition behind the existing signal-source interface.
- Validate across more participants, exercises, placements, and sessions.

## Author

Created by Alex Zoghby.
