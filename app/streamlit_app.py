import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from protocols import (
    SIDE_COMPARISON_SLOTS,
    load_sessions,
    new_side_comparison_session,
    new_weight_ladder_session,
    save_session,
)
from quality_checks import number_from_text, percent_difference, recording_quality
from summary_reader import (
    appears_derived_or_trimmed,
    appears_diagnostic,
    calibration_recordings,
    comparison_mentions_file,
    current_graph_path_for_csv,
    current_summary_path_for_csv,
    graph_path_for_csv,
    latest_comparison,
    load_calibration_from_csv_name,
    load_current_calibration,
    output_is_current,
    recording_summary,
    summary_path_for_csv,
    valid_workout_recordings,
)


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BASE_DIR / "python"
DATA_DIR = BASE_DIR / "data"
SYNTHETIC_EXAMPLES_DIR = BASE_DIR / "examples" / "synthetic"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from acquisition import ReplaySignalSource, SerialSignalSource, available_serial_ports  # noqa: E402
from compare_latest_sets import main as refresh_latest_comparison  # noqa: E402
from detect_reps import (  # noqa: E402
    BASELINE_PERCENTILE,
    END_THRESHOLD_FRACTION,
    SMOOTHING_WINDOW,
    START_THRESHOLD_FRACTION,
    activation_change_phrase,
    analyze_csv_file,
    detect_reps,
    detect_reps_hybrid,
    low_percentile_average,
    moving_average,
    read_signal,
)
from generate_report import main as regenerate_reports  # noqa: E402
from recording_metadata import load_metadata, save_metadata  # noqa: E402


PREFLIGHT_ITEMS = [
    "MacBook unplugged from wall power",
    "No powered dock, monitor, or powered USB hub",
    "Link Shield powered on",
    "Link Shield in ENV mode",
    "TRS cable fully connected",
    "Electrodes attached correctly",
]


def display_value(value, fallback="Not available"):
    return value if value not in ("", None) else fallback


def metric_row(items):
    columns = st.columns(len(items))

    for column, (label, value) in zip(columns, items):
        column.metric(label, display_value(value))


def csv_options(files):
    return [""] + [file.name for file in files]


def selected_csv(csv_name):
    return BASE_DIR / "data" / csv_name if csv_name else None


def replay_recordings():
    recordings = valid_workout_recordings()

    if SYNTHETIC_EXAMPLES_DIR.exists():
        recordings.extend(sorted(SYNTHETIC_EXAMPLES_DIR.glob("*.csv")))

    return recordings


def detector_comparison_recordings():
    recordings = valid_workout_recordings()
    diagnostic_dir = DATA_DIR / "diagnostic_recordings"

    if diagnostic_dir.exists():
        recordings.extend(sorted(diagnostic_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True))

    return recordings


def graph_path_for_any_csv(csv_file, graph_label="reps"):
    relative_parent = csv_file.parent.relative_to(DATA_DIR)
    graph_dir = BASE_DIR / "graphs" / relative_parent
    return graph_dir / f"{csv_file.stem}_{graph_label}.png"


def expected_reps_differ(expected_reps, detected_reps):
    if expected_reps in ("", None) or detected_reps in ("", None):
        return False

    return str(expected_reps).strip() != str(detected_reps).strip()


def rep_intervals(reps):
    return [
        f"{index}: {rep['start_time']:.2f}s-{rep['end_time']:.2f}s"
        for index, rep in enumerate(reps, start=1)
    ]


def detected_reps_from_result(result):
    if not result:
        return ""

    return result["summary"]["total_reps"]


def detector_thresholds(csv_file):
    times, values = read_signal(csv_file)
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    baseline = low_percentile_average(smoothed_values, BASELINE_PERCENTILE)
    max_signal = max(smoothed_values)
    signal_range = max_signal - baseline
    start_threshold = baseline + START_THRESHOLD_FRACTION * signal_range
    end_threshold = baseline + END_THRESHOLD_FRACTION * signal_range
    return times, values, smoothed_values, start_threshold, end_threshold


def detector_result_without_writing(csv_file, method):
    times, values, smoothed_values, start_threshold, end_threshold = detector_thresholds(csv_file)

    if method == "legacy":
        reps = detect_reps(
            times,
            values,
            smoothed_values,
            start_threshold,
            end_threshold,
        )
        diagnostics = None
        graph_file = graph_path_for_any_csv(csv_file)
    elif method == "hybrid":
        reps, diagnostics = detect_reps_hybrid(
            times,
            values,
            smoothed_values,
            start_threshold,
            end_threshold,
        )
        graph_file = graph_path_for_any_csv(csv_file, graph_label="hybrid_reps")
    else:
        raise ValueError(f"Unknown detector method: {method}")

    return {
        "summary": {"total_reps": len(reps)},
        "reps": reps,
        "graph_file": graph_file,
        "diagnostics": diagnostics,
        "method": method,
    }


def show_rep_intervals(label, result):
    st.markdown(f"**{label} rep intervals**")

    if not result:
        st.write("Not available")
        return

    intervals = rep_intervals(result.get("reps", []))

    if intervals:
        st.write(", ".join(intervals))
    else:
        st.write("Intervals are not available from the saved legacy summary.")


def show_hybrid_split_valleys(hybrid_result):
    diagnostics = (hybrid_result or {}).get("diagnostics") or {}
    accepted_valleys = diagnostics.get("accepted_valleys", [])

    st.markdown("**Accepted hybrid split valleys**")

    if not accepted_valleys:
        st.write("None")
        return

    rows = [
        {
            "Time": f"{candidate['time']:.2f}s",
            "Value": f"{candidate['value']:.1f}",
            "Drop": f"{candidate['adjacent_drop']:.1f}",
            "Valley duration": f"{candidate['valley_duration']:.2f}s",
            "Center gap": f"{candidate['center_gap']:.2f}s",
        }
        for candidate in accepted_valleys
    ]
    st.table(rows)


def show_detector_graph(label, result):
    graph_file = (result or {}).get("graph_file")

    if not graph_file:
        st.write(f"{label} graph is not available.")
        return

    if graph_file.exists():
        st.image(str(graph_file), caption=graph_file.name, use_container_width=True)
    else:
        st.warning(f"{label} graph was expected at {graph_file.name}, but it is missing.")


def show_detector_comparison(expected_reps, legacy_result, hybrid_result):
    st.subheader("Experimental detector comparison")
    legacy_reps = detected_reps_from_result(legacy_result)
    hybrid_reps = detected_reps_from_result(hybrid_result)
    methods_agree = str(legacy_reps) == str(hybrid_reps)

    metric_row([
        ("Expected reps", expected_reps),
        ("Legacy detected reps", legacy_reps),
        ("Hybrid detected reps", hybrid_reps),
        ("Methods agree", "Yes" if methods_agree else "No"),
    ])

    if not methods_agree:
        st.warning(
            "Legacy and hybrid detectors disagree. The hybrid result is experimental "
            "and is shown for controlled validation only; RepAI is not automatically "
            "choosing whichever value matches expected reps."
        )

    show_rep_intervals("Legacy", legacy_result)
    show_rep_intervals("Hybrid", hybrid_result)
    show_hybrid_split_valleys(hybrid_result)

    graph_columns = st.columns(2)
    with graph_columns[0]:
        st.markdown("**Legacy graph**")
        show_detector_graph("Legacy", legacy_result)
    with graph_columns[1]:
        st.markdown("**Hybrid graph**")
        show_detector_graph("Hybrid", hybrid_result)


def safe_name(value, fallback):
    cleaned = "".join(
        character.lower() if character.isalnum() else "_"
        for character in value.strip()
    ).strip("_")
    return cleaned or fallback


def unique_recording_path(exercise_name, timestamp):
    base_name = f"{safe_name(exercise_name, 'workout_set')}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    output_file = DATA_DIR / f"{base_name}.csv"
    suffix = 2

    while output_file.exists():
        output_file = DATA_DIR / f"{base_name}_{suffix}.csv"
        suffix += 1

    return output_file


def source_from_selection(source_type, port=None, replay_csv=None, replay_realtime=False):
    if source_type == "Replay CSV":
        return ReplaySignalSource(replay_csv, realtime=replay_realtime)

    return SerialSignalSource(port)


def replay_source_stats(csv_file):
    times, _ = read_signal(csv_file)
    source_duration = (times[-1] - times[0]) if len(times) >= 2 else 0
    return {
        "expected_samples": len(times),
        "source_duration": source_duration,
    }


def run_replay_workflow(replay_csv, replay_realtime=False, progress_callback=None, status_callback=None):
    replay_csv = Path(replay_csv)
    stats = replay_source_stats(replay_csv)
    source = ReplaySignalSource(replay_csv, realtime=replay_realtime)
    readings = []
    malformed_reads = 0

    if status_callback:
        status_callback("Replay starting...")

    source.connect()
    try:
        while True:
            reading = source.read()
            if reading is None:
                break

            readings.append(reading)

            if progress_callback:
                progress_callback(
                    min(1.0, len(readings) / max(1, stats["expected_samples"]))
                )

            if replay_realtime and status_callback:
                status_callback(f"Playback active: {len(readings)} samples read")
    finally:
        source.disconnect()

    if progress_callback:
        progress_callback(1.0)

    legacy_result = analyze_csv_file(replay_csv, show_plot=False, method="legacy")
    hybrid_result = analyze_csv_file(replay_csv, show_plot=False, method="hybrid")

    return {
        "source_csv": replay_csv,
        "samples_read": len(readings),
        "malformed_samples": malformed_reads,
        "source_duration": stats["source_duration"],
        "legacy_result": legacy_result,
        "hybrid_result": hybrid_result,
    }


def collect_readings(source, duration_seconds):
    readings = []
    malformed_reads = 0
    deadline = time.monotonic() + duration_seconds
    source.connect()

    try:
        while time.monotonic() < deadline:
            reading = source.read()
            if reading is None:
                if isinstance(source, ReplaySignalSource):
                    break
                malformed_reads += 1
                continue
            readings.append(reading)
    finally:
        source.disconnect()

    return readings, malformed_reads


def signal_check_stats(readings):
    values = [reading.signal_value for reading in readings]
    timestamps = [reading.host_time_ms for reading in readings]
    duration_ms = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0
    sample_rate = len(readings) / (duration_ms / 1000) if duration_ms > 0 else 0
    gaps = [
        timestamps[index] - timestamps[index - 1]
        for index in range(1, len(timestamps))
    ]
    signal_min = min(values) if values else 0
    signal_max = max(values) if values else 0
    signal_range = signal_max - signal_min
    average_signal = sum(values) / len(values) if values else 0
    pinned_window = max(2, signal_range * 0.01)
    pinned_fraction = (
        len([value for value in values if signal_max - value <= pinned_window]) / len(values)
        if values
        else 0
    )

    return {
        "samples": len(readings),
        "sample_rate": sample_rate,
        "min": signal_min,
        "max": signal_max,
        "average": average_signal,
        "range": signal_range,
        "largest_gap_ms": max(gaps) if gaps else 0,
        "average_gap_ms": sum(gaps) / len(gaps) if gaps else 0,
        "pinned_fraction": pinned_fraction,
    }


def signal_quality_warnings(stats):
    warnings = []

    if stats["samples"] < 20:
        warnings.append("Too few valid samples were collected for a useful signal check.")

    if stats["range"] <= 2:
        warnings.append("The signal is nearly flat. Check power, mode, cables, and electrode contact.")
    elif stats["range"] < 20:
        warnings.append("The usable signal range is very small for a workout recording.")

    if stats["pinned_fraction"] >= 0.25:
        warnings.append("Many samples are pinned near the observed maximum.")

    if stats["largest_gap_ms"] > 1000:
        warnings.append("Large sample gaps were observed during collection.")
    elif stats["average_gap_ms"] > 100 and stats["samples"] > 1:
        warnings.append("Sample timing appears irregular or slower than expected.")

    return warnings


def plot_signal_check(readings):
    if not readings:
        return None

    start_time_ms = readings[0].host_time_ms
    times = [(reading.host_time_ms - start_time_ms) / 1000 for reading in readings]
    values = [reading.signal_value for reading in readings]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, linewidth=1.5)
    ax.set_title("Signal Check")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Signal Value")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def write_workout_recording(source, output_file, metadata, duration_seconds, progress_bar):
    readings = []
    malformed_reads = 0
    deadline = time.monotonic() + duration_seconds
    source.connect()

    try:
        metadata_file = save_metadata(output_file, metadata)

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_ms", "emg_value"])
            first_host_time_ms = None

            while time.monotonic() < deadline:
                progress_bar.progress(
                    min(1.0, 1 - ((deadline - time.monotonic()) / duration_seconds))
                )
                reading = source.read()

                if reading is None:
                    if isinstance(source, ReplaySignalSource):
                        break
                    malformed_reads += 1
                    continue

                if first_host_time_ms is None:
                    first_host_time_ms = reading.host_time_ms

                elapsed_ms = reading.host_time_ms - first_host_time_ms
                writer.writerow([elapsed_ms, reading.signal_value])
                readings.append(reading)
    finally:
        source.disconnect()
        progress_bar.progress(1.0)

    return {
        "csv_file": output_file,
        "metadata_file": metadata_file,
        "readings": readings,
        "malformed_reads": malformed_reads,
    }


def run_post_recording_pipeline(csv_file):
    legacy_result = analyze_csv_file(csv_file, show_plot=False, method="legacy")
    comparison_error = None

    try:
        refresh_latest_comparison()
    except Exception as error:  # noqa: BLE001
        comparison_error = str(error)

    regenerate_reports()
    hybrid_result = analyze_csv_file(csv_file, show_plot=False, method="hybrid")
    return legacy_result, hybrid_result, comparison_error


def show_calibration():
    calibration, source_metadata = load_current_calibration()

    st.subheader("Current Calibration")

    if calibration is None:
        st.warning("Calibration is missing. Normalized activation may be unavailable.")
        return

    if not calibration.get("usable"):
        st.warning("Calibration is present but marked unusable.")

    metric_row([
        ("Source", calibration.get("source_csv", "")),
        ("Muscle", source_metadata.get("muscle", "")),
        ("Side", source_metadata.get("side", "")),
    ])
    metric_row([
        ("Baseline", f"{calibration.get('baseline', 0):.1f}"),
        ("Maximum flex", f"{calibration.get('max_flex', 0):.1f}"),
        ("Signal range", f"{calibration.get('signal_range', 0):.1f}"),
        ("Usable", "Yes" if calibration.get("usable") else "No"),
    ])


def show_recording(recordings):
    st.subheader("Workout Recording")

    if not recordings:
        st.warning("No valid real workout-set recordings were found in data/.")
        return

    selected_name = st.selectbox(
        "Recording",
        [recording.name for recording in recordings],
        key="overview_recording_select",
    )
    selected_file = next(recording for recording in recordings if recording.name == selected_name)
    metadata = load_metadata(selected_file)
    summary_file = summary_path_for_csv(selected_file)
    graph_file = graph_path_for_csv(selected_file)
    current_summary_file = current_summary_path_for_csv(selected_file)
    current_graph_file = current_graph_path_for_csv(selected_file)
    summary = recording_summary(selected_file) if current_summary_file else {
        "detected_reps": "",
        "average_normalized_rep_activation": "",
        "peak_normalized_activation": "",
        "activation_trend": "",
        "user_insights": [],
    }

    if summary_file.exists() and current_summary_file is None:
        st.warning("The saved summary is older than this recording, so it is not shown as current.")
    elif not summary_file.exists():
        st.warning("Rep summary is missing for this recording.")

    if graph_file.exists() and current_graph_file is None:
        st.warning("The rep graph is older than this recording, so it is not shown as current.")
    elif not graph_file.exists():
        st.warning("Rep-detection graph is missing for this recording.")

    if expected_reps_differ(metadata.get("expected_reps", ""), summary["detected_reps"]):
        st.warning("Expected reps and detected reps differ.")

    if appears_derived_or_trimmed(metadata):
        st.warning("This recording appears to be derived or trimmed.")

    if appears_diagnostic(selected_file, metadata):
        st.warning("This recording appears diagnostic.")

    metric_row([
        ("Filename", selected_file.name),
        ("Exercise", metadata.get("exercise_name", "")),
        ("Muscle / side", f"{metadata.get('muscle', '')} / {metadata.get('side', '')}"),
    ])
    metric_row([
        ("Weight", metadata.get("weight", "")),
        ("Expected reps", metadata.get("expected_reps", "")),
        ("Detected reps", summary["detected_reps"]),
    ])
    metric_row([
        ("Avg normalized rep activation", summary["average_normalized_rep_activation"]),
        ("Peak normalized activation", summary["peak_normalized_activation"]),
        ("Activation trend", summary["activation_trend"]),
    ])

    notes = metadata.get("notes", "")
    if notes:
        st.markdown("**Notes**")
        st.write(notes)

    if current_graph_file:
        st.image(str(current_graph_file), caption=current_graph_file.name, use_container_width=True)

    return selected_file


def show_existing_detector_comparison(selected_file):
    recordings = detector_comparison_recordings()

    if not recordings:
        return

    with st.expander("Experimental detector comparison for selected recording"):
        st.caption(
            "Legacy remains the default detector. Running this comparison may create "
            "or refresh the separate hybrid graph and summary for this recording."
        )
        labels = [
            str(recording.relative_to(DATA_DIR))
            for recording in recordings
        ]
        default_label = (
            str(selected_file.relative_to(DATA_DIR))
            if selected_file is not None and selected_file in recordings
            else labels[0]
        )
        selected_label = st.selectbox(
            "Detector comparison recording",
            labels,
            index=labels.index(default_label),
            key="overview_detector_comparison_recording",
        )
        comparison_file = recordings[labels.index(selected_label)]
        metadata = load_metadata(comparison_file)

        if not st.button(
            "Run experimental detector comparison",
            key=f"overview_compare_{comparison_file.stem}",
        ):
            return

        legacy_result = detector_result_without_writing(comparison_file, "legacy")

        with st.spinner("Running experimental hybrid detector..."):
            try:
                hybrid_result = analyze_csv_file(
                    comparison_file,
                    show_plot=False,
                    method="hybrid",
                )
            except Exception as error:  # noqa: BLE001
                st.error(f"Hybrid detector comparison failed: {error}")
                return

        show_detector_comparison(
            metadata.get("expected_reps", ""),
            legacy_result,
            hybrid_result,
        )


def show_latest_comparison(current_csv=None):
    st.subheader("Latest Comparison")
    comparison = latest_comparison()
    comparison_file = BASE_DIR / "summaries" / "latest_set_comparison.txt"
    comparison_graph = BASE_DIR / "graphs" / "latest_set_comparison.png"

    if comparison is None:
        st.warning("Latest comparison summary is missing.")
        return

    if current_csv is not None and not comparison_mentions_file(comparison, current_csv):
        st.warning("Latest comparison does not include the current recording, so it may be stale.")
        return

    if current_csv is not None and not output_is_current(current_csv, comparison_file):
        st.warning("Latest comparison summary is older than the current recording.")
        return

    if current_csv is not None and comparison_graph.exists() and not output_is_current(current_csv, comparison_graph):
        st.warning("Latest comparison graph is older than the current recording.")

    rows = [
        {
            "Set": "Set 1",
            "Filename": comparison["set_1"]["filename"],
            "Reps": comparison["set_1"]["rep_count"],
            "Average normalized activation": comparison["set_1"]["average_normalized_activation"],
            "Peak normalized activation": comparison["set_1"]["peak_normalized_activation"],
        },
        {
            "Set": "Set 2",
            "Filename": comparison["set_2"]["filename"],
            "Reps": comparison["set_2"]["rep_count"],
            "Average normalized activation": comparison["set_2"]["average_normalized_activation"],
            "Peak normalized activation": comparison["set_2"]["peak_normalized_activation"],
        },
    ]
    st.table(rows)

    trend = comparison["activation_trend_comparison"]
    if trend:
        st.markdown("**Activation trend comparison**")
        st.write(trend)

    if comparison["user_insights"]:
        st.markdown("**User insights**")
        for insight in comparison["user_insights"]:
            st.write(f"- {insight}")


def show_preflight():
    st.subheader("Hardware Preflight")
    for item in PREFLIGHT_ITEMS:
        st.checkbox(item, key=f"preflight_{item}")

    return st.checkbox(
        "I confirmed the wired setup is ready for signal collection.",
        key="preflight_confirmed",
    )


def show_source_controls(collection_enabled):
    st.subheader("Connection")
    source_type = st.radio(
        "Signal source",
        ["Serial hardware", "Replay CSV"],
        horizontal=True,
        key="workout_signal_source",
    )
    port = None
    replay_csv = None
    replay_realtime = False

    if source_type == "Serial hardware":
        ports = available_serial_ports()

        if not ports:
            st.warning("No serial ports detected.")
        else:
            likely_ports = [port_info for port_info in ports if port_info["likely_arduino"]]
            options = [f"{port_info['device']} - {port_info['description']}" for port_info in ports]
            default_index = 0

            if len(likely_ports) == 1:
                likely_device = likely_ports[0]["device"]
                default_index = next(
                    index
                    for index, port_info in enumerate(ports)
                    if port_info["device"] == likely_device
                )
            elif len(likely_ports) > 1:
                options.insert(0, "Choose a port")

            selected = st.selectbox(
                "Detected serial ports",
                options,
                index=default_index,
                key="workout_serial_port",
            )

            if selected != "Choose a port":
                port = selected.split(" - ", 1)[0]

            if len(likely_ports) > 1 and port is None:
                st.warning("Multiple likely devices were found. Select the port manually.")

        if st.button(
            "Check Connection",
            disabled=not collection_enabled or port is None,
            key="workout_check_connection",
        ):
            try:
                source = SerialSignalSource(port)
                source.connect()
                source.disconnect()
                st.session_state.connection_status = f"Connected to {port}, then closed safely."
            except Exception as error:  # noqa: BLE001
                st.session_state.connection_status = f"Error: {error}"

    else:
        st.subheader("Replay a saved recording")
        replay_files = replay_recordings()
        if replay_files:
            selected_name = st.selectbox(
                "Saved recording",
                [str(file.relative_to(BASE_DIR)) for file in replay_files],
                key="workout_replay_recording",
            )
            replay_csv = next(
                file
                for file in replay_files
                if str(file.relative_to(BASE_DIR)) == selected_name
            )
            replay_realtime = st.checkbox(
                "Replay with original timing",
                key="workout_replay_realtime",
            )
        else:
            st.warning("No valid workout CSVs are available for replay.")

    status = st.session_state.get("connection_status", "Disconnected")
    if status.startswith("Error:"):
        st.error(status)
    else:
        st.info(status)

    return source_type, port, replay_csv, replay_realtime


def show_replay_result(result):
    st.success("Replay complete")
    metric_row([
        ("Source filename", result["source_csv"].name),
        ("Samples read", result["samples_read"]),
        ("Malformed samples skipped", result["malformed_samples"]),
    ])
    metric_row([
        ("Source duration", f"{result['source_duration']:.2f}s"),
        ("Legacy detected reps", result["legacy_result"]["summary"]["total_reps"]),
        ("Hybrid detected reps", result["hybrid_result"]["summary"]["total_reps"]),
    ])

    st.markdown("**Generated summaries**")
    st.write(str(result["legacy_result"]["summary_file"].relative_to(BASE_DIR)))
    st.write(str(result["hybrid_result"]["summary_file"].relative_to(BASE_DIR)))

    graph_columns = st.columns(2)
    with graph_columns[0]:
        st.markdown("**Legacy graph**")
        st.image(
            str(result["legacy_result"]["graph_file"]),
            caption=result["legacy_result"]["graph_file"].name,
            use_container_width=True,
        )
    with graph_columns[1]:
        st.markdown("**Hybrid graph**")
        st.image(
            str(result["hybrid_result"]["graph_file"]),
            caption=result["hybrid_result"]["graph_file"].name,
            use_container_width=True,
        )


def show_replay_workflow(replay_csv, replay_realtime):
    replay_disabled = replay_csv is None

    if st.button(
        "Start replay",
        disabled=replay_disabled,
        key="workout_start_replay",
        type="primary",
    ):
        progress_bar = st.progress(0.0)
        status = st.empty()
        status.info("Replay starting...")

        try:
            with st.spinner("Running replay..."):
                result = run_replay_workflow(
                    replay_csv,
                    replay_realtime=replay_realtime,
                    progress_callback=progress_bar.progress,
                    status_callback=status.info,
                )
        except Exception as error:  # noqa: BLE001
            st.exception(error)
            return

        st.session_state.replay_result = result

    replay_result = st.session_state.get("replay_result")
    if replay_result:
        show_replay_result(replay_result)


def show_signal_check(collection_enabled, source_type, port, replay_csv, replay_realtime):
    st.subheader("Ten-second Signal Check")

    missing_source = source_type == "Serial hardware" and port is None
    missing_source = missing_source or (source_type == "Replay CSV" and replay_csv is None)

    if st.button(
        "Run Signal Check",
        disabled=not collection_enabled or missing_source,
        key="workout_run_signal_check",
    ):
        source = source_from_selection(source_type, port, replay_csv, replay_realtime)

        with st.spinner("Collecting signal..."):
            try:
                readings, malformed_reads = collect_readings(source, 10)
            except Exception as error:  # noqa: BLE001
                st.error(f"Signal check failed: {error}")
                return

        stats = signal_check_stats(readings)
        st.session_state.signal_check = {
            "readings": readings,
            "stats": stats,
            "malformed_reads": malformed_reads,
            "warnings": signal_quality_warnings(stats),
        }

    signal_check = st.session_state.get("signal_check")
    if not signal_check:
        return

    stats = signal_check["stats"]
    metric_row([
        ("Valid samples", stats["samples"]),
        ("Approx. sample rate", f"{stats['sample_rate']:.1f} Hz"),
        ("Min signal", f"{stats['min']:.1f}"),
    ])
    metric_row([
        ("Max signal", f"{stats['max']:.1f}"),
        ("Average signal", f"{stats['average']:.1f}"),
        ("Signal range", f"{stats['range']:.1f}"),
    ])

    if signal_check["malformed_reads"]:
        st.warning(f"Skipped {signal_check['malformed_reads']} malformed or empty readings.")

    for warning in signal_check["warnings"]:
        st.warning(warning)

    figure = plot_signal_check(signal_check["readings"])
    if figure is not None:
        st.pyplot(figure)
        plt.close(figure)


def show_workout_recording(collection_enabled, source_type, port, replay_csv, replay_realtime):
    st.subheader("Timed Workout Recording")
    missing_source = source_type == "Serial hardware" and port is None
    missing_source = missing_source or (source_type == "Replay CSV" and replay_csv is None)

    with st.form("workout_recording_form"):
        exercise = st.text_input("Exercise", value="synthetic_bicep_curl", key="workout_exercise")
        muscle = st.text_input("Muscle", value="bicep", key="workout_muscle")
        side = st.text_input("Side", value="right", key="workout_side")
        weight = st.text_input("Weight", value="3", key="workout_weight")
        expected_reps = st.text_input("Expected reps", value="6", key="workout_expected_reps")
        notes = st.text_area("Notes", value="", key="workout_notes")
        duration = st.number_input(
            "Recording duration (seconds)",
            min_value=5,
            max_value=180,
            value=45,
            step=5,
            key="workout_duration",
        )
        submitted = st.form_submit_button(
            "Start Recording",
            disabled=not collection_enabled or missing_source,
        )

    if not submitted:
        return

    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now()
    output_file = unique_recording_path(exercise, timestamp)
    metadata = {
        "exercise_name": exercise.strip() or "workout_set",
        "muscle": muscle.strip(),
        "side": side.strip(),
        "weight": weight.strip(),
        "expected_reps": expected_reps.strip(),
        "data_type": "synthetic" if source_type == "Replay CSV" else "real",
        "test_type": "workout_set",
        "notes": notes.strip(),
        "csv_filename": output_file.name,
        "timestamp": timestamp.isoformat(timespec="seconds"),
    }
    source = source_from_selection(source_type, port, replay_csv, replay_realtime)
    progress_bar = st.progress(0.0)

    with st.spinner("Recording workout set..."):
        try:
            recording = write_workout_recording(
                source,
                output_file,
                metadata,
                duration,
                progress_bar,
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Recording failed: {error}")
            return

    st.success(f"Saved recording: {recording['csv_file'].resolve()}")
    st.write(f"Saved metadata: {recording['metadata_file'].resolve()}")

    if recording["malformed_reads"]:
        st.warning(f"Skipped {recording['malformed_reads']} malformed or empty readings.")

    if not recording["readings"]:
        st.error("No valid readings were saved, so analysis was skipped.")
        return

    with st.spinner("Analyzing recording and refreshing reports..."):
        try:
            legacy_result, hybrid_result, comparison_error = run_post_recording_pipeline(output_file)
        except Exception as error:  # noqa: BLE001
            st.error(f"Post-recording analysis failed: {error}")
            return

    st.session_state.latest_recorded_csv = output_file
    summary = legacy_result["summary"]
    detected_reps = summary["total_reps"]

    if expected_reps_differ(expected_reps, detected_reps):
        st.warning("Expected reps and detected reps differ.")

    if comparison_error:
        st.warning(f"Latest compatible comparison was not refreshed: {comparison_error}")

    average_normalized = (
        f"{summary['normalized_average_rep_activation']:.1f}%"
        if summary["has_calibration"]
        else "Not available"
    )
    peak_normalized = (
        f"{summary['normalized_peak_activation']:.1f}%"
        if summary["has_calibration"]
        else "Not available"
    )

    metric_row([
        ("Detected reps", detected_reps),
        ("Expected reps", expected_reps),
        ("Avg normalized rep activation", average_normalized),
    ])
    metric_row([
        ("Peak normalized activation", peak_normalized),
        ("Activation trend", activation_change_phrase(summary["peak_drop_percent"])),
        ("Saved graph", legacy_result["graph_file"].name),
    ])

    stats = signal_check_stats(recording["readings"])
    for warning in signal_quality_warnings(stats):
        st.warning(warning)

    st.image(str(legacy_result["graph_file"]), caption=legacy_result["graph_file"].name, use_container_width=True)
    show_detector_comparison(expected_reps, legacy_result, hybrid_result)


def show_workout_session():
    current_source_type = st.session_state.get("workout_signal_source", "Serial hardware")
    confirmed = True

    if current_source_type == "Serial hardware":
        confirmed = show_preflight()

    source_type, port, replay_csv, replay_realtime = show_source_controls(confirmed)

    if source_type == "Replay CSV":
        show_replay_workflow(replay_csv, replay_realtime)
        return

    show_signal_check(confirmed, source_type, port, replay_csv, replay_realtime)
    show_workout_recording(confirmed, source_type, port, replay_csv, replay_realtime)

    latest_recorded_csv = st.session_state.get("latest_recorded_csv")
    if latest_recorded_csv:
        st.divider()
        show_latest_comparison(latest_recorded_csv)


def choose_session(protocol_type, label):
    sessions = load_sessions(protocol_type)

    if not sessions:
        st.info("No sessions have been created yet.")
        return None

    session_by_label = {
        f"{session.get('created_at', '')} - {session.get('exercise', '')}": session
        for session in sessions
    }
    selected_label = st.selectbox(
        label,
        list(session_by_label.keys()),
        key=f"{protocol_type}_session_select",
    )
    return session_by_label[selected_label]


def render_quality_messages(quality):
    if quality["warnings"]:
        for warning in quality["warnings"]:
            st.warning(warning)
    else:
        st.success("No quality heuristics triggered.")

    if quality.get("saturation"):
        saturation = quality["saturation"]
        active_percent = saturation["near_max_active_sample_percent"]
        active_text = "Not available" if active_percent is None else f"{active_percent:.1f}%"
        st.caption(
            "Saturation heuristic: "
            f"max observed {saturation['maximum_observed_signal']:.1f}; "
            f"{saturation['near_max_sample_percent']:.1f}% of all samples near max; "
            f"{active_text} of active-rep samples near max."
        )


def render_recording_slot(label, csv_name, expected_reps, calibration_csv, required_side):
    st.markdown(f"**{label}**")

    if not csv_name:
        st.info("Pending recording.")
        return None

    csv_file = selected_csv(csv_name)
    metadata = load_metadata(csv_file)
    summary = recording_summary(csv_file)
    graph_path = graph_path_for_csv(csv_file)
    calibration, calibration_metadata = load_calibration_from_csv_name(calibration_csv)
    quality = recording_quality(
        csv_file,
        metadata,
        summary,
        graph_path,
        calibration=calibration,
        calibration_metadata=calibration_metadata,
        required_side=required_side,
        expected_reps=expected_reps,
    )
    analysis = quality.get("analysis", {})

    metric_row([
        ("Status", "Recorded"),
        ("Detected / expected", f"{summary['detected_reps'] or analysis.get('rep_count', '')} / {expected_reps}"),
        ("Avg normalized rep activation", f"{analysis.get('normalized_average_activation', 0):.1f}%" if calibration else "Not available"),
    ])
    metric_row([
        ("Activation trend", display_value(summary["activation_trend"])),
        ("Rep consistency", quality.get("rep_consistency", "Not available")),
        ("Calibration", calibration_csv or "Missing"),
    ])

    render_quality_messages(quality)

    if graph_path.exists():
        st.image(str(graph_path), caption=graph_path.name, use_container_width=True)

    return quality


def side_average(qualities, side, metric):
    values = [
        quality["analysis"].get(metric, 0)
        for key, quality in qualities.items()
        if key.startswith(side) and quality and quality.get("analysis")
    ]
    return sum(values) / len(values) if values else 0


def side_repeatability(qualities, side):
    values = [
        quality["analysis"].get("normalized_average_activation", 0)
        for key, quality in qualities.items()
        if key.startswith(side) and quality and quality.get("analysis")
    ]

    if len(values) != 2:
        return "Not available"

    return f"{percent_difference(values[0], values[1]):.1f}% difference between same-side sets"


def show_side_comparison():
    st.header("Side Comparison")
    st.write(
        "Use one MyoWare sensor sequentially: record one side, move and recalibrate, "
        "then record the other side. Results are standardized side-to-side signal "
        "differences, not a medical diagnosis. Reproduce electrode placement as "
        "closely as possible."
    )

    recordings = valid_workout_recordings()
    calibrations = calibration_recordings()

    with st.expander("Create side-comparison session"):
        with st.form("new_side_comparison"):
            exercise = st.text_input("Exercise", value="bicep curl", key="side_exercise")
            muscle = st.text_input("Muscle", value="bicep", key="side_muscle")
            weight = st.text_input("Weight", key="side_weight")
            expected_reps = st.text_input("Expected reps", value="6", key="side_expected_reps")
            cadence_notes = st.text_area("Cadence notes", key="side_cadence_notes")
            testing_order = st.selectbox(
                "Right/left testing order",
                ["Right then left", "Left then right"],
                key="side_testing_order",
            )
            placement_notes = st.text_area("Placement notes", key="side_placement_notes")
            session_notes = st.text_area("Optional session notes", key="side_session_notes")
            right_calibration_csv = st.selectbox(
                "Right calibration",
                csv_options(calibrations),
                key="side_right_calibration",
            )
            left_calibration_csv = st.selectbox(
                "Left calibration",
                csv_options(calibrations),
                key="side_left_calibration",
            )

            if st.form_submit_button("Create session"):
                session = new_side_comparison_session({
                    "exercise": exercise,
                    "muscle": muscle,
                    "weight": weight,
                    "expected_reps": expected_reps,
                    "cadence_notes": cadence_notes,
                    "testing_order": testing_order,
                    "placement_notes": placement_notes,
                    "session_notes": session_notes,
                    "right_calibration_csv": right_calibration_csv,
                    "left_calibration_csv": left_calibration_csv,
                })
                save_session(session)
                st.success("Side-comparison session created.")

    session = choose_session("single_sensor_side_comparison", "Side-comparison session")

    if session is None:
        return

    st.subheader("Protocol")
    metric_row([
        ("Exercise", session.get("exercise", "")),
        ("Muscle", session.get("muscle", "")),
        ("Weight", session.get("weight", "")),
        ("Expected reps", session.get("expected_reps", "")),
    ])
    st.caption(f"Order: {session.get('testing_order', '')}")

    with st.form(f"assign_side_{session['session_id']}"):
        updated = dict(session)
        updated["recordings"] = dict(session.get("recordings", {}))

        for slot in SIDE_COMPARISON_SLOTS:
            options = csv_options(recordings)
            current_value = session.get("recordings", {}).get(slot["key"], "")
            updated["recordings"][slot["key"]] = st.selectbox(
                slot["label"],
                options,
                index=options.index(current_value) if current_value in options else 0,
                key=f"side_{session['session_id']}_{slot['key']}",
            )

        if st.form_submit_button("Save recording assignments"):
            save_session(updated)
            session = updated
            st.success("Assignments saved.")

    qualities = {}
    columns = st.columns(2)

    for index, slot in enumerate(SIDE_COMPARISON_SLOTS):
        with columns[index % 2]:
            calibration_csv = session.get("calibrations", {}).get(slot["side"], "")
            qualities[slot["key"]] = render_recording_slot(
                slot["label"],
                session.get("recordings", {}).get(slot["key"], ""),
                session.get("expected_reps", ""),
                calibration_csv,
                slot["side"],
            )

    all_recorded = all(session.get("recordings", {}).get(slot["key"]) for slot in SIDE_COMPARISON_SLOTS)

    if not all_recorded:
        st.info("Comparison will appear after all four expected recordings are assigned.")
        return

    st.subheader("Cautious Side-to-Side Comparison")
    right_avg = side_average(qualities, "right", "normalized_average_activation")
    left_avg = side_average(qualities, "left", "normalized_average_activation")
    right_peak = side_average(qualities, "right", "normalized_peak_activation")
    left_peak = side_average(qualities, "left", "normalized_peak_activation")
    right_duration = side_average(qualities, "right", "average_rep_duration")
    left_duration = side_average(qualities, "left", "average_rep_duration")
    saturated = any(
        quality and any("saturation" in warning for warning in quality["warnings"])
        for quality in qualities.values()
    )
    higher_side = "Right" if right_avg > left_avg else "Left" if left_avg > right_avg else "Tie"

    metric_row([
        ("Avg normalized activation difference", f"{percent_difference(right_avg, left_avg):.1f}%"),
        ("Peak normalized activation difference", "Skipped: saturation warning" if saturated else f"{percent_difference(right_peak, left_peak):.1f}%"),
        ("Rep-duration difference", f"{abs(right_duration - left_duration):.2f}s"),
    ])
    metric_row([
        ("Right repeatability", side_repeatability(qualities, "right")),
        ("Left repeatability", side_repeatability(qualities, "left")),
        ("Higher activation in this test", higher_side),
    ])
    st.caption("This is a standardized single-sensor protocol result, not a clinical or definitive muscular-imbalance label.")


def show_weight_ladder():
    st.header("Weight Ladder")
    st.write(
        "Test the same exercise, muscle, side, and calibration across 2-5 weights. "
        "Use the results to compare signal behavior across loads; higher EMG does "
        "not automatically mean better exercise quality or greater strength."
    )

    recordings = valid_workout_recordings()
    calibrations = calibration_recordings()

    with st.expander("Create weight-ladder session"):
        with st.form("new_weight_ladder"):
            exercise = st.text_input("Exercise", value="bicep curl", key="ladder_exercise")
            muscle = st.text_input("Muscle", value="bicep", key="ladder_muscle")
            side = st.selectbox("Side", ["right", "left"], key="ladder_side")
            calibration_csv = st.selectbox("Calibration", csv_options(calibrations), key="ladder_calibration")
            expected_reps = st.text_input("Expected reps per set", value="6", key="ladder_reps")
            rest_time_notes = st.text_area("Rest-time notes", key="ladder_rest_notes")
            weights_text = st.text_input("Weight levels, comma-separated", value="3, 5", key="ladder_weights")
            session_notes = st.text_area("Optional session notes", key="ladder_notes")

            if st.form_submit_button("Create ladder"):
                weights = [weight.strip() for weight in weights_text.split(",")]
                weight_count = len([weight for weight in weights if weight])

                if weight_count < 2 or weight_count > 5:
                    st.error("Use 2-5 weight levels.")
                else:
                    session = new_weight_ladder_session({
                        "exercise": exercise,
                        "muscle": muscle,
                        "side": side,
                        "calibration_csv": calibration_csv,
                        "expected_reps": expected_reps,
                        "rest_time_notes": rest_time_notes,
                        "weights": weights,
                        "session_notes": session_notes,
                    })
                    save_session(session)
                    st.success("Weight-ladder session created.")

    session = choose_session("weight_ladder", "Weight-ladder session")

    if session is None:
        return

    with st.form(f"assign_ladder_{session['session_id']}"):
        updated = dict(session)
        updated["weight_levels"] = [dict(level) for level in session.get("weight_levels", [])]

        for index, level in enumerate(updated["weight_levels"]):
            options = csv_options(recordings)
            current_value = level.get("recording_csv", "")
            level["recording_csv"] = st.selectbox(
                f"{level['weight']} recording",
                options,
                index=options.index(current_value) if current_value in options else 0,
                key=f"ladder_{session['session_id']}_{index}",
            )

        if st.form_submit_button("Save ladder assignments"):
            save_session(updated)
            session = updated
            st.success("Assignments saved.")

    calibration, calibration_metadata = load_calibration_from_csv_name(session.get("calibration_csv", ""))
    rows = []
    qualities = []

    for level in session.get("weight_levels", []):
        csv_name = level.get("recording_csv", "")

        if not csv_name:
            rows.append({
                "Weight": level["weight"],
                "Status": "Pending",
                "Detected reps": "",
                "Avg normalized activation": "",
                "Peak normalized activation": "",
                "Avg rep duration": "",
                "Activation trend": "",
                "Signal-quality warnings": "",
            })
            continue

        csv_file = selected_csv(csv_name)
        metadata = load_metadata(csv_file)
        summary = recording_summary(csv_file)
        graph_path = graph_path_for_csv(csv_file)
        quality = recording_quality(
            csv_file,
            metadata,
            summary,
            graph_path,
            calibration=calibration,
            calibration_metadata=calibration_metadata,
            required_side=session.get("side", ""),
            expected_reps=session.get("expected_reps", ""),
        )
        analysis = quality.get("analysis", {})
        qualities.append((level, quality))
        rows.append({
            "Weight": level["weight"],
            "Status": "Recorded",
            "Detected reps": summary["detected_reps"] or analysis.get("rep_count", ""),
            "Avg normalized activation": f"{analysis.get('normalized_average_activation', 0):.1f}%" if calibration else "Not available",
            "Peak normalized activation": f"{analysis.get('normalized_peak_activation', 0):.1f}%" if calibration else "Not available",
            "Avg rep duration": f"{analysis.get('average_rep_duration', 0):.2f}s",
            "Activation trend": summary["activation_trend"],
            "Signal-quality warnings": "; ".join(quality["warnings"]) or "None",
        })

    st.table(rows)

    for level, quality in qualities:
        with st.expander(f"{level['weight']} details"):
            render_quality_messages(quality)
            graph_path = graph_path_for_csv(selected_csv(level["recording_csv"]))
            if graph_path.exists():
                st.image(str(graph_path), caption=graph_path.name, use_container_width=True)

    complete_rows = [
        row for row in rows
        if row["Status"] == "Recorded" and row["Avg normalized activation"] != "Not available"
    ]

    if len(complete_rows) >= 2:
        activations = [
            (row["Weight"], number_from_text(row["Avg normalized activation"]))
            for row in complete_rows
        ]
        highest = max(activations, key=lambda item: item[1] if item[1] is not None else -1)
        lowest = min(activations, key=lambda item: item[1] if item[1] is not None else 999999)
        st.subheader("Across-Weight Comparison")
        st.write(
            f"Average normalized activation ranged from {lowest[1]:.1f}% at {lowest[0]} "
            f"to {highest[1]:.1f}% at {highest[0]} in this standardized ladder."
        )
        st.caption("Interpret this as EMG signal behavior only; it is not a strength or exercise-quality score.")


def main():
    st.set_page_config(page_title="RepAI", layout="wide")
    st.title("RepAI")

    overview_tab, session_tab, side_tab, ladder_tab = st.tabs([
        "Overview",
        "Workout Session",
        "Side Comparison",
        "Weight Ladder",
    ])

    with overview_tab:
        show_calibration()
        selected_file = show_recording(valid_workout_recordings())
        show_existing_detector_comparison(selected_file)
        show_latest_comparison()

    with session_tab:
        show_workout_session()

    with side_tab:
        show_side_comparison()

    with ladder_tab:
        show_weight_ladder()


if __name__ == "__main__":
    main()
