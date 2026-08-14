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
    compatible_calibration_recordings,
    comparison_mentions_file,
    current_graph_path_for_csv,
    current_summary_path_for_csv,
    graph_path_for_csv,
    latest_comparison,
    load_calibration_from_csv_name,
    load_compatible_current_calibration,
    load_current_calibration,
    output_is_current,
    recording_summary,
    summary_path_for_csv,
    valid_workout_recordings,
)


BASE_DIR = Path(__file__).resolve().parent.parent
PYTHON_DIR = BASE_DIR / "python"
DATA_DIR = BASE_DIR / "data"

if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from acquisition import ReplaySignalSource, SerialSignalSource, available_serial_ports  # noqa: E402
from compare_latest_sets import main as refresh_latest_comparison  # noqa: E402
from dataset_builder import (  # noqa: E402
    annotation_figure,
    create_session_manifest,
    list_dataset_sessions,
    load_annotations,
    new_session_id,
    planned_cue_schedule,
    planned_total_duration,
    recording_duration,
    resolve_repo_path,
    save_annotations,
    save_dataset_session,
    validate_annotation_rows,
)
from detect_reps import (  # noqa: E402
    SMOOTHING_WINDOW,
    activation_change_phrase,
    analyze_csv_file,
    detect_reps,
    detect_reps_hybrid,
    detector_threshold_values,
    moving_average,
    read_signal,
)
from generate_report import main as regenerate_reports  # noqa: E402
from recording_metadata import load_metadata, save_metadata  # noqa: E402
from recording_metadata import metadata_path_for_csv  # noqa: E402
from set_lifecycle import (  # noqa: E402
    SetLifecycle,
    SetLifecycleConfig,
    SetSessionSpec,
    SetState,
    run_once_for_completed_set,
    write_recording_atomic,
)


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


def detector_comparison_recordings():
    recordings = valid_workout_recordings()
    diagnostic_dir = DATA_DIR / "diagnostics"

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
    thresholds = detector_threshold_values(times, smoothed_values, values)
    start_threshold = thresholds["start_threshold"]
    end_threshold = thresholds["end_threshold"]
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


def detector_result_for_samples(samples):
    if len(samples) < 3:
        return {"summary": {"total_reps": 0}, "reps": []}

    times = [sample.time_ms / 1000 for sample in samples]
    values = [sample.emg_value for sample in samples]
    smoothed_values = moving_average(values, SMOOTHING_WINDOW)
    thresholds = detector_threshold_values(times, smoothed_values, values)
    signal_range = thresholds["signal_range"]

    if signal_range <= 0:
        return {"summary": {"total_reps": 0}, "reps": []}

    start_threshold = thresholds["start_threshold"]
    end_threshold = thresholds["end_threshold"]
    reps, diagnostics = detect_reps_hybrid(
        times,
        values,
        smoothed_values,
        start_threshold,
        end_threshold,
    )
    return {
        "summary": {"total_reps": len(reps)},
        "reps": reps,
        "diagnostics": diagnostics,
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


def write_dataset_recording(source, output_file, metadata, duration_seconds, cue_schedule, progress_bar):
    readings = []
    malformed_reads = 0
    deadline = time.monotonic() + duration_seconds
    phase_placeholder = st.empty()
    source.connect()

    try:
        metadata_file = save_metadata(output_file, metadata)

        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["time_ms", "host_time_ms", "emg_value"])
            first_host_time_ms = None

            while time.monotonic() < deadline:
                elapsed_seconds = duration_seconds - (deadline - time.monotonic())
                progress_bar.progress(min(1.0, elapsed_seconds / duration_seconds))
                active_cue = next(
                    (
                        cue for cue in cue_schedule
                        if cue["start_time"] <= elapsed_seconds < cue["end_time"]
                    ),
                    None,
                )
                if active_cue:
                    phase_placeholder.info(
                        f"Rep {active_cue['rep']} - {active_cue['phase'].title()}"
                    )
                else:
                    phase_placeholder.info("Recording")

                reading = source.read()

                if reading is None:
                    malformed_reads += 1
                    continue

                if first_host_time_ms is None:
                    first_host_time_ms = reading.host_time_ms

                elapsed_ms = reading.host_time_ms - first_host_time_ms
                writer.writerow([elapsed_ms, reading.host_time_ms, reading.signal_value])
                readings.append(reading)
    finally:
        source.disconnect()
        progress_bar.progress(1.0)
        phase_placeholder.empty()

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
        ("Setup ID", source_metadata.get("calibration_setup_id", "")),
        ("Recorded", source_metadata.get("timestamp", "")),
        ("Gain note", "Recalibrate after MyoWare ENV gain changes."),
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


def active_set_is_running():
    lifecycle = st.session_state.get("workout_set_lifecycle")
    return lifecycle is not None and lifecycle.state in {
        SetState.COUNTDOWN,
        SetState.RECORDING,
        SetState.POSSIBLE_END,
        SetState.ANALYZING,
    }


WORKOUT_REPLAY_METADATA_FIELDS = {
    "workout_exercise": ("exercise_name", "exercise"),
    "workout_muscle": ("muscle",),
    "workout_side": ("side",),
    "workout_weight": ("weight",),
    "workout_expected_reps": ("planned_reps", "expected_reps"),
    "workout_calibration_csv": ("calibration_csv", "calibration_source_csv", "source_csv"),
}


def replay_metadata_defaults(replay_csv):
    if replay_csv is None:
        return {}

    metadata = load_metadata(Path(replay_csv))
    defaults = {}

    for widget_key, metadata_keys in WORKOUT_REPLAY_METADATA_FIELDS.items():
        for metadata_key in metadata_keys:
            value = metadata.get(metadata_key)
            if value not in ("", None):
                defaults[widget_key] = str(value)
                break

    defaults.setdefault("workout_exercise", Path(replay_csv).stem)
    return defaults


def sync_replay_metadata_to_state(state, replay_csv, ready=True):
    if not ready or replay_csv is None:
        return False

    replay_path = str(Path(replay_csv).resolve())
    if state.get("workout_replay_metadata_source") == replay_path:
        return False

    for key, value in replay_metadata_defaults(replay_csv).items():
        state[key] = value

    state["workout_replay_metadata_source"] = replay_path
    return True


def clear_replay_metadata_source_marker(state):
    state.pop("workout_replay_metadata_source", None)


def show_source_controls(collection_enabled):
    st.subheader("Connection")

    lifecycle = st.session_state.get("workout_set_lifecycle")
    if active_set_is_running() and lifecycle.active_spec is not None:
        spec = lifecycle.active_spec
        st.info(f"Active set source is locked: {spec.source_label}")
        status = st.session_state.get("connection_status", "Disconnected")
        if status.startswith("Error:"):
            st.error(status)
        else:
            st.info(status)
        return spec.source_type, spec.port, spec.replay_csv, spec.replay_realtime

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
        replay_files = valid_workout_recordings()
        if replay_files:
            selected_name = st.selectbox(
                "Replay recording",
                [file.name for file in replay_files],
                key="workout_replay_recording",
            )
            replay_csv = next(file for file in replay_files if file.name == selected_name)
            sync_replay_metadata_to_state(
                st.session_state,
                replay_csv,
                ready=not active_set_is_running(),
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


def workout_lifecycle():
    lifecycle = st.session_state.get("workout_set_lifecycle")

    if lifecycle is None:
        lifecycle = SetLifecycle(config=SetLifecycleConfig())
        st.session_state.workout_set_lifecycle = lifecycle

    return lifecycle


def close_workout_source():
    source = st.session_state.get("workout_set_source")
    if source is None:
        return

    try:
        source.disconnect()
    except Exception:  # noqa: BLE001
        pass

    st.session_state.workout_set_source = None


def workout_setup_payload():
    return {
        "exercise": st.session_state.get("workout_exercise", "workout_set").strip() or "workout_set",
        "muscle": st.session_state.get("workout_muscle", "").strip(),
        "side": st.session_state.get("workout_side", "").strip(),
        "weight": st.session_state.get("workout_weight", "").strip(),
        "planned_reps": str(st.session_state.get("workout_expected_reps", "")).strip(),
        "participant_id": st.session_state.get("workout_participant_id", "").strip(),
        "comparison_target": st.session_state.get("workout_comparison_target", "").strip(),
        "notes": st.session_state.get("workout_notes", "").strip(),
        "calibration_csv": st.session_state.get("workout_calibration_csv", "").strip(),
        "calibration_setup_id": st.session_state.get("workout_calibration_setup_id", "").strip(),
    }


def replay_source_metadata(source_type, replay_csv):
    if source_type != "Replay CSV" or replay_csv is None:
        return {
            "source_replay_csv": "",
            "source_replay_filename": "",
        }

    return {
        "source_replay_csv": str(replay_csv.resolve()),
        "source_replay_filename": replay_csv.name,
    }


def build_workout_session_spec(source_type, port, replay_csv, replay_realtime):
    payload = workout_setup_payload()
    timestamp = datetime.now()
    output_file = unique_recording_path(payload["exercise"], timestamp)
    metadata = {
        "session_id": f"workout_{timestamp.strftime('%Y%m%d_%H%M%S')}",
        "participant_id": payload["participant_id"],
        "exercise_name": payload["exercise"],
        "muscle": payload["muscle"],
        "side": payload["side"],
        "weight": payload["weight"],
        "expected_reps": payload["planned_reps"],
        "planned_reps": payload["planned_reps"],
        "comparison_target": payload["comparison_target"],
        "calibration_csv": payload["calibration_csv"],
        "calibration_setup_id": payload["calibration_setup_id"],
        "source_type": source_type,
        "data_type": "real",
        "test_type": "workout_set",
        "set_end_rule": "sustained_low_emg_activity",
        "notes": payload["notes"],
        "csv_filename": output_file.name,
        "timestamp": timestamp.isoformat(timespec="seconds"),
    }
    metadata.update(replay_source_metadata(source_type, replay_csv))
    return SetSessionSpec(
        source_type=source_type,
        port=port,
        replay_csv=Path(replay_csv) if replay_csv is not None else None,
        replay_realtime=replay_realtime,
        output_file=output_file,
        metadata_file=metadata_path_for_csv(output_file),
        metadata=metadata,
    )


def compatible_previous_recording(current_csv, metadata):
    candidates = []

    for recording in valid_workout_recordings():
        if recording == current_csv:
            continue

        candidate_metadata = load_metadata(recording)
        compatible = all(
            str(candidate_metadata.get(key, "")).strip().lower()
            == str(metadata.get(key, "")).strip().lower()
            for key in ("exercise_name", "side", "weight")
        )

        if compatible:
            candidates.append(recording)

    return candidates[0] if candidates else None


def build_completed_set_dataset_manifest(csv_file, metadata, analysis_result):
    created_at = datetime.now()
    session_id = new_session_id(
        metadata.get("participant_id", "") or "workout",
        metadata.get("exercise_name", "") or "workout_set",
        created_at=created_at,
    )
    exercise_metadata = {
        "exercise": metadata.get("exercise_name", ""),
        "muscle": metadata.get("muscle", ""),
        "side": metadata.get("side", ""),
        "weight": metadata.get("weight", ""),
        "body_position": metadata.get("body_position", ""),
        "grip": metadata.get("grip", ""),
        "placement_id": metadata.get("placement_id", ""),
    }
    cue_schedule = []
    calibration_csv = metadata.get("calibration_csv", "")
    calibration_path = selected_csv(calibration_csv) if calibration_csv else None

    try:
        planned_reps = int(metadata.get("expected_reps") or metadata.get("planned_reps") or 0)
    except (TypeError, ValueError):
        planned_reps = 0

    manifest = create_session_manifest(
        session_id=session_id,
        participant_id=metadata.get("participant_id", "") or "workout",
        recording_csv=csv_file,
        calibration_csv=calibration_path,
        exercise_metadata=exercise_metadata,
        planned_reps=planned_reps,
        cadence={},
        cue_timestamps=cue_schedule,
        recording_started_at=metadata.get("timestamp", ""),
        notes=metadata.get("notes", ""),
    )
    manifest["source_workout_session_id"] = metadata.get("session_id", "")
    manifest["workout_detector_outputs"] = {
        "legacy": (analysis_result.get("legacy") or {}).get("summary", {}),
        "hybrid": (analysis_result.get("hybrid") or {}).get("summary", {}),
    }
    return manifest


def completed_set_identity(metadata, csv_file):
    return {
        "source_replay_filename": metadata.get("source_replay_filename") or "Not replay",
        "output_recording_filename": Path(csv_file).name,
        "exercise": metadata.get("exercise_name", ""),
        "side": metadata.get("side", ""),
        "weight": metadata.get("weight", ""),
    }


def render_completed_set_results(lifecycle):
    analysis_result = lifecycle.analysis_result or {}
    csv_file = Path(analysis_result.get("csv_file", lifecycle.output_file))
    metadata = analysis_result.get("metadata") or load_metadata(csv_file)
    legacy_result = analysis_result.get("legacy")
    hybrid_result = analysis_result.get("hybrid")
    comparison_error = analysis_result.get("comparison_error")
    summary = (hybrid_result or legacy_result)["summary"]
    legacy_summary = (legacy_result or {}).get("summary", {})
    has_calibration = summary.get("has_calibration", False)
    identity = completed_set_identity(metadata, csv_file)

    st.subheader("Set Results")
    st.caption(
        "Hybrid rep detection is experimental. Set completion is inferred from sustained low EMG activity, "
        "not from detecting that the weight was physically put down."
    )

    if not has_calibration:
        st.warning("Calibration is missing or unusable. Normalized activation uses safe fallback behavior and may be unavailable.")

    if comparison_error:
        st.warning(f"Latest compatible comparison was not refreshed: {comparison_error}")

    metric_row([
        ("Hybrid reps", summary["total_reps"]),
        ("Legacy reps", legacy_summary.get("total_reps", "Not available")),
        ("Source replay", identity["source_replay_filename"]),
        ("Output recording", identity["output_recording_filename"]),
    ])
    metric_row([
        (
            "Avg normalized activation",
            f"{summary['normalized_average_rep_activation']:.1f}%" if has_calibration else "Not available",
        ),
        (
            "Peak normalized activation",
            f"{summary['normalized_peak_activation']:.1f}%" if has_calibration else "Not available",
        ),
        ("Rep duration", f"{summary['average_rep_duration']:.2f}s avg"),
    ])
    metric_row([
        ("Activation trend", activation_change_phrase(summary["peak_drop_percent"])),
        ("Exercise", identity["exercise"]),
        ("Side", identity["side"]),
        ("Weight", identity["weight"]),
    ])
    metric_row([
        ("Planned reps", metadata.get("expected_reps", "")),
        ("Session ID", metadata.get("session_id", "")),
    ])

    previous = compatible_previous_recording(csv_file, metadata)
    if previous:
        st.markdown("**Immediately previous compatible set**")
        st.write(previous.name)
        show_latest_comparison(csv_file)
    else:
        st.info("No previous compatible set was found for exercise, side, and weight.")

    if hybrid_result:
        show_rep_intervals("Hybrid", hybrid_result)

    if legacy_result and legacy_result.get("graph_file") and legacy_result["graph_file"].exists():
        st.image(str(legacy_result["graph_file"]), caption=legacy_result["graph_file"].name, use_container_width=True)

    if st.button("Add this set to Dataset Builder", key=f"dataset_from_set_{csv_file.stem}"):
        try:
            manifest = build_completed_set_dataset_manifest(csv_file, metadata, analysis_result)
            manifest_file, annotation_file = save_dataset_session(manifest)
        except Exception as error:  # noqa: BLE001
            st.error(f"Dataset session creation failed: {error}")
        else:
            st.success(f"Saved dataset session: {manifest['session_id']}")
            st.write(f"Manifest: {manifest_file.resolve()}")
            st.write(f"Annotations: {annotation_file.resolve()}")

    if st.button("Start Next Set", key="workout_start_next_set"):
        close_workout_source()
        lifecycle.reset()
        clear_replay_metadata_source_marker(st.session_state)
        st.rerun()


def analyze_completed_set_once(lifecycle, metadata):
    if lifecycle.output_file is None:
        raise ValueError("Completed set is missing an output file.")

    analysis_key = str(lifecycle.output_file.resolve())
    if lifecycle.write_result is None:
        if not lifecycle.samples:
            raise ValueError("No valid readings were captured, so analysis was skipped.")
        metadata = dict(metadata)
        metadata["acquisition_diagnostics"] = lifecycle.acquisition_diagnostics()
        output_file, metadata_file = write_recording_atomic(
            lifecycle.output_file,
            lifecycle.metadata_file,
            metadata,
            lifecycle.samples,
        )
        lifecycle.write_result = {
            "csv_file": output_file,
            "metadata_file": metadata_file,
        }

    def analyze():
        legacy_result, hybrid_result, comparison_error = run_post_recording_pipeline(lifecycle.output_file)
        return {
            "csv_file": lifecycle.output_file,
            "metadata_file": lifecycle.metadata_file,
            "metadata": metadata,
            "legacy": legacy_result,
            "hybrid": hybrid_result,
            "comparison_error": comparison_error,
        }

    return run_once_for_completed_set(lifecycle, analysis_key, analyze)


def render_live_recording_controls(lifecycle, source):
    status = st.session_state.get("workout_set_status", {})
    elapsed = status.get("elapsed_seconds", 0.0)
    approx = detector_result_for_samples(lifecycle.samples)
    auto_stop_armed = status.get("auto_stop_armed", False)

    if lifecycle.state == SetState.POSSIBLE_END:
        state_label = "Possible End"
    elif lifecycle.state == SetState.RECORDING and auto_stop_armed:
        state_label = "Auto-Stop Armed"
    elif lifecycle.state == SetState.RECORDING:
        state_label = "Recording"
    else:
        state_label = lifecycle.state.value.replace("_", " ").title()

    metric_row([
        ("State", state_label),
        ("Elapsed", f"{elapsed:.1f}s"),
        ("Provisional hybrid reps", approx["summary"]["total_reps"]),
    ])
    metric_row([
        ("Smoothed signal", f"{status.get('smoothed_value', 0):.1f}"),
        ("Activity threshold", f"{status.get('threshold', 0):.1f}"),
        ("Activity", "Detected" if status.get("active") else "Low"),
    ])
    metric_row([
        ("Auto-stop", "Armed" if auto_stop_armed else "Not armed"),
        ("Activity episodes", status.get("substantial_activity_episodes", 0)),
        ("Active time", f"{status.get('active_duration_seconds', 0.0):.1f}s"),
    ])

    if lifecycle.state == SetState.RECORDING and not auto_stop_armed:
        st.info("Auto-stop will arm after a real set pattern is detected.")

    if lifecycle.state == SetState.POSSIBLE_END and lifecycle.inference is not None:
        inactivity_elapsed = status.get("inactivity_elapsed_seconds", 0.0)
        remaining = lifecycle.inference.inactivity_remaining_seconds(elapsed)
        st.warning(
            "Possible set ending. "
            f"Low activity for {inactivity_elapsed:.1f}s; "
            f"auto-finish in about {remaining:.1f}s if low activity continues."
        )

    timing_warning = status.get("timing_drift_warning")
    if timing_warning:
        st.warning(f"Acquisition timing warning: {timing_warning}")

    control_cols = st.columns(2)
    finish_clicked = control_cols[0].button("Finish Set", key="workout_manual_finish")
    cancel_clicked = control_cols[1].button("Cancel Set", key="workout_manual_cancel")

    if cancel_clicked:
        lifecycle.cancel()
        close_workout_source()
        st.rerun()

    if finish_clicked:
        lifecycle.finish()
        close_workout_source()
        st.rerun()

    try:
        if hasattr(source, "read_many"):
            readings = source.read_many()
        else:
            reading = source.read()
            readings = [reading] if reading is not None else []
    except Exception as error:  # noqa: BLE001
        close_workout_source()
        lifecycle.fail(error)
        st.rerun()

    if not readings:
        close_workout_source()
        if lifecycle.inference and lifecycle.inference.auto_stop_armed:
            lifecycle.finish()
        else:
            lifecycle.cancel("Signal source ended before meaningful contraction activity was observed.")
        st.rerun()

    for reading in readings:
        status = lifecycle.observe(reading)
        st.session_state.workout_set_status = status

        if lifecycle.state == SetState.ANALYZING:
            break

    if lifecycle.state == SetState.ANALYZING:
        close_workout_source()
        st.rerun()

    time.sleep(0.05)
    st.rerun()


def show_workout_recording(collection_enabled, source_type, port, replay_csv, replay_realtime):
    st.subheader("Real-Time Set")
    lifecycle = workout_lifecycle()
    missing_source = source_type == "Serial hardware" and port is None
    missing_source = missing_source or (source_type == "Replay CSV" and replay_csv is None)

    if lifecycle.state == SetState.READY:
        with st.form("workout_recording_form"):
            exercise = st.text_input("Exercise", value="synthetic_bicep_curl", key="workout_exercise")
            muscle = st.text_input("Muscle", value="bicep", key="workout_muscle")
            side = st.text_input("Side", value="right", key="workout_side")
            weight = st.text_input("Weight", value="3", key="workout_weight")
            expected_reps = st.text_input("Planned reps", value="6", key="workout_expected_reps")
            participant_id = st.text_input("Participant ID (optional)", value="", key="workout_participant_id")
            comparison_target = st.text_input("Optional comparison target", value="", key="workout_comparison_target")
            setup_id = st.text_input(
                "Sensor setup ID",
                value=st.session_state.get("workout_calibration_setup_id", ""),
                key="workout_calibration_setup_id",
            )
            notes = st.text_area("Notes", value="", key="workout_notes")
            compatible_calibrations = compatible_calibration_recordings(muscle, side, setup_id)
            if not setup_id.strip():
                st.warning(
                    "Set a sensor setup ID and record a new calibration whenever MyoWare ENV gain or electrode setup changes."
                )
            if compatible_calibrations:
                st.info(f"Calibration: {compatible_calibrations[0].name}")
            else:
                st.warning("A compatible calibration for this muscle and side is required before recording.")
            submitted = st.form_submit_button(
                "Start Countdown",
                disabled=not collection_enabled or missing_source or not compatible_calibrations,
            )

        metric_row([
            ("Exercise", exercise),
            ("Side", side),
            ("Weight", weight),
            ("Planned reps", expected_reps),
        ])
        if comparison_target:
            st.info(f"Comparison target: {comparison_target}")

        if not submitted:
            return

        DATA_DIR.mkdir(exist_ok=True)
        spec = build_workout_session_spec(source_type, port, replay_csv, replay_realtime)
        st.session_state.workout_set_countdown_started_at = lifecycle.begin_session(spec)
        st.rerun()
        return

    if lifecycle.state == SetState.COUNTDOWN:
        spec = lifecycle.active_spec
        if spec is None:
            lifecycle.fail("Set session was not initialized.")
            st.rerun()

        started_at = st.session_state.get("workout_set_countdown_started_at", time.monotonic())
        remaining = max(0, 3 - int(time.monotonic() - started_at))
        if remaining > 0:
            st.warning(f"Recording begins in {remaining}")
            if st.button("Cancel Set", key="workout_cancel_countdown"):
                lifecycle.cancel()
                st.rerun()
            time.sleep(0.25)
            st.rerun()

        calibration, _ = load_compatible_current_calibration(
            spec.metadata.get("muscle", ""),
            spec.metadata.get("side", ""),
            spec.metadata.get("calibration_setup_id", ""),
        )
        if calibration is None:
            lifecycle.fail(
                "A compatible calibration for this muscle and side is required before recording."
            )
            st.rerun()

        source = source_from_selection(
            spec.source_type,
            port=spec.port,
            replay_csv=spec.replay_csv,
            replay_realtime=spec.replay_realtime,
        )
        try:
            source.connect()
        except Exception as error:  # noqa: BLE001
            lifecycle.fail(error)
            st.rerun()

        lifecycle.start_recording(calibration=calibration)
        lifecycle.remember_source_object(source)
        st.session_state.workout_set_source = source
        metadata = dict(spec.metadata)
        metadata["timestamp"] = lifecycle.started_at
        metadata["calibration_csv"] = (calibration or {}).get("source_csv", "")
        st.session_state.workout_set_metadata = metadata
        st.rerun()

    if lifecycle.state in (SetState.RECORDING, SetState.POSSIBLE_END):
        source = st.session_state.get("workout_set_source")
        if source is None:
            lifecycle.fail("Signal source was not available during recording.")
            st.rerun()
        render_live_recording_controls(lifecycle, source)
        return

    if lifecycle.state == SetState.ANALYZING:
        st.info("Analyzing completed set...")
        metadata = st.session_state.get("workout_set_metadata", {})
        try:
            result = analyze_completed_set_once(lifecycle, metadata)
        except Exception as error:  # noqa: BLE001
            lifecycle.fail(error)
            st.rerun()

        st.session_state.latest_recorded_csv = result["csv_file"]
        lifecycle.mark_results(result)
        st.rerun()

    if lifecycle.state == SetState.RESULTS:
        render_completed_set_results(lifecycle)
        return

    if lifecycle.state == SetState.CANCELLED:
        st.warning(lifecycle.cancelled_reason or "Set was cancelled. No analysis was run.")
        if st.button("Start Next Set", key="workout_reset_cancelled"):
            close_workout_source()
            lifecycle.reset()
            clear_replay_metadata_source_marker(st.session_state)
            st.rerun()
        return

    if lifecycle.state == SetState.ERROR:
        st.error(lifecycle.error_message)
        if st.button("Reset Set Flow", key="workout_reset_error"):
            close_workout_source()
            lifecycle.reset()
            clear_replay_metadata_source_marker(st.session_state)
            st.rerun()


def show_workout_session():
    confirmed = show_preflight()
    source_type, port, replay_csv, replay_realtime = show_source_controls(confirmed)
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


def dataset_source_controls():
    source_type = st.radio(
        "Signal source",
        ["Serial hardware", "Replay CSV"],
        horizontal=True,
        key="dataset_signal_source",
    )
    port = None
    replay_csv = None
    replay_realtime = False

    if source_type == "Serial hardware":
        ports = available_serial_ports()

        if not ports:
            st.warning("No serial ports detected.")
        else:
            options = [f"{port_info['device']} - {port_info['description']}" for port_info in ports]
            selected = st.selectbox("Detected serial ports", options, key="dataset_serial_port")
            port = selected.split(" - ", 1)[0]
    else:
        replay_files = valid_workout_recordings()

        if replay_files:
            selected_name = st.selectbox(
                "Replay recording",
                [file.name for file in replay_files],
                key="dataset_replay_recording",
            )
            replay_csv = next(file for file in replay_files if file.name == selected_name)
            replay_realtime = st.checkbox(
                "Replay with original timing during cue preview",
                key="dataset_replay_realtime",
            )
        else:
            st.warning("No valid workout CSVs are available for replay.")

    return source_type, port, replay_csv, replay_realtime


def dataset_collection_metadata():
    st.subheader("Session Metadata")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        participant_id = st.text_input("Pseudonymous participant ID", key="dataset_participant")
        exercise = st.text_input("Exercise", value="bicep curl", key="dataset_exercise")
        muscle = st.text_input("Muscle", value="bicep", key="dataset_muscle")
        side = st.selectbox("Side", ["right", "left"], key="dataset_side")

    with col_b:
        weight = st.text_input("Weight", key="dataset_weight")
        planned_reps = st.number_input(
            "Planned repetitions",
            min_value=1,
            max_value=100,
            value=6,
            step=1,
            key="dataset_planned_reps",
        )
        body_position = st.text_input("Body position", value="seated", key="dataset_body_position")
        grip = st.text_input("Grip", value="supinated", key="dataset_grip")

    with col_c:
        placement_id = st.text_input("Placement ID", key="dataset_placement_id")
        calibration_setup_id = st.text_input("Sensor setup ID", key="dataset_calibration_setup_id")
        calibration_options = csv_options(
            compatible_calibration_recordings(muscle, side, calibration_setup_id)
            if calibration_setup_id.strip()
            else calibration_recordings()
        )
        calibration_csv = st.selectbox(
            "Calibration CSV",
            calibration_options,
            key="dataset_calibration_csv",
        )
        notes = st.text_area("Session notes", key="dataset_notes")

    if not calibration_setup_id.strip():
        st.warning(
            "Set a sensor setup ID and record a new calibration whenever MyoWare ENV gain or electrode setup changes."
        )

    st.subheader("Planned Cadence")
    cadence_cols = st.columns(4)
    cadence = {
        "seconds_up": cadence_cols[0].number_input("Seconds up", min_value=0.1, value=1.0, step=0.1),
        "hold_seconds": cadence_cols[1].number_input("Optional hold", min_value=0.0, value=0.0, step=0.1),
        "seconds_down": cadence_cols[2].number_input("Seconds down", min_value=0.1, value=2.0, step=0.1),
        "bottom_rest_seconds": cadence_cols[3].number_input("Bottom rest", min_value=0.0, value=1.0, step=0.1),
    }

    return {
        "participant_id": participant_id.strip(),
        "exercise": exercise.strip(),
        "muscle": muscle.strip(),
        "side": side,
        "weight": weight.strip(),
        "planned_reps": int(planned_reps),
        "body_position": body_position.strip(),
        "grip": grip.strip(),
        "placement_id": placement_id.strip(),
        "calibration_csv": calibration_csv,
        "calibration_setup_id": calibration_setup_id.strip(),
        "notes": notes.strip(),
        "cadence": cadence,
    }


def build_dataset_manifest(session_metadata, recording_csv, cue_schedule, recording_started_at):
    created_at = datetime.now()
    session_id = new_session_id(
        session_metadata["participant_id"],
        session_metadata["exercise"],
        created_at=created_at,
    )
    exercise_metadata = {
        "exercise": session_metadata["exercise"],
        "muscle": session_metadata["muscle"],
        "side": session_metadata["side"],
        "weight": session_metadata["weight"],
        "body_position": session_metadata["body_position"],
        "grip": session_metadata["grip"],
        "placement_id": session_metadata["placement_id"],
        "calibration_setup_id": session_metadata["calibration_setup_id"],
    }
    calibration_csv = selected_csv(session_metadata["calibration_csv"])
    return create_session_manifest(
        session_id=session_id,
        participant_id=session_metadata["participant_id"],
        recording_csv=recording_csv,
        calibration_csv=calibration_csv,
        exercise_metadata=exercise_metadata,
        planned_reps=session_metadata["planned_reps"],
        cadence=session_metadata["cadence"],
        cue_timestamps=cue_schedule,
        recording_started_at=recording_started_at,
        notes=session_metadata["notes"],
    )


def show_dataset_collection():
    st.header("Dataset Builder")
    st.caption(
        "Collect cue-assisted recordings and create manifests for later human labeling. "
        "Planned cues are stored as timing cues only, not as verified rep boundaries."
    )
    source_type, port, replay_csv, replay_realtime = dataset_source_controls()
    session_metadata = dataset_collection_metadata()
    cue_schedule = planned_cue_schedule(session_metadata["planned_reps"], session_metadata["cadence"])
    duration_seconds = planned_total_duration(session_metadata["planned_reps"], session_metadata["cadence"])

    metric_row([
        ("Planned duration", f"{duration_seconds:.1f}s"),
        ("Planned reps", session_metadata["planned_reps"]),
        ("Cue count", len(cue_schedule)),
    ])

    with st.expander("Cue schedule"):
        st.table(cue_schedule)

    missing_participant = not session_metadata["participant_id"]
    missing_source = source_type == "Serial hardware" and port is None
    missing_source = missing_source or (source_type == "Replay CSV" and replay_csv is None)

    if missing_participant:
        st.warning("Use a pseudonymous participant ID. Do not enter names or contact details.")

    if source_type == "Replay CSV":
        if st.button(
            "Create Dataset Session From Replay",
            disabled=missing_participant or missing_source,
            key="dataset_create_replay_session",
        ):
            try:
                manifest = build_dataset_manifest(
                    session_metadata,
                    replay_csv,
                    cue_schedule,
                    recording_started_at="replay_existing_csv",
                )
                manifest_file, annotation_file = save_dataset_session(manifest)
            except Exception as error:  # noqa: BLE001
                st.error(f"Dataset session creation failed: {error}")
                return

            st.success(f"Saved dataset session: {manifest['session_id']}")
            st.write(f"Manifest: {manifest_file.resolve()}")
            st.write(f"Annotations: {annotation_file.resolve()}")
        return

    if not st.button(
        "Start Cue-Assisted Recording",
        disabled=missing_participant or missing_source,
        key="dataset_start_recording",
    ):
        return

    countdown = st.empty()
    for seconds_remaining in range(3, 0, -1):
        countdown.warning(f"Recording begins in {seconds_remaining}")
        time.sleep(1)
    countdown.success("Recording started")

    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now()
    output_file = unique_recording_path(session_metadata["exercise"], timestamp)
    metadata = {
        "exercise_name": session_metadata["exercise"],
        "muscle": session_metadata["muscle"],
        "side": session_metadata["side"],
        "weight": session_metadata["weight"],
        "expected_reps": "",
        "planned_reps": str(session_metadata["planned_reps"]),
        "data_type": "real",
        "test_type": "dataset_collection",
        "notes": session_metadata["notes"],
        "csv_filename": output_file.name,
        "timestamp": timestamp.isoformat(timespec="seconds"),
    }
    source = source_from_selection(source_type, port, None, False)
    progress_bar = st.progress(0.0)

    try:
        recording = write_dataset_recording(
            source,
            output_file,
            metadata,
            duration_seconds,
            cue_schedule,
            progress_bar,
        )
    except Exception as error:  # noqa: BLE001
        st.error(f"Recording failed: {error}")
        return

    if recording["malformed_reads"]:
        st.warning(f"Skipped {recording['malformed_reads']} malformed or empty readings.")

    if not recording["readings"]:
        st.error("No valid readings were saved, so the dataset session was not created.")
        return

    try:
        manifest = build_dataset_manifest(
            session_metadata,
            output_file,
            cue_schedule,
            recording_started_at=timestamp.isoformat(timespec="seconds"),
        )
        manifest_file, annotation_file = save_dataset_session(manifest)
    except Exception as error:  # noqa: BLE001
        st.error(f"Dataset session creation failed: {error}")
        return

    st.success(f"Saved dataset session: {manifest['session_id']}")
    st.write(f"Raw recording: {output_file.resolve()}")
    st.write(f"Manifest: {manifest_file.resolve()}")
    st.write(f"Annotations: {annotation_file.resolve()}")


def session_label(session):
    metadata = session.get("exercise_metadata", {})
    return (
        f"{session.get('created_at', '')} - {session.get('participant_id', '')} - "
        f"{metadata.get('exercise', '')} - {session.get('session_id', '')}"
    )


def default_annotation_rows(annotations):
    rows = [
        {
            "rep_number": interval.get("rep_number", index),
            "start_time": interval.get("start_time", 0.0),
            "end_time": interval.get("end_time", 0.0),
            "confidence": interval.get("confidence", ""),
            "note": interval.get("note", ""),
        }
        for index, interval in enumerate(annotations.get("verified_rep_intervals", []), start=1)
    ]

    return rows or [{
        "rep_number": 1,
        "start_time": 0.0,
        "end_time": 0.0,
        "confidence": "",
        "note": "",
    }]


def default_false_interval_rows(annotations):
    rows = [
        {
            "start_time": interval.get("start_time", 0.0),
            "end_time": interval.get("end_time", 0.0),
            "confidence": interval.get("confidence", ""),
            "note": interval.get("note", ""),
        }
        for interval in annotations.get("excluded_false_intervals", [])
    ]

    return rows or [{
        "start_time": 0.0,
        "end_time": 0.0,
        "confidence": "",
        "note": "",
    }]


def remove_blank_editor_rows(rows):
    cleaned = []

    for row in rows:
        start_time = row.get("start_time", 0)
        end_time = row.get("end_time", 0)
        confidence = row.get("confidence", "")
        note = row.get("note", "")

        if (
            start_time in ("", 0, 0.0, None)
            and end_time in ("", 0, 0.0, None)
            and confidence in ("", None)
            and note in ("", None)
        ):
            continue

        cleaned.append(row)

    return cleaned


def show_dataset_annotation():
    st.header("Annotate Dataset Session")
    sessions = list_dataset_sessions()

    if not sessions:
        st.info("No dataset sessions have been created yet.")
        return

    labels = [session_label(session) for session in sessions]
    selected_label = st.selectbox("Dataset session", labels, key="dataset_annotation_session")
    manifest = sessions[labels.index(selected_label)]
    annotations = load_annotations(manifest["session_id"])
    csv_file = resolve_repo_path(manifest["recording_csv"])

    if annotations.get("annotation_status") == "locked":
        st.warning("This annotation is locked. Unlock it in the JSON file only if you intentionally need to revise it.")

    metric_row([
        ("Session ID", manifest["session_id"]),
        ("Planned reps", manifest.get("planned_reps", "")),
        ("Actual reps", annotations.get("actual_reps") or "Not set"),
        ("Status", annotations.get("annotation_status", "unreviewed")),
    ])

    try:
        figure = annotation_figure(manifest, annotations)
        st.pyplot(figure)
        plt.close(figure)
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not render annotation plot: {error}")

    st.subheader("Verified Repetitions")
    edited_rows = st.data_editor(
        default_annotation_rows(annotations),
        num_rows="dynamic",
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_rep_editor_{manifest['session_id']}",
    )

    st.subheader("False Detector Intervals")
    default_false_rows = default_false_interval_rows(annotations)
    false_rows = st.data_editor(
        default_false_rows,
        num_rows="dynamic",
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_false_editor_{manifest['session_id']}",
    )

    status = st.selectbox(
        "Annotation status",
        ["unreviewed", "reviewed", "locked"],
        index=["unreviewed", "reviewed", "locked"].index(
            annotations.get("annotation_status", "unreviewed")
        ),
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_annotation_status_{manifest['session_id']}",
    )
    actual_reps = st.number_input(
        "Actual reps (optional)",
        min_value=0,
        value=int(annotations.get("actual_reps") or 0),
        step=1,
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_actual_reps_{manifest['session_id']}",
    )
    actual_reps_is_set = st.checkbox(
        "Store actual reps value",
        value=annotations.get("actual_reps") is not None,
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_actual_reps_is_set_{manifest['session_id']}",
    )
    confidence = st.slider(
        "Overall confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(annotations.get("confidence") or 0.0),
        step=0.05,
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_confidence_{manifest['session_id']}",
    )
    notes = st.text_area(
        "Annotation notes",
        value=annotations.get("notes", ""),
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_annotation_notes_{manifest['session_id']}",
    )

    if not st.button(
        "Save Annotations",
        disabled=annotations.get("annotation_status") == "locked",
        key=f"dataset_save_annotations_{manifest['session_id']}",
    ):
        return

    try:
        max_time = recording_duration(csv_file)
        errors, normalized_rows = validate_annotation_rows(
            remove_blank_editor_rows(edited_rows),
            max_time,
        )
        false_errors, normalized_false_rows = validate_annotation_rows(
            remove_blank_editor_rows(false_rows),
            max_time,
            require_rep_numbers=False,
        )
    except Exception as error:  # noqa: BLE001
        st.error(f"Validation failed: {error}")
        return

    errors.extend(error.replace("Rep", "False interval") for error in false_errors)

    if errors:
        for error in errors:
            st.error(error)
        return

    updated = {
        "schema_version": annotations.get("schema_version", "boundary_annotations_v1"),
        "annotation_status": status,
        "actual_reps": actual_reps if actual_reps_is_set else None,
        "verified_rep_intervals": normalized_rows,
        "excluded_false_intervals": normalized_false_rows,
        "confidence": confidence,
        "notes": notes,
        "last_modified": annotations.get("last_modified", ""),
    }
    try:
        annotation_file = save_annotations(manifest["session_id"], updated)
    except Exception as error:  # noqa: BLE001
        st.error(f"Annotations were not saved: {error}")
        return

    st.success(f"Saved annotations: {annotation_file.resolve()}")


def show_dataset_builder():
    collection_tab, annotation_tab = st.tabs(["Collection", "Annotation"])

    with collection_tab:
        show_dataset_collection()

    with annotation_tab:
        show_dataset_annotation()


def main():
    st.set_page_config(page_title="RepAI", layout="wide")
    st.title("RepAI")

    overview_tab, session_tab, side_tab, ladder_tab, dataset_tab = st.tabs([
        "Overview",
        "Workout Session",
        "Side Comparison",
        "Weight Ladder",
        "Dataset Builder",
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

    with dataset_tab:
        show_dataset_builder()


if __name__ == "__main__":
    main()
