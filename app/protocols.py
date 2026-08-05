import json
import re
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
SESSIONS_DIR = BASE_DIR / "app" / "protocol_sessions"


SIDE_COMPARISON_SLOTS = [
    {"key": "right_set_1", "label": "Right Set 1", "side": "right"},
    {"key": "right_set_2", "label": "Right Set 2", "side": "right"},
    {"key": "left_set_1", "label": "Left Set 1", "side": "left"},
    {"key": "left_set_2", "label": "Left Set 2", "side": "left"},
]


def slugify(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "session"


def session_path(session_id):
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(session):
    SESSIONS_DIR.mkdir(exist_ok=True)

    with open(session_path(session["session_id"]), "w", encoding="utf-8") as file:
        json.dump(session, file, indent=2)
        file.write("\n")


def load_sessions(protocol_type=None):
    if not SESSIONS_DIR.exists():
        return []

    sessions = []

    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        with open(path, "r", encoding="utf-8") as file:
            session = json.load(file)

        if protocol_type is None or session.get("protocol_type") == protocol_type:
            sessions.append(session)

    return sessions


def new_side_comparison_session(values):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"side_comparison_{now}_{slugify(values['exercise'])}"
    return {
        "schema_version": 1,
        "protocol_type": "single_sensor_side_comparison",
        "session_id": session_id,
        "created_at": now,
        "exercise": values["exercise"],
        "muscle": values["muscle"],
        "weight": values["weight"],
        "expected_reps": values["expected_reps"],
        "cadence_notes": values["cadence_notes"],
        "testing_order": values["testing_order"],
        "placement_notes": values["placement_notes"],
        "session_notes": values["session_notes"],
        "calibrations": {
            "right": values.get("right_calibration_csv", ""),
            "left": values.get("left_calibration_csv", ""),
        },
        "recordings": {slot["key"]: "" for slot in SIDE_COMPARISON_SLOTS},
    }


def new_weight_ladder_session(values):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"weight_ladder_{now}_{slugify(values['exercise'])}"
    weights = [weight.strip() for weight in values["weights"] if weight.strip()]
    return {
        "schema_version": 1,
        "protocol_type": "weight_ladder",
        "session_id": session_id,
        "created_at": now,
        "exercise": values["exercise"],
        "muscle": values["muscle"],
        "side": values["side"],
        "expected_reps": values["expected_reps"],
        "rest_time_notes": values["rest_time_notes"],
        "session_notes": values["session_notes"],
        "calibration_csv": values.get("calibration_csv", ""),
        "weight_levels": [
            {"weight": weight, "recording_csv": ""}
            for weight in weights
        ],
    }
