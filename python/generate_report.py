from datetime import datetime
from html import escape
from os.path import relpath
from pathlib import Path

from calibration_utils import load_latest_calibration
from recording_metadata import load_metadata


BASE_DIR = Path(__file__).resolve().parent.parent
GRAPHS_DIR = BASE_DIR / "graphs"
SUMMARIES_DIR = BASE_DIR / "summaries"
REPORTS_DIR = BASE_DIR / "reports"
DATA_DIR = BASE_DIR / "data"

COMPARISON_FILE = SUMMARIES_DIR / "latest_set_comparison.txt"
COMPARISON_GRAPH = GRAPHS_DIR / "latest_set_comparison.png"
REPORT_FILE = REPORTS_DIR / "latest_report.html"
INDEX_FILE = REPORTS_DIR / "index.html"


def read_text_file(path):
    if not path.exists():
        return ""

    return path.read_text(encoding="utf-8").strip()


def split_comparison_sections(comparison_text):
    if not comparison_text:
        return "", ""

    marker = "\nUser Insights\n"

    if marker not in comparison_text:
        return comparison_text, ""

    comparison_section, insights_section = comparison_text.split(marker, 1)
    return comparison_section.strip(), insights_section.strip()


def latest_rep_graph():
    rep_graphs = list(GRAPHS_DIR.glob("*_reps.png"))

    if not rep_graphs:
        return None

    return max(rep_graphs, key=lambda path: path.stat().st_mtime)


def sorted_files(folder, pattern):
    if not folder.exists():
        return []

    return sorted(
        folder.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def format_datetime(timestamp):
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %I:%M %p")


def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"

    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"

    return f"{size_bytes / (1024 * 1024):.1f} MB"


def graph_path_for_html(path):
    return relpath(path, REPORTS_DIR).replace("\\", "/")


def text_block(text, empty_message):
    if not text:
        return f"<p class=\"muted\">{escape(empty_message)}</p>"

    return f"<pre>{escape(text)}</pre>"


def calibration_block(calibration):
    if calibration is None:
        return '<p class="muted">No calibration file found. Normalized activation is not available yet.</p>'

    usable_text = "yes" if calibration.get("usable") else "no"
    lines = [
        "Calibration Available",
        f"Source CSV: {calibration.get('source_csv', 'unknown')}",
        f"Baseline: {calibration.get('baseline', 0):.1f}",
        f"Max flex: {calibration.get('max_flex', 0):.1f}",
        f"Signal range: {calibration.get('signal_range', 0):.1f}",
        f"Usable calibration: {usable_text}",
        "Note: full-recording averages include rest time.",
        "Note: normalized activation can exceed 100% when a workout exceeds the calibration max.",
    ]

    return text_block("\n".join(lines), "No calibration file found.")


def graph_card(title, path):
    if path is None or not path.exists():
        return ""

    return f"""
        <article class="graph-card">
          <h3>{escape(title)}</h3>
          <img src="{escape(graph_path_for_html(path))}" alt="{escape(title)}">
        </article>
    """


def page_css():
    return """    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --card: #ffffff;
      --text: #17202a;
      --muted: #68717d;
      --border: #d9e0e7;
      --accent: #1769aa;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }

    main {
      max-width: 980px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }

    header {
      margin-bottom: 24px;
    }

    h1 {
      margin: 0 0 6px;
      font-size: 2rem;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 1.25rem;
      color: var(--accent);
    }

    h3 {
      margin: 0 0 12px;
      font-size: 1rem;
    }

    .generated {
      margin: 0;
      color: var(--muted);
    }

    section,
    .graph-card,
    .stat-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 3px rgba(23, 32, 42, 0.08);
      margin: 18px 0;
      padding: 22px;
    }

    pre {
      margin: 0;
      white-space: pre-wrap;
      font: 0.95rem/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }

    .muted {
      color: var(--muted);
      margin: 0;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin: 18px 0;
    }

    .stat-card {
      margin: 0;
    }

    .stat-value {
      display: block;
      font-size: 1.8rem;
      font-weight: 700;
    }

    .stat-label {
      color: var(--muted);
      font-size: 0.9rem;
    }

    .graphs {
      display: grid;
      gap: 18px;
    }

    img {
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }

    th,
    td {
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }

    th {
      background: #eef3f7;
      font-weight: 700;
    }

    tr:nth-child(even) td {
      background: #fafbfd;
    }

    .table-wrap {
      overflow-x: auto;
    }

    .recording-groups {
      display: grid;
      gap: 16px;
    }

    .recording-group {
      border-top: 1px solid var(--border);
      padding-top: 16px;
    }
"""


def page_shell(title, body_html):
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
{page_css()}
  </style>
</head>
<body>
  <main>
{body_html}
  </main>
</body>
</html>
"""


def build_latest_report(comparison_section, insights_section, rep_graph, calibration):
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    comparison_graph = COMPARISON_GRAPH if COMPARISON_GRAPH.exists() else None

    body_html = f"""
    <header>
      <h1>RepAI Report</h1>
      <p class="generated">Generated {escape(generated_at)}</p>
    </header>

    <section>
      <h2>Latest Set Comparison</h2>
      {text_block(comparison_section, "No latest set comparison was found.")}
    </section>

    <section>
      <h2>User Insights</h2>
      {text_block(insights_section, "No user insights were found.")}
    </section>

    <section>
      <h2>Calibration</h2>
      {calibration_block(calibration)}
    </section>

    <section>
      <h2>Graphs</h2>
      <div class="graphs">
        {graph_card("Latest Set Comparison", comparison_graph)}
        {graph_card("Latest Rep Detection", rep_graph)}
        {"" if comparison_graph or rep_graph else '<p class="muted">No graph images were found.</p>'}
      </div>
    </section>
"""

    return page_shell("RepAI Report", body_html)


def stat_card(label, value):
    return f"""
        <div class="stat-card">
          <span class="stat-value">{escape(str(value))}</span>
          <span class="stat-label">{escape(label)}</span>
        </div>
    """


def csv_row(csv_file, include_data_type=True):
    stats = csv_file.stat()
    metadata = load_metadata(csv_file)
    data_type_cell = (
        f"<td>{escape(metadata.get('data_type', ''))}</td>"
        if include_data_type
        else ""
    )

    return (
        "<tr>"
        f"<td>{escape(csv_file.name)}</td>"
        f"<td>{escape(metadata.get('exercise_name', ''))}</td>"
        f"<td>{escape(metadata.get('muscle', ''))}</td>"
        f"<td>{escape(metadata.get('side', ''))}</td>"
        f"<td>{escape(metadata.get('weight', ''))}</td>"
        f"{data_type_cell}"
        f"<td>{escape(metadata.get('test_type', ''))}</td>"
        f"<td>{escape(format_datetime(stats.st_mtime))}</td>"
        f"<td>{escape(format_file_size(stats.st_size))}</td>"
        "</tr>"
    )


def csv_table(csv_files, include_data_type=True):
    if not csv_files:
        return '<p class="muted">No CSV recordings were found.</p>'

    rows = [csv_row(csv_file, include_data_type) for csv_file in csv_files]
    data_type_header = "<th>Data Type</th>" if include_data_type else ""

    return f"""
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Filename</th>
              <th>Exercise</th>
              <th>Muscle</th>
              <th>Side</th>
              <th>Weight</th>
              {data_type_header}
              <th>Test Type</th>
              <th>Modified</th>
              <th>File Size</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </div>
    """


def recording_category(csv_file):
    metadata = load_metadata(csv_file)
    data_type = metadata.get("data_type", "")
    test_type = metadata.get("test_type", "")

    if data_type == "real" and test_type == "workout_set":
        return "real_workouts"

    if data_type == "real" and test_type in ("calibration", "flex_test"):
        return "real_tests"

    if data_type == "fake" and test_type == "workout_set":
        return "fake_workouts"

    if data_type == "fake" and test_type in ("calibration", "flex_test"):
        return "fake_tests"

    return "unknown"


def categorized_recordings(csv_files):
    categories = {
        "real_workouts": [],
        "real_tests": [],
        "fake_workouts": [],
        "fake_tests": [],
        "unknown": [],
    }

    for csv_file in csv_files:
        categories[recording_category(csv_file)].append(csv_file)

    return categories


def organized_recording_sections(csv_files):
    categories = categorized_recordings(csv_files)
    labels = [
        ("real_workouts", "Real Workout Sets"),
        ("real_tests", "Real Calibration / Flex Tests"),
        ("fake_workouts", "Fake Workout Sets"),
        ("fake_tests", "Fake Calibration / Flex Tests"),
        ("unknown", "Unknown / Missing Metadata"),
    ]
    sections = []

    for key, label in labels:
        files = categories[key]
        sections.append(
            f"""
        <div class="recording-group">
          <h3>{escape(label)} ({len(files)})</h3>
          {csv_table(files, include_data_type=False)}
        </div>
            """
        )

    return "\n".join(sections)


def recent_graph_cards(graph_files, limit=6):
    recent_graphs = graph_files[:limit]

    if not recent_graphs:
        return '<p class="muted">No graph images were found.</p>'

    return "\n".join(
        graph_card(graph_file.name, graph_file)
        for graph_file in recent_graphs
    )


def build_index(comparison_text, calibration):
    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    csv_files = sorted_files(DATA_DIR, "*.csv")
    graph_files = sorted_files(GRAPHS_DIR, "*.png")
    summary_files = sorted_files(SUMMARIES_DIR, "*.txt")

    body_html = f"""
    <header>
      <h1>RepAI Dashboard</h1>
      <p class="generated">Generated {escape(generated_at)}</p>
    </header>

    <div class="stats">
      {stat_card("CSV recordings", len(csv_files))}
      {stat_card("Graph images", len(graph_files))}
      {stat_card("Summary files", len(summary_files))}
    </div>

    <section>
      <h2>Latest Comparison</h2>
      {text_block(comparison_text, "No latest set comparison was found.")}
    </section>

    <section>
      <h2>Calibration</h2>
      {calibration_block(calibration)}
    </section>

    <section>
      <h2>Recorded Sets</h2>
      <div class="recording-groups">
        {organized_recording_sections(csv_files)}
      </div>
    </section>

    <section>
      <h2>All Recorded Sets</h2>
      {csv_table(csv_files)}
    </section>

    <section>
      <h2>Graphs</h2>
      <div class="graphs">
        {recent_graph_cards(graph_files)}
      </div>
    </section>
"""

    return page_shell("RepAI Dashboard", body_html)


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    comparison_text = read_text_file(COMPARISON_FILE)
    calibration = load_latest_calibration()
    comparison_section, insights_section = split_comparison_sections(comparison_text)
    rep_graph = latest_rep_graph()
    latest_report_html = build_latest_report(
        comparison_section,
        insights_section,
        rep_graph,
        calibration,
    )
    index_html = build_index(comparison_text, calibration)

    REPORT_FILE.write_text(latest_report_html, encoding="utf-8")
    INDEX_FILE.write_text(index_html, encoding="utf-8")
    print(f"Generated latest report: {REPORT_FILE.resolve()}")
    print(f"Generated dashboard: {INDEX_FILE.resolve()}")


if __name__ == "__main__":
    main()
