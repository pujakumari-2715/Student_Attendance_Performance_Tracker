"""
PS 46 - Student Attendance & Performance Tracker
================================================
One file, two ways to run it:

  Console menu (PS 46 test cases):  python student_app_ps46.py students.csv
  Web dashboard (Streamlit):        streamlit run student_app_ps46.py

CSV columns : Name, Classes_Held, Classes_Attended, Marks (out of 100)
              (a header row is optional)
At Risk     : attendance below 75%  OR  marks below 40
Needs       : pandas, numpy, matplotlib  (+ streamlit for the web dashboard)
"""

import argparse
import io
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# SETTINGS
# ----------------------------------------------------------------------
ATTENDANCE_THRESHOLD = 75.0     # At Risk if attendance % is below this
MARKS_THRESHOLD = 40.0          # At Risk if marks are below this
WATCH_MARGIN = 5.0              # "Watch list" = OK students within 5 points of a limit
COLUMNS = ["Name", "Classes_Held", "Classes_Attended", "Marks"]
CHARTS_MESSAGE = "Bar chart and scatter chart 'Attendance vs Marks' displayed"
NO_DATA_MESSAGE = "No valid data to analyse."
RED, GREEN = "#C0392B", "#27AE60"

# The four test cases from the PS 46 sheet (used by the in-app Test Cases tab)
TC1_CSV = "Aman,40,36,80\nRiya,40,28,65\nKaran,40,30,35\n"
TEST_CASES = [
    {"name": "Test Case 1: Normal Input", "csv": TC1_CSV, "option": 1,
     "menu": "Menu option 1 (Report)",
     "expected": ["Aman: 90.00% | Marks 80 | OK",
                  "Riya: 70.00% | Marks 65 | At Risk (attendance below 75%)",
                  "Karan: 75.00% | Marks 35 | At Risk (marks below 40)"]},
    {"name": "Test Case 2: Boundary / Minimum Input", "csv": "Neha,40,30,40\n", "option": 1,
     "menu": "Menu option 1 (Report)",
     "expected": ["Neha: 75.00% | Marks 40 | OK"]},
    {"name": "Test Case 3: Invalid Input", "csv": "Aman,40,45,80\nRiya,40,30,abc\n", "option": None,
     "menu": "(no menu - nothing valid to analyse)",
     "expected": ["Row 1 skipped: attended (45) cannot exceed held (40).",
                  "Row 2 skipped: invalid Marks 'abc'.",
                  "No valid data to analyse."]},
    {"name": "Test Case 4: Different Valid Scenario", "csv": TC1_CSV, "option": 2,
     "menu": "Menu option 2 (Summary and charts)",
     "expected": ["Average attendance: 78.33% | Average marks: 60.00",
                  "At Risk students: 2 (Riya, Karan)",
                  "Bar chart and scatter chart 'Attendance vs Marks' displayed"]},
]


# ----------------------------------------------------------------------
# 1. LOAD & CLEAN  (pandas)
# ----------------------------------------------------------------------
def load_and_clean(source):
    """Read the CSV with pandas, validate every row and return (clean_df, messages).

    Invalid rows are skipped and explained in `messages` (row numbers start at 1
    and do not count the optional header row).
    """
    empty = pd.DataFrame(columns=COLUMNS)
    messages = []
    try:
        raw = pd.read_csv(source, header=None, dtype=str, skipinitialspace=True,
                          keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return empty, messages
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        return empty, [f"Could not read the CSV file: {exc}"]

    # optional header row -> map columns by name, otherwise assume the standard order
    first = [str(v).strip().lower().replace(" ", "_") for v in raw.iloc[0]]
    if first[0] == "name":
        names, seen = [], set()
        for i, h in enumerate(first):
            col = next((c for c in COLUMNS if h.startswith(c.lower())), None)
            if col is None or col in seen:
                col = f"_extra{i}"
            else:
                seen.add(col)
            names.append(col)
        missing = [c for c in COLUMNS if c not in seen]
        if missing:
            return empty, [f"Missing required column(s): {', '.join(missing)}."]
        raw.columns = names
        raw = raw.iloc[1:]
    else:
        if raw.shape[1] < 4:
            return empty, ["Expected 4 columns: Name, Classes_Held, Classes_Attended, Marks."]
        raw = raw.iloc[:, :4]
        raw.columns = COLUMNS
    raw = raw[COLUMNS]

    def text(v):
        return v.strip() if isinstance(v, str) else ""

    def number(s):
        try:
            x = float(s)
        except ValueError:
            return None
        return x if np.isfinite(x) else None

    rows = []
    for n, rec in enumerate(raw.itertuples(index=False), start=1):
        name, held_s, att_s, marks_s = (text(v) for v in rec)
        if not name:
            messages.append(f"Row {n} skipped: missing Name.")
            continue
        held, att, marks = number(held_s), number(att_s), number(marks_s)

        problem = None
        for label, value, shown, whole in (("Classes_Held", held, held_s, True),
                                           ("Classes_Attended", att, att_s, True),
                                           ("Marks", marks, marks_s, False)):
            if value is None or (whole and value != int(value)):
                problem = f"invalid {label} '{shown}'"
                break
        if problem is None:
            if held <= 0:
                problem = f"Classes_Held must be greater than 0 (got {held:g})"
            elif att < 0:
                problem = f"Classes_Attended cannot be negative ({att:g})"
            elif att > held:
                problem = f"attended ({att:g}) cannot exceed held ({held:g})"
            elif not 0 <= marks <= 100:
                problem = f"Marks ({marks:g}) must be between 0 and 100"
        if problem:
            messages.append(f"Row {n} skipped: {problem}.")
            continue
        rows.append((name, int(held), int(att), marks))

    return pd.DataFrame(rows, columns=COLUMNS), messages


# ----------------------------------------------------------------------
# 2. CALCULATIONS  (pandas columns + boolean filtering)
# ----------------------------------------------------------------------
def add_metrics(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    """Add attendance %, risk flags, status and the reason for each student."""
    df = df.copy()
    df["Attendance_Pct"] = df["Classes_Attended"] * 100 / df["Classes_Held"]
    df["Low_Attendance"] = df["Attendance_Pct"] < att_thr
    df["Low_Marks"] = df["Marks"] < marks_thr
    df["At_Risk"] = df["Low_Attendance"] | df["Low_Marks"]
    df["Status"] = np.where(df["At_Risk"], "At Risk", "OK")
    df["Reason"] = np.select(
        [df["Low_Attendance"] & df["Low_Marks"], df["Low_Attendance"], df["Low_Marks"]],
        [f"attendance below {att_thr:g}% and marks below {marks_thr:g}",
         f"attendance below {att_thr:g}%",
         f"marks below {marks_thr:g}"],
        default="")
    return df


def report_lines(df):
    """Menu option 1 - one line per student."""
    lines = []
    for r in df.itertuples(index=False):
        status = f"At Risk ({r.Reason})" if r.At_Risk else "OK"
        lines.append(f"{r.Name}: {r.Attendance_Pct:.2f}% | Marks {r.Marks:g} | {status}")
    return lines


def summary_lines(df):
    """Menu option 2 - averages and At Risk list (the charts are drawn separately)."""
    at_risk = df.loc[df["At_Risk"], "Name"].tolist()
    risk_text = f"At Risk students: {len(at_risk)}"
    if at_risk:
        risk_text += f" ({', '.join(at_risk)})"
    return [f"Average attendance: {df['Attendance_Pct'].mean():.2f}% | "
            f"Average marks: {df['Marks'].mean():.2f}", risk_text]


def simulate(csv_text, option):
    """Run a menu option on CSV text and return the output lines (used for testing)."""
    df, msgs = load_and_clean(io.StringIO(csv_text))
    out = list(msgs)
    if df.empty:
        return out + [NO_DATA_MESSAGE]
    df = add_metrics(df)
    if option == 1:
        out += report_lines(df)
    elif option == 2:
        out += summary_lines(df) + [CHARTS_MESSAGE]
    return out


# ----------------------------------------------------------------------
# 3. KPI METRICS
# ----------------------------------------------------------------------
def compute_kpis(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    n = len(df)
    low_att, low_marks, risk = df["Low_Attendance"], df["Low_Marks"], df["At_Risk"]
    top, low = df.loc[df["Marks"].idxmax()], df.loc[df["Marks"].idxmin()]
    best, worst = df.loc[df["Attendance_Pct"].idxmax()], df.loc[df["Attendance_Pct"].idxmin()]
    corr = df["Attendance_Pct"].corr(df["Marks"]) if n >= 3 else np.nan
    watch = df[~risk & ((df["Attendance_Pct"] < att_thr + WATCH_MARGIN)
                        | (df["Marks"] < marks_thr + WATCH_MARGIN))]
    return {
        "students": n,
        "avg_attendance": df["Attendance_Pct"].mean(),
        "median_attendance": df["Attendance_Pct"].median(),
        "pooled_attendance": df["Classes_Attended"].sum() * 100 / df["Classes_Held"].sum(),
        "avg_marks": df["Marks"].mean(),
        "median_marks": df["Marks"].median(),
        "std_marks": df["Marks"].std(ddof=0),
        "at_risk": int(risk.sum()),
        "at_risk_pct": risk.mean() * 100,
        "att_only": int((low_att & ~low_marks).sum()),
        "marks_only": int((~low_att & low_marks).sum()),
        "both": int((low_att & low_marks).sum()),
        "compliance": (~low_att).mean() * 100,
        "pass_rate": (~low_marks).mean() * 100,
        "correlation": None if np.isnan(corr) else float(corr),
        "top_scorer": (top["Name"], top["Marks"]),
        "lowest_scorer": (low["Name"], low["Marks"]),
        "best_attendance": (best["Name"], best["Attendance_Pct"]),
        "lowest_attendance": (worst["Name"], worst["Attendance_Pct"]),
        "watch_list": watch["Name"].tolist(),
    }


def corr_strength(r):
    a = abs(r)
    if a < 0.2:
        return "negligible"
    word = "very strong" if a >= .8 else "strong" if a >= .6 else "moderate" if a >= .4 else "weak"
    return f"{word} {'positive' if r > 0 else 'negative'}"


def insights(k, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    """Plain-English findings generated from the KPIs."""
    out = []
    if k["at_risk"] == 0:
        out.append("No student is At Risk.")
    else:
        out.append(f"{k['at_risk']} of {k['students']} students ({k['at_risk_pct']:.1f}%) are At Risk.")
        by_att, by_marks = k["att_only"] + k["both"], k["marks_only"] + k["both"]
        if by_att > by_marks:
            out.append(f"Attendance is the bigger problem: {by_att} students are below "
                       f"{att_thr:g}% vs {by_marks} below {marks_thr:g} marks.")
        elif by_marks > by_att:
            out.append(f"Marks are the bigger problem: {by_marks} students are below "
                       f"{marks_thr:g} vs {by_att} below {att_thr:g}% attendance.")
        else:
            out.append(f"Attendance and marks contribute equally ({by_att} students each).")
        if k["both"]:
            out.append(f"{k['both']} student(s) fail BOTH rules - highest priority for support.")
    if k["correlation"] is not None:
        out.append(f"Attendance and marks show a {corr_strength(k['correlation'])} relationship "
                   f"(r = {k['correlation']:.2f}).")
    if k["watch_list"]:
        out.append(f"Watch list: {', '.join(k['watch_list'])} are OK now but within "
                   f"{WATCH_MARGIN:g} points of a limit.")
    return out


def kpi_lines(k, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    corr = ("n/a (needs at least 3 students)" if k["correlation"] is None
            else f"r = {k['correlation']:.2f} ({corr_strength(k['correlation'])})")
    rows = [
        ("Students analysed", k["students"]),
        ("Average attendance", f"{k['avg_attendance']:.2f}% (median {k['median_attendance']:.2f}%)"),
        ("Overall attendance (pooled)", f"{k['pooled_attendance']:.2f}%"),
        ("Average marks", f"{k['avg_marks']:.2f} (median {k['median_marks']:.2f}, "
                          f"std {k['std_marks']:.2f})"),
        ("At Risk students", f"{k['at_risk']} of {k['students']} ({k['at_risk_pct']:.1f}%)"),
        ("  - low attendance only", k["att_only"]),
        ("  - low marks only", k["marks_only"]),
        ("  - both", k["both"]),
        (f"Attendance compliance (>={att_thr:g}%)", f"{k['compliance']:.1f}% of students"),
        (f"Pass rate (marks >= {marks_thr:g})", f"{k['pass_rate']:.1f}% of students"),
        ("Attendance-marks correlation", corr),
        ("Top scorer", f"{k['top_scorer'][0]} ({k['top_scorer'][1]:g})"),
        ("Lowest scorer", f"{k['lowest_scorer'][0]} ({k['lowest_scorer'][1]:g})"),
        ("Best attendance", f"{k['best_attendance'][0]} ({k['best_attendance'][1]:.2f}%)"),
        ("Lowest attendance", f"{k['lowest_attendance'][0]} ({k['lowest_attendance'][1]:.2f}%)"),
        ("Watch list (OK, near a limit)", ", ".join(k["watch_list"]) or "none"),
    ]
    lines = ["===== KPI DASHBOARD ====="] + [f"{a:<34}: {b}" for a, b in rows]
    lines += ["", "Key insights:"] + [f" * {t}" for t in insights(k, att_thr, marks_thr)]
    return lines


# ----------------------------------------------------------------------
# 4. CHARTS  (matplotlib) - every function returns a Figure
# ----------------------------------------------------------------------
def _legend_handles(thr_label=None):
    h = [Patch(color=GREEN, label="OK"), Patch(color=RED, label="At Risk")]
    if thr_label:
        h.append(Line2D([0], [0], color="black", ls="--", label=thr_label))
    return h


def _bar(df, col, thr, thr_label, title, ylabel):
    n = len(df)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(min(20, max(7, 0.4 * n)), 5))
    ax.bar(x, df[col], color=np.where(df["At_Risk"], RED, GREEN))
    ax.axhline(thr, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Name"], rotation=90 if n > 12 else 0, fontsize=7 if n > 40 else 9)
    if n <= 15:
        for xi, v in zip(x, df[col]):
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set(title=title, xlabel="Student", ylabel=ylabel, ylim=(0, 120))
    ax.legend(handles=_legend_handles(thr_label), loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig


def make_bar_chart(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    return _bar(df, "Attendance_Pct", att_thr, f"Minimum attendance ({att_thr:g}%)",
                "Attendance Percentage by Student", "Attendance (%)")


def make_marks_bar(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    return _bar(df, "Marks", marks_thr, f"Pass mark ({marks_thr:g})",
                "Marks by Student", "Marks (out of 100)")


def make_scatter_chart(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["Attendance_Pct"], df["Marks"], c=np.where(df["At_Risk"], RED, GREEN),
               s=80, edgecolor="black", zorder=3)
    ax.axvline(att_thr, color="black", ls="--", lw=1)
    ax.axhline(marks_thr, color="black", ls="--", lw=1)
    lo_x = max(0, min(df["Attendance_Pct"].min(), att_thr) - 10)
    lo_y = max(0, min(df["Marks"].min(), marks_thr) - 10)
    ax.axvspan(-100, att_thr, color=RED, alpha=0.07)
    ax.axhspan(-100, marks_thr, color=RED, alpha=0.07)
    if len(df) <= 40:
        for r in df.itertuples(index=False):
            ax.annotate(r.Name, (r.Attendance_Pct, r.Marks), xytext=(6, 6),
                        textcoords="offset points", fontsize=9)
    ax.set(title="Attendance vs Marks", xlabel="Attendance (%)", ylabel="Marks (out of 100)",
           xlim=(lo_x, 102), ylim=(lo_y, 102))
    ax.legend(handles=_legend_handles(), loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def make_grouped_bar(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    n = len(df)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(min(20, max(7, 0.5 * n)), 5))
    ax.bar(x - 0.2, df["Attendance_Pct"], 0.4, label="Attendance %", color="#2E75B6")
    ax.bar(x + 0.2, df["Marks"], 0.4, label="Marks", color="#C55A11")
    ax.axhline(att_thr, color="#2E75B6", ls="--", lw=1)
    ax.axhline(marks_thr, color="#C55A11", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Name"], rotation=90 if n > 12 else 0, fontsize=7 if n > 40 else 9)
    ax.set(title="Attendance % and Marks by Student", xlabel="Student", ylabel="Value (0-100)",
           ylim=(0, 120))
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def _hist(df, col, thr, title, xlabel):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df[col], bins=np.arange(0, 101, 10), color="#2E75B6", edgecolor="white")
    ax.axvline(thr, color=RED, ls="--", lw=2, label=f"Limit ({thr:g})")
    ax.axvline(df[col].mean(), color="black", ls=":", lw=2, label=f"Mean ({df[col].mean():.1f})")
    ax.set(title=title, xlabel=xlabel, ylabel="Students")
    ax.legend()
    fig.tight_layout()
    return fig


def make_marks_hist(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    return _hist(df, "Marks", marks_thr, "Marks Distribution", "Marks")


def make_attendance_hist(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    return _hist(df, "Attendance_Pct", att_thr, "Attendance Distribution", "Attendance (%)")


def make_status_pie(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    counts = df["Status"].value_counts().reindex(["OK", "At Risk"]).fillna(0)
    counts = counts[counts > 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts, labels=counts.index, autopct="%1.0f%%", startangle=90,
           colors=[GREEN if i == "OK" else RED for i in counts.index],
           wedgeprops=dict(edgecolor="white"))
    ax.set_title("OK vs At Risk")
    fig.tight_layout()
    return fig


def make_reason_bar(df, att_thr=ATTENDANCE_THRESHOLD, marks_thr=MARKS_THRESHOLD):
    vals = [int((df["Low_Attendance"] & ~df["Low_Marks"]).sum()),
            int((~df["Low_Attendance"] & df["Low_Marks"]).sum()),
            int((df["Low_Attendance"] & df["Low_Marks"]).sum())]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(["Low attendance\nonly", "Low marks\nonly", "Both"], vals,
           color=["#E67E22", "#8E44AD", RED])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.05, str(v), ha="center")
    ax.set(title="Why Students Are At Risk", ylabel="Students", ylim=(0, max(vals + [1]) + 1))
    fig.tight_layout()
    return fig


CHARTS = {
    "Bar: attendance % by student": make_bar_chart,
    "Scatter: Attendance vs Marks": make_scatter_chart,
    "Bar: marks by student": make_marks_bar,
    "Grouped bar: attendance % and marks": make_grouped_bar,
    "Histogram: marks distribution": make_marks_hist,
    "Histogram: attendance distribution": make_attendance_hist,
    "Pie: OK vs At Risk": make_status_pie,
    "Bar: why students are At Risk": make_reason_bar,
}


# ----------------------------------------------------------------------
# 5. CONSOLE MODE
# ----------------------------------------------------------------------
MENU = """
===== Student Attendance & Performance Tracker =====
1. Report (attendance % and status of every student)
2. Summary and charts
3. KPI dashboard
0. Exit"""


def run_console():
    parser = argparse.ArgumentParser(description="PS 46 - Student Attendance & Performance Tracker")
    parser.add_argument("csv_file", nargs="?", default="students.csv",
                        help="CSV with Name, Classes_Held, Classes_Attended, Marks")
    parser.add_argument("--save-charts", metavar="DIR", help="also save the two charts as PNG files")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        print(f"File '{args.csv_file}' not found.")
        sys.exit(1)

    clean, messages = load_and_clean(args.csv_file)
    for m in messages:
        print(m)
    if clean.empty:
        print(NO_DATA_MESSAGE)
        return
    df = add_metrics(clean)

    while True:
        print(MENU)
        try:
            choice = input("Enter choice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "1":
            print("\n".join(report_lines(df)))
        elif choice == "2":
            print("\n".join(summary_lines(df)))
            bar, scatter = make_bar_chart(df), make_scatter_chart(df)
            if args.save_charts:
                os.makedirs(args.save_charts, exist_ok=True)
                bar.savefig(os.path.join(args.save_charts, "attendance_bar_chart.png"),
                            dpi=150, bbox_inches="tight")
                scatter.savefig(os.path.join(args.save_charts, "attendance_vs_marks_scatter.png"),
                                dpi=150, bbox_inches="tight")
            print(CHARTS_MESSAGE)
            plt.show()
            plt.close("all")
        elif choice == "3":
            print("\n".join(kpi_lines(compute_kpis(df))))
        elif choice == "0":
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3 or 0.")


# ----------------------------------------------------------------------
# 6. STREAMLIT MODE
# ----------------------------------------------------------------------
def sample_class_csv():
    """A random 40-student class (with a header row) for demos."""
    rng = np.random.default_rng(7)
    att = np.clip(rng.normal(31, 5, 40).round(), 15, 40).astype(int)
    marks = np.clip(55 + (att - 31) * 2.2 + rng.normal(0, 12, 40), 10, 100).round().astype(int)
    lines = ["Name,Classes_Held,Classes_Attended,Marks"]
    lines += [f"Student{i + 1:02d},40,{a},{m}" for i, (a, m) in enumerate(zip(att, marks))]
    return "\n".join(lines) + "\n"


def run_streamlit():
    import streamlit as st

    plt.switch_backend("Agg")
    st.set_page_config(page_title="Attendance & Performance Tracker", page_icon="🎓", layout="wide")
    st.title("🎓 Student Attendance & Performance Tracker")
    st.caption("PS 46 | At Risk = attendance below the minimum OR marks below the pass mark")

    def show(fig):
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- sidebar: data source + rules ----------------
    st.sidebar.header("Data source")
    sources = ["Sample: Test Case 1", "Sample: random class (40 students)", "Upload CSV"]
    if os.path.exists("students.csv"):
        sources.insert(0, "students.csv (local file)")
    choice = st.sidebar.radio("Choose data", sources)
    if choice == "Upload CSV":
        up = st.sidebar.file_uploader("CSV: Name, Classes_Held, Classes_Attended, Marks", type=["csv"])
        if up is None:
            st.info("Upload a CSV with columns **Name, Classes_Held, Classes_Attended, Marks** "
                    "(header row optional), or pick a sample from the sidebar.")
            st.stop()
        source = io.StringIO(up.getvalue().decode("utf-8-sig", errors="replace"))
    elif choice.startswith("students.csv"):
        source = "students.csv"
    elif choice.startswith("Sample: Test"):
        source = io.StringIO(TEST_CASES[0]["csv"])
    else:
        source = io.StringIO(sample_class_csv())

    st.sidebar.header("Risk rules")
    att_thr = st.sidebar.slider("Minimum attendance (%)", 50, 95, int(ATTENDANCE_THRESHOLD))
    marks_thr = st.sidebar.slider("Pass marks", 20, 80, int(MARKS_THRESHOLD))

    clean, messages = load_and_clean(source)
    skipped = [m for m in messages if m.startswith("Row")]
    if clean.empty:
        for m in messages:
            st.warning(m)
        st.error(NO_DATA_MESSAGE)
        st.stop()

    df = add_metrics(clean, att_thr, marks_thr)
    k = compute_kpis(df, att_thr, marks_thr)

    tab_kpi, tab_rep, tab_chart, tab_dq, tab_test = st.tabs(
        ["📌 KPI Dashboard", "📋 Report", "📊 Charts", "🧹 Data Quality", "🧪 Test Cases"])

    # ---------------- KPI dashboard ----------------
    with tab_kpi:
        c = st.columns(5)
        c[0].metric("Students analysed", k["students"])
        c[1].metric("Average attendance", f"{k['avg_attendance']:.2f}%",
                    f"{k['avg_attendance'] - att_thr:+.1f} pts vs minimum")
        c[2].metric("Average marks", f"{k['avg_marks']:.2f}",
                    f"{k['avg_marks'] - marks_thr:+.1f} pts vs pass mark")
        c[3].metric("At Risk students", k["at_risk"], f"{k['at_risk_pct']:.1f}% of class",
                    delta_color="off")
        c[4].metric("Pass rate", f"{k['pass_rate']:.1f}%")

        c = st.columns(5)
        c[0].metric("Attendance compliance", f"{k['compliance']:.1f}%")
        c[1].metric("Overall attendance (pooled)", f"{k['pooled_attendance']:.2f}%")
        c[2].metric("Median marks", f"{k['median_marks']:.1f}")
        c[3].metric("Marks std deviation", f"{k['std_marks']:.2f}")
        c[4].metric("Attendance-marks correlation",
                    "n/a" if k["correlation"] is None else f"{k['correlation']:.2f}",
                    None if k["correlation"] is None else corr_strength(k["correlation"]),
                    delta_color="off")

        c = st.columns(4)
        c[0].metric("Top scorer", str(k["top_scorer"][0]), f"{k['top_scorer'][1]:g} marks", delta_color="off")
        c[1].metric("Lowest scorer", str(k["lowest_scorer"][0]), f"{k['lowest_scorer'][1]:g} marks",
                    delta_color="off")
        c[2].metric("Best attendance", str(k["best_attendance"][0]), f"{k['best_attendance'][1]:.2f}%",
                    delta_color="off")
        c[3].metric("Lowest attendance", str(k["lowest_attendance"][0]),
                    f"{k['lowest_attendance'][1]:.2f}%", delta_color="off")

        st.subheader("Key insights")
        st.markdown("\n".join(f"- {t}" for t in insights(k, att_thr, marks_thr)))

        left, right = st.columns(2)
        with left:
            show(make_status_pie(df))
        with right:
            show(make_reason_bar(df))

        st.subheader("At Risk students")
        risk_df = df[df["At_Risk"]][["Name", "Attendance_Pct", "Marks", "Reason"]]
        if risk_df.empty:
            st.success("No student is At Risk.")
        else:
            st.dataframe(risk_df.sort_values("Attendance_Pct").round(2), use_container_width=True)
        if k["watch_list"]:
            st.subheader("Watch list (OK now, but close to a limit)")
            st.dataframe(df[df["Name"].isin(k["watch_list"]) & ~df["At_Risk"]]
                         [["Name", "Attendance_Pct", "Marks"]].round(2), use_container_width=True)

    # ---------------- report ----------------
    with tab_rep:
        c1, c2, c3 = st.columns(3)
        status = c1.radio("Show", ["All", "At Risk", "OK"], horizontal=True)
        sort_by = c2.selectbox("Sort by", ["Original order", "Attendance_Pct", "Marks", "Name"])
        search = c3.text_input("Search name")
        view = df
        if status != "All":
            view = view[view["Status"] == status]
        if search:
            view = view[view["Name"].str.contains(search, case=False)]
        if sort_by != "Original order":
            view = view.sort_values(sort_by)
        st.markdown("**Console-style report**")
        st.code("\n".join(report_lines(view)) or "(no students match)")
        st.dataframe(
            view[["Name", "Classes_Held", "Classes_Attended", "Attendance_Pct", "Marks", "Status", "Reason"]]
            .round(2), use_container_width=True)
        st.download_button("⬇️ Download report (CSV)", view.round(2).to_csv(index=False).encode(),
                           "attendance_report.csv", "text/csv")

    # ---------------- charts ----------------
    with tab_chart:
        picks = st.multiselect("Charts to display", list(CHARTS),
                               default=["Bar: attendance % by student", "Scatter: Attendance vs Marks"])
        order = st.selectbox("Order students by (bar charts)",
                             ["Original order", "Attendance (high to low)", "Marks (high to low)"])
        plot_df = df
        if order.startswith("Attendance"):
            plot_df = df.sort_values("Attendance_Pct", ascending=False)
        elif order.startswith("Marks"):
            plot_df = df.sort_values("Marks", ascending=False)
        for label in picks:
            show(CHARTS[label](plot_df, att_thr, marks_thr))

    # ---------------- data quality ----------------
    with tab_dq:
        c = st.columns(3)
        c[0].metric("Rows read", len(clean) + len(skipped))
        c[1].metric("Valid rows", len(clean))
        c[2].metric("Skipped rows", len(skipped))
        if messages:
            for m in messages:
                st.warning(m)
        else:
            st.success("All rows are valid.")

    # ---------------- test cases ----------------
    with tab_test:
        results = [simulate(t["csv"], t["option"]) == t["expected"] for t in TEST_CASES]
        (st.success if all(results) else st.error)(f"{sum(results)}/{len(TEST_CASES)} test cases passed")
        for t, ok in zip(TEST_CASES, results):
            with st.expander(f"{'✅' if ok else '❌'} {t['name']}", expanded=False):
                a, b, d = st.columns(3)
                a.markdown(f"**Input** - {t['menu']}")
                a.code(t["csv"])
                b.markdown("**Expected output**")
                b.code("\n".join(t["expected"]))
                d.markdown("**Actual output**")
                d.code("\n".join(simulate(t["csv"], t["option"])))


# ----------------------------------------------------------------------
# ENTRY POINT - decides between console and Streamlit
# ----------------------------------------------------------------------
def _in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    try:
        return get_script_run_ctx(suppress_warning=True) is not None
    except TypeError:
        return get_script_run_ctx() is not None


if __name__ == "__main__":
    if _in_streamlit():
        run_streamlit()
    else:
        run_console()
