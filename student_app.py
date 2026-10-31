"""
Student Performance Explorer - Streamlit web app
Run with:  streamlit run student_app.py
Needs:     pip install streamlit pandas numpy matplotlib openpyxl
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Student Performance Explorer", page_icon="🎓", layout="wide")
plt.style.use("ggplot")

DEFAULT_FILE = "student_performance_dataset.xlsx"

RENAME = {
    "Maths Marks": "Maths", "Science Marks": "Science", "English Marks": "English",
    "Social Studies Marks": "Social_Studies", "Computer Marks": "Computer",
    "Study Hours/Day": "Study_Hours", "Travel Time (min/day)": "Travel_Time",
    "Assignment Score (%)": "Assignment", "Practical Work (%)": "Practical",
    "Attendance (%)": "Attendance", "Social Media (hrs/day)": "Social_Media",
    "Average Marks": "Avg_Marks", "Predicted Performance Score": "Score",
}
SUBJECTS = ["Maths", "Science", "English", "Social_Studies", "Computer"]
HABITS = ["Study_Hours", "Travel_Time", "Assignment", "Practical", "Attendance", "Social_Media"]
FACTORS = SUBJECTS + HABITS + ["Avg_Marks"]
REG_FEATURES = ["Avg_Marks", "Study_Hours", "Travel_Time", "Assignment",
                "Practical", "Attendance", "Social_Media"]
NUMERIC = FACTORS + ["Score"]
GRADE_ORDER = ["A+", "A", "B", "C", "D", "F"]


# ----------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------
def grade_of(s):
    return ("A+" if s >= 85 else "A" if s >= 75 else "B" if s >= 65
            else "C" if s >= 55 else "D" if s >= 45 else "F")


def prepare(df):
    df = df.iloc[:, :15].rename(columns=RENAME)
    df = df.dropna(subset=["Student", "Score"]).reset_index(drop=True)
    if "Grade" not in df.columns:
        df["Grade"] = df["Score"].apply(grade_of)
    return df


@st.cache_data
def load_file(source):
    return prepare(pd.read_excel(source, sheet_name="Student Data"))


st.sidebar.title("🎓 Student Explorer")
uploaded = st.sidebar.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])
if uploaded is not None:
    data = load_file(uploaded)
elif os.path.exists(DEFAULT_FILE):
    data = load_file(DEFAULT_FILE)
else:
    st.title("🎓 Student Performance Explorer")
    st.info(f"Upload your Excel file in the sidebar, or place **{DEFAULT_FILE}** "
            "next to this script.")
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR FILTERS
# ----------------------------------------------------------------------
st.sidebar.header("Filters")
grades_present = [g for g in GRADE_ORDER if g in data["Grade"].unique()]
sel_grades = st.sidebar.multiselect("Grade", grades_present, default=grades_present)
smin, smax = float(data["Score"].min()), float(data["Score"].max())
sel_score = st.sidebar.slider("Score range", smin, smax, (smin, smax))
amin, amax = int(data["Attendance"].min()), int(data["Attendance"].max())
sel_att = st.sidebar.slider("Attendance range (%)", amin, amax, (amin, amax))

df = data[
    data["Grade"].isin(sel_grades)
    & data["Score"].between(*sel_score)
    & data["Attendance"].between(*sel_att)
].reset_index(drop=True)

st.sidebar.markdown(f"**Students selected:** {len(df)} / {len(data)}")
palette = st.sidebar.selectbox("Colour theme",
                               ["viridis", "plasma", "coolwarm", "RdYlGn", "Blues", "tab10"])

st.title("🎓 Student Performance Explorer")
if len(df) < 5:
    st.warning("Too few students match the filters. Widen the filters in the sidebar.")
    st.stop()

tabs = st.tabs(["📊 Overview", "📈 Distribution", "🔗 Relationships", "🌡️ Correlation",
                "⚖️ Factor Impact", "🧩 Group Comparison", "🥇 Top vs Bottom",
                "🎛️ What-If", "🗂️ Data"])

# ======================================================================
# TAB 1 - OVERVIEW
# ======================================================================
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Students", len(df))
    c2.metric("Mean score", f"{df['Score'].mean():.1f}")
    c3.metric("Highest", f"{df['Score'].max():.1f}")
    c4.metric("Lowest", f"{df['Score'].min():.1f}")
    c5.metric("Std dev", f"{df['Score'].std():.1f}")

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df["Score"], bins=15, color="#2E75B6", edgecolor="white")
        ax.axvline(df["Score"].mean(), color="red", ls="--", label="Mean")
        ax.set(title="Score distribution", xlabel="Score", ylabel="Students")
        ax.legend()
        st.pyplot(fig)
    with right:
        counts = df["Grade"].value_counts().reindex(GRADE_ORDER).dropna()
        kind = st.radio("Grade chart type", ["Pie", "Bar"], horizontal=True)
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if kind == "Pie":
            ax.pie(counts, labels=counts.index, autopct="%1.0f%%", startangle=90,
                   wedgeprops=dict(edgecolor="white"))
        else:
            ax.bar(counts.index, counts.values, color="#548235")
            ax.set(xlabel="Grade", ylabel="Students")
        ax.set_title("Grade share")
        st.pyplot(fig)

    st.subheader("Summary statistics")
    st.dataframe(df[NUMERIC].describe().round(2).T, use_container_width=True)

# ======================================================================
# TAB 2 - DISTRIBUTION
# ======================================================================
with tabs[1]:
    st.subheader("Look at the distribution of any factor")
    c1, c2, c3 = st.columns(3)
    var = c1.selectbox("Factor", NUMERIC, index=NUMERIC.index("Score"))
    ctype = c2.selectbox("Chart type", ["Histogram", "Box plot", "Violin plot",
                                        "Cumulative (ECDF)"])
    split = c3.selectbox("Split by", ["None", "Grade"])
    bins = st.slider("Histogram bins", 5, 40, 15) if ctype == "Histogram" else 15

    fig, ax = plt.subplots(figsize=(9, 4.8))
    if split == "None":
        groups = {var: df[var].values}
    else:
        groups = {g: df.loc[df["Grade"] == g, var].values
                  for g in GRADE_ORDER if g in df["Grade"].values}
    cols = plt.get_cmap(palette)(np.linspace(0.15, 0.85, len(groups)))

    if ctype == "Histogram":
        for (k, v), c in zip(groups.items(), cols):
            ax.hist(v, bins=bins, alpha=0.65, label=k, color=c, edgecolor="white")
        ax.set(xlabel=var, ylabel="Students")
    elif ctype == "Box plot":
        bp = ax.boxplot(list(groups.values()), patch_artist=True)
        ax.set_xticklabels(list(groups.keys()))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
        ax.set(ylabel=var)
    elif ctype == "Violin plot":
        vp = ax.violinplot(list(groups.values()), showmeans=True)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels(list(groups.keys()))
        for body, c in zip(vp["bodies"], cols):
            body.set_facecolor(c)
            body.set_alpha(0.7)
        ax.set(ylabel=var)
    else:
        for (k, v), c in zip(groups.items(), cols):
            xs = np.sort(v)
            ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post", label=k, color=c, lw=2)
        ax.set(xlabel=var, ylabel="Cumulative share of students")
    if split != "None" and ctype in ("Histogram", "Cumulative (ECDF)"):
        ax.legend(title="Grade")
    ax.set_title(f"{ctype} of {var}")
    st.pyplot(fig)

    m = df[var]
    st.caption(f"Mean **{m.mean():.2f}** | Median **{m.median():.2f}** | "
               f"Std **{m.std():.2f}** | Min **{m.min():.2f}** | Max **{m.max():.2f}**")

# ======================================================================
# TAB 3 - RELATIONSHIPS
# ======================================================================
with tabs[2]:
    st.subheader("Relationship between any two factors")
    c1, c2, c3 = st.columns(3)
    xvar = c1.selectbox("X axis", NUMERIC, index=NUMERIC.index("Study_Hours"))
    yvar = c2.selectbox("Y axis", NUMERIC, index=NUMERIC.index("Score"))
    ctype = c3.selectbox("Chart type", ["Scatter", "Bubble", "Hexbin", "Line (binned mean)"])

    c4, c5, c6 = st.columns(3)
    colour_by = c4.selectbox("Colour by", ["None"] + NUMERIC, index=0) if ctype in ("Scatter", "Bubble") else "None"
    size_by = c5.selectbox("Bubble size by", NUMERIC, index=NUMERIC.index("Attendance")) if ctype == "Bubble" else None
    trend = c6.checkbox("Show trend line", value=True) if ctype in ("Scatter", "Bubble", "Hexbin") else False

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x, y = df[xvar].values.astype(float), df[yvar].values.astype(float)

    if ctype in ("Scatter", "Bubble"):
        sizes = 30
        if ctype == "Bubble":
            s = df[size_by].values.astype(float)
            sizes = (s - s.min()) / (s.max() - s.min() + 1e-9) * 250 + 20
        if colour_by != "None":
            sc = ax.scatter(x, y, s=sizes, c=df[colour_by], cmap=palette, alpha=0.75,
                            edgecolor="black", linewidth=0.3)
            fig.colorbar(sc, ax=ax, label=colour_by)
        else:
            ax.scatter(x, y, s=sizes, alpha=0.7, color="#2E75B6",
                       edgecolor="black", linewidth=0.3)
    elif ctype == "Hexbin":
        hb = ax.hexbin(x, y, gridsize=12, cmap=palette, mincnt=1)
        fig.colorbar(hb, ax=ax, label="Students")
    else:
        q = st.slider("Number of bins", 3, 12, 6)
        b = pd.qcut(df[xvar], q=q, duplicates="drop")
        g = df.groupby(b, observed=True).agg(xm=(xvar, "mean"), ym=(yvar, "mean"),
                                            ys=(yvar, "std"))
        ax.errorbar(g["xm"], g["ym"], yerr=g["ys"], fmt="-o", capsize=4,
                    color="#C55A11", lw=2)

    if trend and xvar != yvar:
        k, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, k * xs + b0, color="red", lw=2, label=f"y = {k:.2f}x + {b0:.1f}")
        ax.legend()
    ax.set(xlabel=xvar, ylabel=yvar, title=f"{yvar} vs {xvar}")
    st.pyplot(fig)

    r = np.corrcoef(x, y)[0, 1]
    strength = "very strong" if abs(r) > .8 else "strong" if abs(r) > .6 else \
        "moderate" if abs(r) > .4 else "weak" if abs(r) > .2 else "very weak/none"
    st.info(f"Correlation (r) = **{r:.3f}** → a **{strength} "
            f"{'positive' if r > 0 else 'negative'}** relationship.")

    with st.expander("Show many scatter plots at once (small multiples)"):
        picks = st.multiselect("Factors to plot against Score", FACTORS,
                               default=["Study_Hours", "Attendance", "Social_Media", "Avg_Marks"])
        if picks:
            ncol = min(3, len(picks))
            nrow = int(np.ceil(len(picks) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False)
            for ax, col in zip(axes.ravel(), picks):
                xx, yy = df[col].values.astype(float), df["Score"].values
                kk, bb = np.polyfit(xx, yy, 1)
                xs = np.linspace(xx.min(), xx.max(), 50)
                ax.scatter(xx, yy, s=14, alpha=0.6, color="#2E75B6")
                ax.plot(xs, kk * xs + bb, color="red")
                ax.set_title(f"{col} (r={np.corrcoef(xx, yy)[0, 1]:.2f})", fontsize=10)
            for ax in axes.ravel()[len(picks):]:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)

# ======================================================================
# TAB 4 - CORRELATION
# ======================================================================
with tabs[3]:
    st.subheader("Correlation heatmap")
    chosen = st.multiselect("Variables to include", NUMERIC,
                            default=NUMERIC, key="corr_vars")
    if len(chosen) >= 2:
        cm = df[chosen].corr()
        show_vals = st.checkbox("Show numbers", True)
        fig, ax = plt.subplots(figsize=(max(6, len(chosen) * 0.8), max(5, len(chosen) * 0.7)))
        im = ax.imshow(cm.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(chosen)))
        ax.set_yticks(range(len(chosen)))
        ax.set_xticklabels(chosen, rotation=45, ha="right")
        ax.set_yticklabels(chosen)
        ax.grid(False)
        if show_vals:
            for i in range(len(chosen)):
                for j in range(len(chosen)):
                    ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Select at least two variables.")

# ======================================================================
# TAB 5 - FACTOR IMPACT
# ======================================================================
with tabs[4]:
    st.subheader("Which factors affect the score the most?")
    pool = st.multiselect("Factors to analyse", FACTORS, default=FACTORS, key="impact_vars")
    view = st.radio("Chart", ["Correlation bar", "Regression importance (bar)",
                              "Regression share (pie)"], horizontal=True)

    if pool:
        if view == "Correlation bar":
            cs = df[pool].corrwith(df["Score"]).sort_values()
            fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(cs) + 1.5)))
            ax.barh(cs.index, cs.values, color=["#C00000" if v < 0 else "#2E75B6" for v in cs])
            for yv, v in enumerate(cs.values):
                ax.text(v + (0.01 if v >= 0 else -0.01), yv, f"{v:.2f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9)
            ax.axvline(0, color="black", lw=0.8)
            ax.set(xlabel="Correlation with Score", title="Correlation of each factor with Score")
            st.pyplot(fig)
        else:
            feats = [f for f in pool if f in REG_FEATURES]
            skipped = [f for f in pool if f not in REG_FEATURES]
            if skipped:
                st.caption("Regression uses overall Avg_Marks plus habit factors, so individual "
                           f"subjects ({', '.join(skipped)}) are skipped here.")
            if len(feats) >= 1:
                X = df[feats].values.astype(float)
                y = df["Score"].values.astype(float)
                Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
                yz = (y - y.mean()) / y.std()
                A = np.column_stack([np.ones(len(Xz)), Xz])
                beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
                coef = pd.Series(beta[1:], index=feats)
                r2 = 1 - np.sum((yz - A @ beta) ** 2) / np.sum((yz - yz.mean()) ** 2)
                if view == "Regression importance (bar)":
                    o = coef.sort_values()
                    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(o) + 1.5)))
                    ax.barh(o.index, o.values,
                            color=["#C00000" if v < 0 else "#2E75B6" for v in o])
                    ax.axvline(0, color="black", lw=0.8)
                    ax.set(xlabel="Standardised coefficient", title=f"Regression importance (R² = {r2:.3f})")
                    st.pyplot(fig)
                else:
                    share = (coef.abs() / coef.abs().sum() * 100).sort_values(ascending=False)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.pie(share, labels=share.index, autopct="%1.1f%%", startangle=90,
                           colors=plt.get_cmap("tab10")(np.arange(len(share))),
                           wedgeprops=dict(edgecolor="white"))
                    ax.set_title("Share of influence on Score")
                    st.pyplot(fig)
                st.caption("Standardised coefficients put every factor on the same scale, "
                           "so they can be compared directly. Red = lowers score.")
            else:
                st.info("Select at least one non-subject factor for regression views.")

# ======================================================================
# TAB 6 - GROUP COMPARISON
# ======================================================================
with tabs[5]:
    st.subheader("Average score across levels of a factor")
    c1, c2, c3 = st.columns(3)
    fvar = c1.selectbox("Group by factor", HABITS + SUBJECTS + ["Avg_Marks"], key="grp_var")
    nb = c2.slider("Number of groups", 2, 8, 5)
    gchart = c3.selectbox("Chart", ["Bar", "Line with error bars", "Box plot"], key="grp_chart")

    bins_ = pd.qcut(df[fvar], q=nb, duplicates="drop")
    labels = [f"{iv.left:.1f}-{iv.right:.1f}" for iv in bins_.cat.categories]
    bins_ = bins_.cat.rename_categories(labels)
    grp = df.groupby(bins_, observed=True)["Score"].agg(["mean", "std", "count"])

    fig, ax = plt.subplots(figsize=(9, 5))
    cols = plt.get_cmap(palette)(np.linspace(0.15, 0.85, len(grp)))
    if gchart == "Bar":
        ax.bar(grp.index.astype(str), grp["mean"], color=cols)
        for i, v in enumerate(grp["mean"]):
            ax.text(i, v + 0.3, f"{v:.1f}", ha="center")
    elif gchart == "Line with error bars":
        ax.errorbar(grp.index.astype(str), grp["mean"], yerr=grp["std"], fmt="-o",
                    capsize=4, color="#C55A11", lw=2)
    else:
        bp = ax.boxplot([df.loc[bins_ == k, "Score"] for k in grp.index], patch_artist=True)
        ax.set_xticklabels(grp.index.astype(str))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
    ax.set(xlabel=f"{fvar} range", ylabel="Score", title=f"Score by {fvar} group")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    st.dataframe(grp.round(2), use_container_width=True)

    st.subheader("Two-factor heat map")
    c1, c2 = st.columns(2)
    hx = c1.selectbox("Row factor", HABITS, index=0, key="hx")
    hy = c2.selectbox("Column factor", HABITS, index=5, key="hy")
    if hx != hy:
        bx = pd.qcut(df[hx], 4, duplicates="drop")
        by = pd.qcut(df[hy], 4, duplicates="drop")
        pt = df.pivot_table(index=bx, columns=by, values="Score", aggfunc="mean", observed=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pt.values, cmap=palette, aspect="auto", origin="lower")
        ax.set_xticks(range(pt.shape[1]))
        ax.set_yticks(range(pt.shape[0]))
        ax.set_xticklabels([f"{i.left:.1f}-{i.right:.1f}" for i in pt.columns], rotation=30)
        ax.set_yticklabels([f"{i.left:.1f}-{i.right:.1f}" for i in pt.index])
        for i in range(pt.shape[0]):
            for j in range(pt.shape[1]):
                if not np.isnan(pt.values[i, j]):
                    ax.text(j, i, f"{pt.values[i, j]:.0f}", ha="center", va="center",
                            color="white", fontsize=9)
        ax.grid(False)
        ax.set(xlabel=hy, ylabel=hx, title=f"Mean score: {hx} × {hy}")
        fig.colorbar(im, ax=ax, label="Mean score")
        st.pyplot(fig)
    else:
        st.info("Pick two different factors.")

# ======================================================================
# TAB 7 - TOP VS BOTTOM
# ======================================================================
with tabs[6]:
    st.subheader("Compare high and low performers")
    pct = st.slider("Group size (top/bottom %)", 10, 50, 25)
    hi = df[df["Score"] >= df["Score"].quantile(1 - pct / 100)]
    lo = df[df["Score"] <= df["Score"].quantile(pct / 100)]
    pick = st.multiselect("Factors to compare", HABITS + SUBJECTS,
                          default=HABITS + SUBJECTS, key="tb_vars")
    style = st.radio("Chart", ["Grouped bar", "Radar"], horizontal=True)

    if len(pick) >= 3 and len(hi) and len(lo):
        mx = data[pick].max()          # normalise by global max so units are comparable
        nh, nl = hi[pick].mean() / mx * 100, lo[pick].mean() / mx * 100
        if style == "Grouped bar":
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(pick))
            ax.bar(x - 0.2, nh, 0.4, label=f"Top {pct}% (n={len(hi)})", color="#2E75B6")
            ax.bar(x + 0.2, nl, 0.4, label=f"Bottom {pct}% (n={len(lo)})", color="#C00000")
            ax.set_xticks(x)
            ax.set_xticklabels(pick, rotation=35, ha="right")
            ax.set(ylabel="% of max value", title="Top vs bottom performers")
            ax.legend()
        else:
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, polar=True)
            ang = np.linspace(0, 2 * np.pi, len(pick), endpoint=False).tolist()
            ang += ang[:1]
            for vals, lab, c in [(nh, "Top", "#2E75B6"), (nl, "Bottom", "#C00000")]:
                v = vals.tolist() + vals.tolist()[:1]
                ax.plot(ang, v, color=c, lw=2, label=lab)
                ax.fill(ang, v, color=c, alpha=0.2)
            ax.set_xticks(ang[:-1])
            ax.set_xticklabels(pick, fontsize=9)
            ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        st.pyplot(fig)

        comp = pd.DataFrame({"Top avg": hi[pick].mean(), "Bottom avg": lo[pick].mean()})
        comp["Gap"] = comp["Top avg"] - comp["Bottom avg"]
        st.dataframe(comp.round(2).sort_values("Gap", ascending=False), use_container_width=True)
    else:
        st.info("Select at least 3 factors.")

# ======================================================================
# TAB 8 - WHAT-IF
# ======================================================================
with tabs[7]:
    st.subheader("What-if simulator")
    st.caption("A linear model is fitted on the full dataset. Change the sliders to see how the "
               "predicted score responds.")
    X = data[REG_FEATURES].values.astype(float)
    y = data["Score"].values.astype(float)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, coefs = beta[0], pd.Series(beta[1:], index=REG_FEATURES)

    start = st.selectbox("Start from", ["Class average"] + data["Student"].tolist())
    base = data[REG_FEATURES].mean() if start == "Class average" else \
        data.loc[data["Student"] == start, REG_FEATURES].iloc[0]

    cols = st.columns(3)
    vals = {}
    for i, f in enumerate(REG_FEATURES):
        lo_, hi_ = float(data[f].min()), float(data[f].max())
        step = 0.1 if f in ("Study_Hours", "Social_Media", "Avg_Marks") else 1.0
        vals[f] = cols[i % 3].slider(f, lo_, hi_, float(np.clip(base[f], lo_, hi_)), step=step)

    pred = float(np.clip(intercept + sum(coefs[f] * vals[f] for f in REG_FEATURES), 0, 100))
    base_pred = float(np.clip(intercept + sum(coefs[f] * base[f] for f in REG_FEATURES), 0, 100))
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted score", f"{pred:.1f}", f"{pred - base_pred:+.1f} vs start")
    m2.metric("Grade", grade_of(pred))
    m3.metric("Class mean", f"{data['Score'].mean():.1f}")

    contrib = pd.Series({f: coefs[f] * (vals[f] - base[f]) for f in REG_FEATURES}).sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(contrib.index, contrib.values,
            color=["#C00000" if v < 0 else "#548235" for v in contrib])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="Score points gained / lost", title="Effect of your changes on the score")
    st.pyplot(fig)

# ======================================================================
# TAB 9 - DATA
# ======================================================================
with tabs[8]:
    st.subheader("Filtered data")
    search = st.text_input("Search student name (e.g. AB)")
    show = df if not search else df[df["Student"].str.contains(search, case=False)]
    sort_by = st.selectbox("Sort by", ["Student"] + NUMERIC, index=len(NUMERIC))
    asc = st.checkbox("Ascending", False)
    show = show.sort_values(sort_by, ascending=asc)
    st.dataframe(show, use_container_width=True, height=450)
    st.download_button("⬇️ Download filtered data (CSV)", show.to_csv(index=False).encode(),
                       "filtered_students.csv", "text/csv")
