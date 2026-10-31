"""
Student Performance Analysis
----------------------------
Analyses every factor that affects the predicted performance score using
pandas, numpy and matplotlib.

Usage:
    python analyse_student_performance.py
Requires: student_performance_dataset.xlsx in the same folder (plus openpyxl
for pandas.read_excel).  Charts are saved to the 'charts' folder.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 0. SETTINGS
# ----------------------------------------------------------------------
FILE = "student_performance_dataset.xlsx"
OUT_DIR = "charts"
os.makedirs(OUT_DIR, exist_ok=True)
plt.style.use("ggplot")
plt.rcParams.update({"figure.dpi": 100, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150)
    print(f"  saved -> {OUT_DIR}/{name}")


# ----------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ----------------------------------------------------------------------
df = pd.read_excel(FILE, sheet_name="Student Data")
df = df.iloc[:, :15]                       # ignore the summary cells at the side
df = df.rename(columns={
    "Maths Marks": "Maths", "Science Marks": "Science", "English Marks": "English",
    "Social Studies Marks": "Social_Studies", "Computer Marks": "Computer",
    "Study Hours/Day": "Study_Hours", "Travel Time (min/day)": "Travel_Time",
    "Assignment Score (%)": "Assignment", "Practical Work (%)": "Practical",
    "Attendance (%)": "Attendance", "Social Media (hrs/day)": "Social_Media",
    "Average Marks": "Avg_Marks", "Predicted Performance Score": "Score",
})
df = df.dropna(subset=["Student", "Score"]).reset_index(drop=True)

subjects = ["Maths", "Science", "English", "Social_Studies", "Computer"]
habits = ["Study_Hours", "Travel_Time", "Assignment", "Practical",
          "Attendance", "Social_Media"]
factors = subjects + habits + ["Avg_Marks"]

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {len(df)}   Columns: {df.shape[1]}")
print("Missing values:", int(df.isna().sum().sum()))
print("\nDescriptive statistics:")
print(df[factors + ["Score"]].describe().round(2).T)

# ----------------------------------------------------------------------
# 2. CORRELATION OF EVERY FACTOR WITH THE SCORE
# ----------------------------------------------------------------------
corr = df[factors + ["Score"]].corr()
corr_with_score = corr["Score"].drop("Score").sort_values()
print("\nCorrelation with Score (Pearson r):")
print(corr_with_score.round(3).to_string())

# ----------------------------------------------------------------------
# 3. CHART 1 - Score distribution (histogram + boxplot) and grade split
# ----------------------------------------------------------------------
print("\nCreating charts...")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].hist(df["Score"], bins=15, color="#2E75B6", edgecolor="white")
ax[0].axvline(df["Score"].mean(), color="red", ls="--",
              label=f"Mean = {df['Score'].mean():.1f}")
ax[0].set(title="Score Distribution (Histogram)", xlabel="Score", ylabel="Students")
ax[0].legend()

ax[1].boxplot(df["Score"], vert=True, patch_artist=True,
              boxprops=dict(facecolor="#9DC3E6"))
ax[1].set(title="Score Spread (Box Plot)", ylabel="Score", xticks=[])

grade_order = ["A+", "A", "B", "C", "D", "F"]
grade_counts = df["Grade"].value_counts().reindex(grade_order).dropna()
ax[2].pie(grade_counts, labels=grade_counts.index, autopct="%1.0f%%",
          startangle=90, colors=plt.cm.Blues(np.linspace(0.9, 0.3, len(grade_counts))),
          wedgeprops=dict(edgecolor="white"))
ax[2].set_title("Grade Share (Pie Chart)")
save(fig, "01_score_distribution.png")

# ----------------------------------------------------------------------
# 4. CHART 2 - Correlation heatmap (all variables)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
ax.grid(False)
fig.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Correlation Heatmap")
save(fig, "02_correlation_heatmap.png")

# ----------------------------------------------------------------------
# 5. CHART 3 - Horizontal bar: factor correlation with Score
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#C00000" if v < 0 else "#2E75B6" for v in corr_with_score]
ax.barh(corr_with_score.index, corr_with_score.values, color=colors)
for y, v in enumerate(corr_with_score.values):
    ax.text(v + (0.01 if v >= 0 else -0.01), y, f"{v:.2f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set(title="Which Factors Move the Score? (Blue = positive, Red = negative)",
       xlabel="Correlation with Score")
save(fig, "03_factor_correlation_bar.png")

# ----------------------------------------------------------------------
# 6. CHART 4 - Scatter plot + trend line for EVERY factor
# ----------------------------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
for ax, col in zip(axes.ravel(), factors):
    x, y = df[col].values, df["Score"].values
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.scatter(x, y, s=18, alpha=0.6, color="#2E75B6")
    ax.plot(xs, slope * xs + intercept, color="red", lw=2)
    ax.set_title(f"{col}  (r = {np.corrcoef(x, y)[0, 1]:.2f})")
    ax.set_xlabel(col)
    ax.set_ylabel("Score")
for ax in axes.ravel()[len(factors):]:
    ax.axis("off")
fig.suptitle("Score vs Each Factor (Scatter + Regression Line)", fontsize=15)
save(fig, "04_scatter_all_factors.png")

# ----------------------------------------------------------------------
# 7. CHART 5 - Subject marks: box plot, mean bar, correlation bar
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].boxplot([df[s] for s in subjects], labels=subjects, patch_artist=True,
              boxprops=dict(facecolor="#BDD7EE"))
ax[0].set(title="Marks Spread by Subject (Box Plot)", ylabel="Marks")
ax[0].tick_params(axis="x", rotation=25)

means = df[subjects].mean()
ax[1].bar(subjects, means, color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)))
for i, v in enumerate(means):
    ax[1].text(i, v + 0.5, f"{v:.1f}", ha="center")
ax[1].set(title="Average Marks per Subject", ylabel="Mean marks")
ax[1].tick_params(axis="x", rotation=25)

sub_corr = df[subjects].corrwith(df["Score"])
ax[2].bar(subjects, sub_corr, color="#548235")
for i, v in enumerate(sub_corr):
    ax[2].text(i, v + 0.01, f"{v:.2f}", ha="center")
ax[2].set(title="Subject Correlation with Score", ylabel="r", ylim=(0, 1))
ax[2].tick_params(axis="x", rotation=25)
save(fig, "05_subject_analysis.png")

# ----------------------------------------------------------------------
# 8. CHART 6 - Binned analysis: mean score per range of each habit
#    (line charts for continuous behaviour, bar charts for % factors)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, col in zip(axes.ravel(), habits):
    bins = pd.qcut(df[col], q=5, duplicates="drop")
    grp = df.groupby(bins, observed=True)["Score"].agg(["mean", "std"])
    labels = [f"{iv.left:.0f}-{iv.right:.0f}" if df[col].max() > 20
              else f"{iv.left:.1f}-{iv.right:.1f}" for iv in grp.index]
    ax.errorbar(labels, grp["mean"], yerr=grp["std"], fmt="-o", capsize=4,
                color="#C55A11", lw=2)
    ax.set(title=f"Avg Score by {col} Group", xlabel=f"{col} range (quintiles)",
           ylabel="Mean score")
    ax.tick_params(axis="x", rotation=20)
fig.suptitle("Average Score Across Low -> High Levels of Each Habit "
             "(error bars = 1 std dev)", fontsize=14)
save(fig, "06_binned_habit_effects.png")

# ----------------------------------------------------------------------
# 9. CHART 7 - Top 25% vs Bottom 25% students (grouped bar + radar)
# ----------------------------------------------------------------------
q75, q25 = df["Score"].quantile(0.75), df["Score"].quantile(0.25)
top = df[df["Score"] >= q75]
bottom = df[df["Score"] <= q25]

fig = plt.figure(figsize=(16, 6))
ax1 = fig.add_subplot(1, 2, 1)
cols = habits + subjects
x = np.arange(len(cols))
w = 0.38
# normalise each column 0-100 by its overall max so different units can share an axis
norm = lambda d: (d[cols].mean() / df[cols].max() * 100)
ax1.bar(x - w / 2, norm(top), w, label=f"Top 25% (n={len(top)})", color="#2E75B6")
ax1.bar(x + w / 2, norm(bottom), w, label=f"Bottom 25% (n={len(bottom)})", color="#C00000")
ax1.set_xticks(x)
ax1.set_xticklabels(cols, rotation=40, ha="right")
ax1.set(title="Top vs Bottom Students (each factor as % of its max)", ylabel="% of max")
ax1.legend()

# Radar chart
ax2 = fig.add_subplot(1, 2, 2, polar=True)
angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
angles += angles[:1]
for grp_df, label, c in [(top, "Top 25%", "#2E75B6"), (bottom, "Bottom 25%", "#C00000")]:
    vals = norm(grp_df).tolist()
    vals += vals[:1]
    ax2.plot(angles, vals, color=c, lw=2, label=label)
    ax2.fill(angles, vals, color=c, alpha=0.2)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(cols, fontsize=8)
ax2.set_title("Radar Profile: Top vs Bottom", pad=20)
ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
save(fig, "07_top_vs_bottom.png")

# ----------------------------------------------------------------------
# 10. CHART 8 - Score by grade (box plots) for key factors
# ----------------------------------------------------------------------
key = ["Avg_Marks", "Study_Hours", "Attendance", "Social_Media"]
present = [g for g in grade_order if g in df["Grade"].values]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
for ax, col in zip(axes, key):
    ax.boxplot([df.loc[df["Grade"] == g, col] for g in present],
               labels=present, patch_artist=True,
               boxprops=dict(facecolor="#C6E0B4"))
    ax.set(title=f"{col} by Grade", xlabel="Grade", ylabel=col)
save(fig, "08_factor_by_grade.png")

# ----------------------------------------------------------------------
# 11. CHART 9 - Bubble chart: study vs score, size = attendance, colour = social media
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))
sc = ax.scatter(df["Study_Hours"], df["Score"],
                s=(df["Attendance"] - df["Attendance"].min() + 5) * 6,
                c=df["Social_Media"], cmap="RdYlGn_r", alpha=0.7,
                edgecolor="black", linewidth=0.4)
fig.colorbar(sc, ax=ax, label="Social media hours/day")
ax.set(title="Study Hours vs Score\n(bubble size = attendance, colour = social media use)",
       xlabel="Study hours/day", ylabel="Score")
save(fig, "09_bubble_chart.png")

# ----------------------------------------------------------------------
# 12. CHART 10 - Trade-off: Study vs Social media (hexbin + stacked area-style)
# ----------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
hb = ax[0].hexbin(df["Study_Hours"], df["Social_Media"], C=df["Score"],
                  gridsize=8, cmap="viridis", reduce_C_function=np.mean)
fig.colorbar(hb, ax=ax[0], label="Mean score")
ax[0].set(title="Mean Score: Study Hours x Social Media",
          xlabel="Study hours/day", ylabel="Social media hours/day")

# attendance bands stacked by grade
bands = pd.cut(df["Attendance"], bins=[0, 70, 80, 90, 100],
               labels=["<70", "70-80", "80-90", "90-100"])
ct = pd.crosstab(bands, df["Grade"]).reindex(columns=present, fill_value=0)
bottom_vals = np.zeros(len(ct))
cmap = plt.cm.RdYlGn(np.linspace(0.95, 0.05, len(present)))
for g, c in zip(present, cmap):
    ax[1].bar(ct.index.astype(str), ct[g], bottom=bottom_vals, label=g, color=c)
    bottom_vals += ct[g].values
ax[1].set(title="Grades within Each Attendance Band (Stacked Bar)",
          xlabel="Attendance band (%)", ylabel="Students")
ax[1].legend(title="Grade")
save(fig, "10_tradeoffs.png")

# ----------------------------------------------------------------------
# 13. CHART 11 - Scatter matrix of the main habits
# ----------------------------------------------------------------------
from pandas.plotting import scatter_matrix
axs = scatter_matrix(df[["Score", "Avg_Marks", "Study_Hours", "Attendance", "Social_Media"]],
                     figsize=(11, 11), diagonal="hist", alpha=0.6,
                     hist_kwds={"bins": 12, "color": "#2E75B6"})
plt.gcf().suptitle("Scatter Matrix of Key Factors", y=0.92, fontsize=14)
save(plt.gcf(), "11_scatter_matrix.png")

# ----------------------------------------------------------------------
# 14. REGRESSION (NumPy least squares) - which factor matters MOST?
#     Standardised coefficients are comparable across different units.
# ----------------------------------------------------------------------
reg_features = ["Avg_Marks", "Study_Hours", "Travel_Time", "Assignment",
                "Practical", "Attendance", "Social_Media"]
X = df[reg_features].values.astype(float)
y = df["Score"].values.astype(float)
Xz = (X - X.mean(axis=0)) / X.std(axis=0)
yz = (y - y.mean()) / y.std()

A = np.column_stack([np.ones(len(Xz)), Xz])
beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
std_coef = pd.Series(beta[1:], index=reg_features)

pred = A @ beta
r2 = 1 - np.sum((yz - pred) ** 2) / np.sum((yz - yz.mean()) ** 2)

# raw (unstandardised) coefficients: "points of score per 1 unit"
A_raw = np.column_stack([np.ones(len(X)), X])
raw_beta, *_ = np.linalg.lstsq(A_raw, y, rcond=None)
raw_coef = pd.Series(raw_beta[1:], index=reg_features)

print("\nMULTIPLE LINEAR REGRESSION")
print(f"R-squared = {r2:.3f}")
print("\nRaw effect (score points per 1-unit increase):")
print(raw_coef.round(3).to_string())
print("\nStandardised importance:")
print(std_coef.sort_values(key=abs, ascending=False).round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
order = std_coef.sort_values()
ax[0].barh(order.index, order.values,
           color=["#C00000" if v < 0 else "#2E75B6" for v in order])
ax[0].set(title=f"Factor Importance (Standardised Coef.)\nR² = {r2:.3f}",
          xlabel="Std. coefficient")
ax[0].axvline(0, color="black", lw=0.8)

# relative importance as share of total |coef|
share = std_coef.abs() / std_coef.abs().sum() * 100
share = share.sort_values(ascending=False)
ax[1].pie(share, labels=share.index, autopct="%1.1f%%", startangle=90,
          colors=plt.cm.tab10(np.arange(len(share))),
          wedgeprops=dict(edgecolor="white"))
ax[1].set_title("Share of Influence on Score")

ax[2].scatter(y, pred * y.std() + y.mean(), alpha=0.6, color="#548235")
lims = [y.min(), y.max()]
ax[2].plot(lims, lims, "r--", label="Perfect fit")
ax[2].set(title="Actual vs Model-Predicted Score", xlabel="Actual", ylabel="Predicted")
ax[2].legend()
save(fig, "12_regression_importance.png")

# ----------------------------------------------------------------------
# 15. WHAT-IF: effect of changing one habit while holding others fixed
# ----------------------------------------------------------------------
base = df[reg_features].mean()
fig, ax = plt.subplots(figsize=(10, 6))
for col in ["Study_Hours", "Social_Media", "Attendance", "Travel_Time"]:
    grid = np.linspace(df[col].min(), df[col].max(), 50)
    scores = []
    for g in grid:
        row = base.copy()
        row[col] = g
        scores.append(raw_beta[0] + np.dot(raw_beta[1:], row.values))
    pct = (grid - grid.min()) / (grid.max() - grid.min()) * 100   # common 0-100 axis
    ax.plot(pct, scores, lw=2.5, label=col)
ax.set(title="What-If: Moving One Habit From its Min to Max (others at average)",
       xlabel="Position within the factor's range (0% = lowest, 100% = highest)",
       ylabel="Predicted score")
ax.legend()
save(fig, "13_what_if_analysis.png")

# ----------------------------------------------------------------------
# 16. TOP / BOTTOM STUDENTS TABLE + KEY INSIGHTS
# ----------------------------------------------------------------------
print("\nTop 5 students:")
print(df.nlargest(5, "Score")[["Student", "Avg_Marks", "Study_Hours",
                               "Attendance", "Social_Media", "Score"]].to_string(index=False))
print("\nBottom 5 students:")
print(df.nsmallest(5, "Score")[["Student", "Avg_Marks", "Study_Hours",
                                "Attendance", "Social_Media", "Score"]].to_string(index=False))

best_pos = corr_with_score.idxmax()
best_neg = corr_with_score.idxmin()
print("\nKEY INSIGHTS")
print("-" * 60)
print(f"* Strongest positive factor : {best_pos} (r = {corr_with_score[best_pos]:.2f})")
print(f"* Strongest negative factor : {best_neg} (r = {corr_with_score[best_neg]:.2f})")
print(f"* Most influential in regression: {std_coef.abs().idxmax()}")
print(f"* Each extra study hour/day  ~ {raw_coef['Study_Hours']:+.2f} score points")
print(f"* Each extra social-media hr ~ {raw_coef['Social_Media']:+.2f} score points")
print(f"* Each +10% attendance       ~ {raw_coef['Attendance'] * 10:+.2f} score points")
print(f"* Top 25% study {top['Study_Hours'].mean():.1f} h/day vs bottom 25% "
      f"{bottom['Study_Hours'].mean():.1f} h/day")
print(f"* Top 25% use social media {top['Social_Media'].mean():.1f} h/day vs bottom 25% "
      f"{bottom['Social_Media'].mean():.1f} h/day")

plt.show()
