# =====================================
# MCQ QUIZ RESPONSE ANALYTICS PROJECT
# =====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. LOAD DATA (EXCEL)
# ===============================

df = pd.read_excel("data.xlsx")

print("Data Loaded Successfully\n")
print(df.head())

# ===============================
# 2. DATA CLEANING
# ===============================

df.columns = df.columns.str.strip()
df.fillna("Not Answered", inplace=True)

# ===============================
# 3. ANSWER KEY
# ===============================

answer_key = {
    "Q1": "A",
    "Q2": "C",
    "Q3": "C",
    "Q4": "B",
    "Q5": "D"
}

# ===============================
# 4. CALCULATE SCORES
# ===============================

def calculate_score(row):
    score = 0
    for q in answer_key:
        if row[q] == answer_key[q]:
            score += 1
    return score

df["Score"] = df.apply(calculate_score, axis=1)

print("\nScores Calculated\n")
print(df[["Name", "Score"]])

# ===============================
# 5. OVERALL STATISTICS
# ===============================

print("\n===== OVERALL STATISTICS =====")
print("Total Students:", len(df))
print("Average Score:", df["Score"].mean())
print("Highest Score:", df["Score"].max())
print("Lowest Score:", df["Score"].min())

# ===============================
# 6. TOP STUDENTS
# ===============================

print("\n===== TOP STUDENTS =====")
top_students = df.sort_values("Score", ascending=False).head(5)
print(top_students[["Name", "Department", "Score"]])

# ===============================
# 7. DEPARTMENT ANALYSIS
# ===============================

print("\n===== DEPARTMENT ANALYSIS =====")
dept_perf = df.groupby("Department")["Score"].mean()
print(dept_perf)

plt.figure()
dept_perf.plot(kind="bar")
plt.title("Department Performance")
plt.show()

# ===============================
# 8. COLLEGE ANALYSIS (NEW)
# ===============================

print("\n===== COLLEGE ANALYSIS =====")

college_perf = df.groupby("College")["Score"].mean()
print(college_perf)

plt.figure()
college_perf.sort_values().plot(kind="barh")
plt.title("College Performance")
plt.show()

# ===============================
# 9. QUESTION ANALYSIS
# ===============================

print("\n===== QUESTION ANALYSIS =====")

question_accuracy = {}

for q in answer_key:
    correct = (df[q] == answer_key[q]).sum()
    question_accuracy[q] = correct / len(df)

question_df = pd.DataFrame.from_dict(
    question_accuracy,
    orient="index",
    columns=["Accuracy"]
)

# Difficulty classification
def classify_difficulty(acc):
    if acc > 0.8:
        return "Easy"
    elif acc >= 0.5:
        return "Medium"
    else:
        return "Difficult"

question_df["Difficulty"] = question_df["Accuracy"].apply(classify_difficulty)

print(question_df)

print("\nEasiest Question:", question_df["Accuracy"].idxmax())
print("Most Difficult Question:", question_df["Accuracy"].idxmin())

plt.figure()
question_df["Accuracy"].plot(kind="bar")
plt.title("Question Accuracy")
plt.show()

# ===============================
# 10. HEATMAP (VERY IMPORTANT)
# ===============================

print("\n===== HEATMAP =====")

pivot = df.pivot_table(values="Score", index="Department", columns="College")

plt.figure()
sns.heatmap(pivot, annot=True, cmap="coolwarm")
plt.title("Department vs College Performance")
plt.show()

# ===============================
# 11. SCORE DISTRIBUTION
# ===============================

plt.figure()
sns.histplot(df["Score"], bins=5, kde=True)
plt.title("Score Distribution")
plt.show()

# ===============================
# 12. EXPORT REPORTS
# ===============================

df.to_excel("quiz_results.xlsx", index=False)
question_df.to_excel("question_analysis.xlsx")

print("\nReports Generated Successfully!")
