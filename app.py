from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# LOGIN SYSTEM
# ======================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            st.rerun()

        else:
            st.error("❌ Invalid Credentials")

    st.stop()


# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(page_title="MCQ Dashboard", layout="wide")
st.title("📊 MCQ Quiz Analytics Dashboard")

st.write("This dashboard analyzes student performance in MCQ quizzes, including scores, rankings, and insights.")
st.markdown("---")

# ===============================
# LOAD DATA
# ===============================
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if len(df.columns) == 1:
        df = df.iloc[:, 0].str.split(",", expand=True)
        df.columns = ["NAME","COLLEGE","DEPARTMENT","Q1","Q2","Q3","Q4","Q5"]

    st.success("✅ File Uploaded Successfully")

else:
    st.warning("⚠ Please upload a CSV file to continue")
    st.stop()

# ===============================
# CLEAN DATA
# ===============================
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df.columns = df.columns.str.strip().str.upper()
df.fillna("Not Answered", inplace=True)

# ===============================
# FILTERS
# ===============================
st.sidebar.title("📊 Filters")

selected_dept = st.sidebar.selectbox(
    "Department",
    ["All"] + list(df["DEPARTMENT"].unique())
)

has_college = "COLLEGE" in df.columns

if has_college:
    selected_college = st.sidebar.selectbox(
        "College",
        ["All"] + list(df["COLLEGE"].unique())
    )
else:
    selected_college = "All"

if selected_dept != "All":
    df = df[df["DEPARTMENT"] == selected_dept]

if has_college and selected_college != "All":
    df = df[df["COLLEGE"] == selected_college]

# ===============================
# ANSWER KEY
# ===============================
answer_key = {
    "Q1": "A",
    "Q2": "C",
    "Q3": "C",
    "Q4": "B",
    "Q5": "D"
}

# ===============================
# CALCULATE SCORE
# ===============================
def calculate_score(row):
    score = 0
    for q in answer_key:
        if row[q] == answer_key[q]:
            score += 1
    return score

df["SCORE"] = df.apply(calculate_score, axis=1)

# ===============================
# RESULT & RANK
# ===============================
df["RANK"] = df["SCORE"].rank(ascending=False, method="dense")
df["RESULT"] = df["SCORE"].apply(lambda x: "Pass" if x >= 3 else "Fail")

# ===============================
# MACHINE LEARNING MODEL (REGRESSION VERSION)
# ===============================
mapping = {"A":1, "B":2, "C":3, "D":4}

df_ml = df.copy()

for q in answer_key:
    df_ml[q] = df_ml[q].map(mapping)

X = df_ml[list(answer_key.keys())]
y = df["SCORE"]

if len(df) > 3:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # FEATURE IMPORTANCE
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "Question": X.columns,
        "Importance": importance
    })

    df["PREDICTED_SCORE"] = model.predict(X)
    df["PREDICTED_SCORE"] = df["PREDICTED_SCORE"].round(2)

    df["PREDICTED_RESULT"] = df["PREDICTED_SCORE"].apply(
        lambda x: "Pass" if x >= 3 else "Fail"
    )

    y_pred = model.predict(X_test)
    model_accuracy = None

else:
    st.info("More filtered student data is needed for ML analysis.")
    df["PREDICTED_SCORE"] = 0
    df["PREDICTED_RESULT"] = "N/A"
    model_accuracy = None

# ===============================
# OVERALL STATS
# ===============================
st.subheader("📌 Overall Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", len(df))
col2.metric("Average Score", round(df["SCORE"].mean(), 2))
col3.metric("Highest Score", df["SCORE"].max())
col4.metric("Lowest Score", df["SCORE"].min())

st.markdown("---")

# ===============================
# ===============================
# SCORE DISTRIBUTION
# ===============================
st.subheader("📊 Score Distribution")

fig1 = plt.figure(figsize=(5,3))

sns.histplot(df["SCORE"], bins=5, kde=True)

plt.title("Distribution of Student Scores")
plt.xlabel("Score")
plt.ylabel("Number of Students")

st.pyplot(fig1)
# ===============================
# PASS VS FAIL
# ===============================
st.subheader("✅ Pass vs Fail")

result_counts = df["RESULT"].value_counts()

fig_pf = plt.figure(figsize=(4,4))

plt.pie(
    result_counts,
    labels=result_counts.index,
    autopct='%1.1f%%'
)

plt.title("Pass vs Fail Distribution")

st.pyplot(fig_pf)
# DEPARTMENT PERFORMANCE
# ===============================
st.subheader("🏢 Department Performance")

dept_perf = df.groupby("DEPARTMENT")["SCORE"].mean()

if len(dept_perf) > 0:

    fig2 = plt.figure(figsize=(5,3))

    dept_perf.plot(kind="bar")

    plt.title("Department-wise Average Score")
    plt.xlabel("Department")
    plt.ylabel("Average Score")

    st.pyplot(fig2)

else:
    st.warning("⚠ No department data available for selected filters.")
# QUESTION ANALYSIS
# ===============================
st.subheader("❓ Question Analysis")

question_accuracy = {}

for q in answer_key:
    correct = (df[q] == answer_key[q]).sum()
    question_accuracy[q] = correct / len(df)

question_df = pd.DataFrame.from_dict(
    question_accuracy,
    orient="index",
    columns=["Accuracy"]
)

fig4 = plt.figure(figsize=(5,3))

question_df["Accuracy"].plot(kind="bar")

plt.title("Question-wise Accuracy")
plt.xlabel("Questions")
plt.ylabel("Accuracy")

st.pyplot(fig4)

# BEST & HARDEST QUESTION
best_q = question_df["Accuracy"].idxmax()
worst_q = question_df["Accuracy"].idxmin()

st.write(f"✔ Easiest Question: {best_q}")
st.write(f"⚠ Hardest Question: {worst_q}")

# ===============================
# SMART INSIGHTS
# ===============================
st.subheader("🧠 Insights")

pass_rate = (df["RESULT"] == "Pass").mean() * 100
st.write(f"Overall Pass Rate: {round(pass_rate,2)}%")

# ===============================
# ⚠ WEAK STUDENTS DETECTION (NEW)
# ===============================
st.subheader("⚠ Students Needing Improvement")

weak_students = df[df["SCORE"] < 3]

if len(weak_students) > 0:
    st.dataframe(weak_students[["NAME", "DEPARTMENT", "SCORE"]])
else:
    st.success("No weak students 🎉")
    # ===============================
# RECOMMENDATION SYSTEM
# ===============================
st.subheader("🎯 Performance Recommendations")

avg_score = df["SCORE"].mean()

if avg_score >= 4:
    st.success("Excellent overall performance. Encourage advanced-level practice.")

elif avg_score >= 3:
    st.info("Students are performing well. Focus on improving weaker topics.")

else:
    st.warning("Performance needs improvement. More MCQ practice is recommended.")

# ===============================
# AI PREDICTION
# ===============================
st.subheader("🤖 AI Prediction")

st.dataframe(df[["NAME", "SCORE", "PREDICTED_SCORE", "PREDICTED_RESULT"]])

# ===============================

# FEATURE IMPORTANCE
# ===============================
st.subheader("📊 Feature Importance")

if len(df) > 3:

    fig_imp = plt.figure(figsize=(5,3))

    plt.bar(importance_df["Question"], importance_df["Importance"])

    plt.xlabel("Questions")
    plt.ylabel("Importance")
    plt.title("Feature Importance")

    st.pyplot(fig_imp)

else:
    st.info("Feature importance requires more filtered records.")

# ===============================
# 🔍 INDIVIDUAL STUDENT ANALYSIS
# ===============================
st.subheader("🔍 Individual Student Analysis")

student_name = st.selectbox("Select Student", df["NAME"].unique())
student_data = df[df["NAME"] == student_name]

st.dataframe(student_data)

score = int(student_data["SCORE"].values[0])
st.write("Score:", score)
st.write("Result:", student_data["RESULT"].values[0])

# ===============================
# TOP STUDENTS
# ===============================
st.subheader("🏆 Top Students")

top_students = df.sort_values("SCORE", ascending=False).head(5)

if has_college:
    st.dataframe(top_students[["NAME", "RANK", "DEPARTMENT", "COLLEGE", "SCORE"]])
else:
    st.dataframe(top_students[["NAME", "RANK", "DEPARTMENT", "SCORE"]])

# ===============================
# DOWNLOAD BUTTON
# ===============================
st.download_button(
    label="📥 Download Results as CSV",
    data=df.to_csv(index=False),
    file_name="mcq_results.csv",
    mime="text/csv"
)

# ===============================
# FULL DATA
# ===============================
st.subheader("📋 Full Data")
st.dataframe(df)
