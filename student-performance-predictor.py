import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load and preprocess dataset
df = pd.read_csv("student-performance.csv")
df['Extracurricular Activities'] = LabelEncoder().fit_transform(df['Extracurricular Activities'])

# Features and target
X = df.drop('Performance Index', axis=1)
y = df['Performance Index']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Support Vector Regressor": SVR()
}
for name in models:
    models[name].fit(X_scaled, y)

# Streamlit UI
st.title("🎓 Student Performance Predictor")

# --- Data Visualizations ---
st.header("📊 Data Visualizations")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
sns.histplot(df['Performance Index'], kde=True, ax=axes[0, 0]).set_title("Performance Index Distribution")
sns.scatterplot(data=df, x='Hours Studied', y='Performance Index', ax=axes[0, 1]).set_title(
    "Hours Studied vs Performance")
sns.scatterplot(data=df, x='Previous Scores', y='Performance Index', ax=axes[0, 2]).set_title(
    "Previous Scores vs Performance")
sns.boxplot(data=df, x='Extracurricular Activities', y='Performance Index', ax=axes[1, 0]).set_title(
    "Extracurricular vs Performance")
sns.scatterplot(data=df, x='Sleep Hours', y='Performance Index', ax=axes[1, 1]).set_title("Sleep Hours vs Performance")
sns.scatterplot(data=df, x='Sample Question Papers Practiced', y='Performance Index', ax=axes[1, 2]).set_title(
    "Sample Papers vs Performance")
st.pyplot(fig)

# --- User Input Section ---
st.sidebar.header("Enter Student Information:")
hours_studied = st.sidebar.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.sidebar.slider("Previous Scores", 0, 100, 75)
extracurricular = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"])
sleep_hours = st.sidebar.slider("Sleep Hours", 0, 12, 7)
sample_papers = st.sidebar.slider("Sample Question Papers Practiced", 0, 100, 10)

# --- Prediction Section ---
if st.sidebar.button("Predict Performance"):
    extracurricular_encoded = 1 if extracurricular.lower() == "yes" else 0
    input_df = pd.DataFrame([{
        "Hours Studied": hours_studied,
        "Previous Scores": previous_scores,
        "Extracurricular Activities": extracurricular_encoded,
        "Sleep Hours": sleep_hours,
        "Sample Question Papers Practiced": sample_papers
    }])

    input_scaled = scaler.transform(input_df)

    st.header("📈 Predicted Student Performance Index")
    for name, model in models.items():
        pred = model.predict(input_scaled)[0]
        st.write(f"**{name}**: {pred:.2f}")
