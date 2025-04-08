import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def main():
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

    # 1. Histogram for Performance Index
    axes[0, 0].hist(df['Performance Index'], bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Performance Index Distribution")
    axes[0, 0].set_xlabel("Performance Index")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Scatter: Hours Studied vs Performance Index
    axes[0, 1].scatter(df['Hours Studied'], df['Performance Index'], color='green', alpha=0.6)
    axes[0, 1].set_title("Hours Studied vs Performance")
    axes[0, 1].set_xlabel("Hours Studied")
    axes[0, 1].set_ylabel("Performance Index")

    # 3. Scatter: Previous Scores vs Performance Index
    axes[0, 2].scatter(df['Previous Scores'], df['Performance Index'], color='purple', alpha=0.6)
    axes[0, 2].set_title("Previous Scores vs Performance")
    axes[0, 2].set_xlabel("Previous Scores")
    axes[0, 2].set_ylabel("Performance Index")

    # 4. Box plot: Extracurricular Activities vs Performance Index
    extracurricular_groups = [df[df['Extracurricular Activities'] == val]['Performance Index'] for val in [0, 1]]
    axes[1, 0].boxplot(extracurricular_groups, labels=["No", "Yes"])
    axes[1, 0].set_title("Extracurricular vs Performance")
    axes[1, 0].set_xlabel("Extracurricular Activities")
    axes[1, 0].set_ylabel("Performance Index")

    # 5. Scatter: Sleep Hours vs Performance Index
    axes[1, 1].scatter(df['Sleep Hours'], df['Performance Index'], color='orange', alpha=0.6)
    axes[1, 1].set_title("Sleep Hours vs Performance")
    axes[1, 1].set_xlabel("Sleep Hours")
    axes[1, 1].set_ylabel("Performance Index")

    # 6. Scatter: Sample Papers vs Performance Index
    axes[1, 2].scatter(df['Sample Question Papers Practiced'], df['Performance Index'], color='red', alpha=0.6)
    axes[1, 2].set_title("Sample Papers vs Performance")
    axes[1, 2].set_xlabel("Sample Question Papers Practiced")
    axes[1, 2].set_ylabel("Performance Index")

    plt.tight_layout()
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
