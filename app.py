import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Income Insights Dashboard",
    page_icon="💼",
    layout="wide"
)

pipeline = joblib.load("adult_income_pipeline.pkl")

# Header
st.markdown(
"""
# 💼 Income Insights Dashboard
Predict whether an individual earns **more than $50K annually** based on demographic and employment characteristics.
"""
)

st.divider()

st.subheader("📊 Enter Individual Information")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 18, 100)

with col2:
    education = st.selectbox(
        "Education Level",
        ["Bachelors","HS-grad","Some-college","Masters","Doctorate"]
    )

with col3:
    hours_per_week = st.number_input("Hours Worked Per Week", 1, 100)

col4, col5 = st.columns(2)

with col4:
    occupation = st.selectbox(
        "Occupation",
        ["Exec-managerial","Prof-specialty","Craft-repair","Sales","Tech-support"]
    )

with col5:
    workclass = st.selectbox(
        "Workclass",
        ["Private","Self-emp","Government"]
    )

st.divider()

if st.button("🔍 Analyze Income Potential"):

    input_data = pd.DataFrame({
        "age":[age],
        "workclass":[workclass],
        "fnlwgt":[0],
        "education":[education],
        "education.num":[0],
        "marital.status":["Never-married"],
        "occupation":[occupation],
        "relationship":["Not-in-family"],
        "race":["White"],
        "sex":["Male"],
        "capital.gain":[0],
        "capital.loss":[0],
        "hours.per.week":[hours_per_week],
        "native.country":["United-States"]
    })

    prediction = pipeline.predict(input_data)
    probability = pipeline.predict_proba(input_data)[0][1]

    st.subheader("📈 Income Prediction")

    colA, colB = st.columns(2)

    with colA:
        if prediction[0] == ">50K":
            st.success("💰 High Income Potential (> $50K)")
        else:
            st.info("Income likely ≤ $50K")

    with colB:
        st.metric("Probability of High Income", f"{probability:.2f}")

    st.progress(probability)

    if probability < 0.3:
        st.warning("Low probability of earning above $50K")
    elif probability < 0.6:
        st.info("Moderate probability of earning above $50K")
    else:
        st.success("Strong probability of earning above $50K")