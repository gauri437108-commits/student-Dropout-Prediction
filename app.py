import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("student_data.csv")

X = df[["Attendance","Marks","StudyHours"]]
y = df["Dropout"]

model = DecisionTreeClassifier()
model.fit(X, y)

st.title("🎓 Student Dropout Prediction")

attendance = st.number_input("Attendance (%)",0,100,75)
marks = st.number_input("Marks",0,100,70)
study = st.number_input("Study Hours",0,12,4)

if st.button("Predict"):

    result = model.predict([[attendance, marks, study]])

    if result[0] == 1:
        st.error("⚠ Student is likely to Drop Out")
    else:
        st.success("✅ Student is Not Likely to Drop Out")
