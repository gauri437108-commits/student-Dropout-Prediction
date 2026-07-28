import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("student_data.csv")

X = df[["Attendance","Marks","StudyHours"]]
y = df["Dropout"]

model = DecisionTreeClassifier()
model.fit(X, y)
