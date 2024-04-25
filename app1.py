import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_excel("survey.xlsx")

X = data[["Access Minute","Take off Hour"]]
y = data['Company Name']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# عرض واجهة المستخدم
st.title('Flight Delay Prediction')

# إضافة مدخلات للمستخدم
takeoff_hour = st.slider("Take off Hour", 0, 23, 10)
access_hour = st.slider("Access Hour", 0, 23, 10)
airline = st.selectbox("Choose Airline", ["Fly Nas", "Saudi Airline", "SaudiGulf Airline", "Fly Adeal"])
#-----------
prediction_proba = model.predict_proba([[takeoff_hour, access_hour]])

threshold = 50  

result = "delayed" if prediction_proba[0][1] > threshold else "not delayed"

# عرض النتيجة
st.write(f'The model predicts with {round(prediction_proba[0][1]*100, 2)}% confidence that the flight of {airline} will be {result}.')
