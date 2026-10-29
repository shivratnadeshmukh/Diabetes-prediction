import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle
import openai

def load_model():
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def predict_diabetes(model, features):
    prediction = model.predict([features])
    return prediction[0]

from transformers import pipeline

# Load a free local LLM model
chatbot = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

def chatbot_response(prompt):
    response = chatbot(prompt, max_length=200, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response


# Streamlit App
st.title("Diabetes Prediction App with Chatbot")

# Sidebar for Chatbot
st.sidebar.header("💬 Diabetes Chatbot")
user_input = st.sidebar.text_input("Ask me anything about diabetes!")
if st.sidebar.button("Send") and user_input:
    response = chatbot_response(user_input)
    st.sidebar.text_area("Response:", value=response, height=200)

df = pd.read_csv("diabetes.csv")

# HEADINGS
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X AND Y DATA
x = df.drop(['Outcome'], axis=1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# FUNCTION
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)
st.write("User Data Columns:", user_data.columns.tolist())

# MODEL
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# VISUALISATIONS
st.title('Visualised Patient Report')
user_data.columns = user_data.columns.str.lower()

# COLOR FUNCTION
color = 'blue' if user_result[0] == 0 else 'red'

# OUTPUT
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(f"{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%")

# Button for Precaution Report
if st.button("Show Precaution Report"):
    st.subheader('Precaution Report:')
    if user_result[0] == 0:
        st.write("""
            - Maintain a healthy lifestyle by eating a balanced diet rich in fruits, vegetables, and whole grains.
            - Exercise regularly for at least 30 minutes a day to maintain optimal weight and fitness.
            - Monitor your glucose levels periodically to ensure they remain within a healthy range.
            - Avoid excessive sugar intake and processed foods.
            - Get regular health checkups to prevent potential health risks.
        """)
    else:
        st.write("""
            - Consult a doctor and follow prescribed medications or insulin therapy if advised.
            - Focus on a diet with low carbohydrates, low sugar, and high fiber.
            - Incorporate regular physical activities like walking, yoga, or strength training into your routine.
            - Monitor your blood sugar levels frequently to manage fluctuations.
            - Avoid smoking, alcohol, and high-stress environments.
            - Ensure regular foot care and watch for any cuts or infections, as diabetes can impact healing.
            - Stay hydrated and get enough sleep to improve overall health.
        """)

# Risk Level Analysis
st.subheader('Risk Level Analysis')
if user_result[0] == 0:
    if user_data['glucose'].values[0] < 140 and user_data['bmi'].values[0] < 25:
        st.write("Low Risk: Maintain a healthy lifestyle.")
    else:
        st.write("Moderate Risk: Consider regular checkups and lifestyle adjustments.")
else:
    st.write("High Risk: Immediate medical attention and lifestyle changes are advised.")

# Visualizations
visualizations = [
    ('Pregnancy count Graph', 'Pregnancies', 'Greens'),
    ('Glucose Value Graph', 'Glucose', 'magma'),
    ('Blood Pressure Value Graph', 'BloodPressure', 'Reds'),
    ('Skin Thickness Value Graph', 'SkinThickness', 'Blues'),
    ('Insulin Value Graph', 'Insulin', 'rocket'),
    ('BMI Value Graph', 'BMI', 'rainbow'),
    ('DPF Value Graph', 'DiabetesPedigreeFunction', 'YlOrBr')
]

for title, feature, palette in visualizations:
    st.header(f'{title} (Others vs Yours)')
    fig = plt.figure()
    sns.scatterplot(x='Age', y=feature, data=df, hue='Outcome', palette=palette)
    sns.scatterplot(x=user_data['age'], y=user_data[feature.lower()], s=150, color=color)
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig)
