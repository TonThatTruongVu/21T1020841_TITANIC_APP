import streamlit as st
import pickle
import pandas as pd

# Load model từ file .pkl
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Giao diện Streamlit
st.title("Titanic Survival Prediction")

st.write("Nhập thông tin hành khách để dự đoán khả năng sống sót:")

# Form nhập dữ liệu người dùng
pclass = st.selectbox("Hạng vé (1: First, 2: Second, 3: Third)", [1, 2, 3])
sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
age = st.number_input("Tuổi", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Số anh/chị/em đi cùng", min_value=0, max_value=10, value=0)
parch = st.number_input("Số cha/mẹ/con đi cùng", min_value=0, max_value=10, value=0)
fare = st.number_input("Giá vé", min_value=0, value=50)
embarked = st.selectbox("Cảng đi", ["Cherbourg", "Queenstown", "Southampton"])

# Chuyển đổi dữ liệu đầu vào
sex = 1 if sex == "Nữ" else 0
embarked_C = 1 if embarked == "Cherbourg" else 0
embarked_Q = 1 if embarked == "Queenstown" else 0
embarked_S = 1 if embarked == "Southampton" else 0

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex": [sex],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_C": [embarked_C],
    "Embarked_Q": [embarked_Q],
    "Embarked_S": [embarked_S],
})

# Dự đoán khi nhấn nút
if st.button("Dự đoán"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("Hành khách có khả năng **SỐNG SÓT**.")
    else:
        st.error("Hành khách có khả năng **KHÔNG SỐNG SÓT**.")
