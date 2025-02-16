import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#  Đặt URI cho MLflow Server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

#  Đặt tên thí nghiệm MLflow
experiment_name = "Titanic_Survival_Prediction"
mlflow.set_experiment(experiment_name)

#  Load dataset từ GitHub
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#  Xử lý dữ liệu
df = df.drop(columns=["PassengerId","Name", "Ticket", "Cabin"])  # Xóa các cột không cần thiết
df = df.dropna()  # Xóa hàng chứa giá trị thiếu

#  Chuyển đổi dữ liệu categorical thành số
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"])

#  Chia tập dữ liệu: Train (70%) / Valid (15%) / Test (15%)
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#  Huấn luyện mô hình với MLflow Tracking
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    #  Cross Validation trên Train + Valid
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)

    model.fit(X_train, y_train)

    # Đánh giá trên tập Test
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    #  Ghi log vào MLflow
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("cv_accuracy", mean_cv_score)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f" Model trained and logged with Test Accuracy: {test_accuracy:.4f}")
