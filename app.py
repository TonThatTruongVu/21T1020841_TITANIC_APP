import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import zscore
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import mean_squared_error
import mlflow
import io
from sklearn.model_selection import KFold



import os
from mlflow.tracking import MlflowClient
def mlflow_input():
    st.title("MLflow DAGsHub Tracking với Streamlit")

    # Cấu hình DAGsHub MLflow URI
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/TonThatTruongVu/TITANIC_APP_Linear.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

    # Kiểm tra và khởi tạo giá trị session_state nếu chưa tồn tại
    if "mlflow_url" not in st.session_state:
        st.session_state["mlflow_url"] = DAGSHUB_MLFLOW_URI

    # Thiết lập biến môi trường (NÊN sử dụng file .env thay vì hardcode)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "TonThatTruongVu"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "aeb2dd8b26ef573bd0bc81a57d7cd8d55f87c3df"

    # Đặt thí nghiệm (nếu chưa tồn tại, tự động tạo)
    experiment_name = "Linear_replication"
    mlflow.set_experiment(experiment_name)

    st.success(f"✅ MLflow tracking đã thiết lập cho experiment: {experiment_name}")


def drop(df):
    st.subheader(" Xóa cột dữ liệu")
    
    if "df" not in st.session_state:
        st.session_state.df = df  # Lưu vào session_state nếu chưa có

    df = st.session_state.df
    columns_to_drop = st.multiselect(" Chọn cột muốn xóa:", df.columns.tolist())

    if st.button(" Xóa cột đã chọn"):
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)  # Tạo bản sao thay vì inplace=True
            st.session_state.df = df  # Cập nhật session_state
            st.success(f" Đã xóa cột: {', '.join(columns_to_drop)}")
            st.dataframe(df.head())
        else:
            st.warning(" Vui lòng chọn ít nhất một cột để xóa!")

    return df

def choose_label(df):
    st.subheader(" Chọn cột dự đoán (label)")

    if "target_column" not in st.session_state:
        st.session_state.target_column = None

    selected_label = st.selectbox(" Chọn cột dự đoán", df.columns, 
                                  index=df.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0)

    X, y = df.drop(columns=[selected_label]), df[selected_label]  # Mặc định
    
    if st.button(" Xác nhận Label"):
        st.session_state.target_column = selected_label
        X, y = df.drop(columns=[selected_label]), df[selected_label]
        st.success(f" Đã chọn cột: **{selected_label}**")
    
    return X, y

import streamlit as st
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split

def train_test_size():
    if "df" not in st.session_state:
        st.error("❌ Dữ liệu chưa được tải lên!")
        st.stop()
    
    df = st.session_state.df  # Lấy dữ liệu từ session_stat
    X, y = choose_label(df)
    
    st.subheader("📊 Chia dữ liệu Train - Validation - Test")   
    
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    remaining_size = 100 - test_size
    val_size = st.slider("📌 Chọn % dữ liệu Validation (trong phần Train)", 0, 50, 15)

    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Validation={val_size}%, Train={remaining_size - val_size}%")

    

    if st.button("✅ Xác nhận Chia"):
        # st.write("⏳ Đang chia dữ liệu...")

        stratify_option = y if y.nunique() > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        stratify_option = y_train_full if y_train_full.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # st.write(f"📊 Kích thước tập Train: {X_train.shape[0]} mẫu")
        # st.write(f"📊 Kích thước tập Validation: {X_val.shape[0]} mẫu")
        # st.write(f"📊 Kích thước tập Test: {X_test.shape[0]} mẫu")

        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_val = X_val  # ✅ Thêm X_val
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_val = y_val  # ✅ Thêm y_val
        st.session_state.y_test = y_test
        st.session_state.y = y
        st.session_state.X_train_shape = X_train.shape[0]
        st.session_state.X_val_shape = X_val.shape[0]
        st.session_state.X_test_shape = X_test.shape[0]
        summary_df = pd.DataFrame({
            "Tập dữ liệu": ["Train", "Validation", "Test"],
            "Số lượng mẫu": [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        })
        st.table(summary_df)

        # **Log dữ liệu vào MLflow**



def xu_ly_gia_tri_thieu(df):
    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    df = st.session_state.df

    # Tìm cột có giá trị thiếu
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        st.success("✅ Dữ liệu không có giá trị thiếu!")
        return df

    selected_col = st.selectbox("📌 Chọn cột chứa giá trị thiếu:", missing_cols)
    method = st.radio("🔧 Chọn phương pháp xử lý:", ["Thay thế bằng Mean", "Thay thế bằng Median", "Xóa giá trị thiếu"])
    

    
    if df[selected_col].dtype in ['int64', 'float64'] and method == "Thay thế bằng Mean":
        df[selected_col].fillna(df[selected_col].mean(), inplace=True)

      
    if df[selected_col].dtype in ['int64', 'float64'] and method == "Thay thế bằng Median":
        df[selected_col].fillna(df[selected_col].median(), inplace=True)
        
        
        
    if st.button(" Xử lý giá trị thiếu"):
        if df[selected_col].dtype == 'object':
            

            if method == "Thay thế bằng Mean":
                unique_values = df[selected_col].dropna().unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                df[selected_col] = df[selected_col].map(encoding_map)
                
                df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
            elif method == "Thay thế bằng Median":
                
                unique_values = df[selected_col].dropna().unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                df[selected_col] = df[selected_col].map(encoding_map)
            
                df[selected_col] = df[selected_col].fillna(df[selected_col].median())
            elif method == "Xóa giá trị thiếu":
                df = df.dropna(subset=[selected_col])
        else:
            if method == "Thay thế bằng Mean":
                df[selected_col] = df[selected_col].fillna(df[selected_col].mean())
            elif method == "Thay thế bằng Median":
                df[selected_col] = df[selected_col].fillna(df[selected_col].median())
            elif method == "Xóa giá trị thiếu":
                df = df.dropna(subset=[selected_col])
    
        st.session_state.df = df
        st.success(f" Đã xử lý giá trị thiếu trong cột `{selected_col}`")

    st.dataframe(df.head())
    return df





import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder


def chuyen_doi_kieu_du_lieu(df):
    st.subheader(" Chuyển đổi kiểu dữ liệu")

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        st.success(" Không có cột dạng chuỗi cần chuyển đổi!")
        return df

    selected_col = st.selectbox(" Chọn cột để chuyển đổi:", categorical_cols)
    unique_values = df[selected_col].unique()
     # Kiểm tra nếu cột chứa dữ liệu như "C85", "B42" → Áp dụng Label Encoding
    if all(any(char.isdigit() for char in str(val)) for val in unique_values):
        st.info("🔄 Cột chứa dữ liệu dạng chữ + số → Áp dụng Label Encoding.")

        # Áp dụng Label Encoding
        label_encoder = LabelEncoder()
        df[selected_col] = label_encoder.fit_transform(df[selected_col])

        # Lưu vào session_state
        st.session_state.df = df
        st.success(f"✅ Đã mã hóa cột `{selected_col}` thành số duy nhất (Label Encoding).")
        st.rerun()  

    # Khởi tạo session_state nếu chưa có
    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = {}

    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    mapping_dict = {}
    input_values = []  # Danh sách để kiểm tra trùng lặp
    has_duplicate = False  # Biến kiểm tra trùng lặp

    if len(unique_values) < 5:
        for val in unique_values:
            key = f"{selected_col}_{val}"
            if key not in st.session_state.text_inputs:
                st.session_state.text_inputs[key] = ""

            new_val = st.text_input(f" Nhập giá trị thay thế cho `{val}`:", 
                                    key=key, 
                                    value=st.session_state.text_inputs[key])

            # Cập nhật session_state với giá trị nhập mới
            st.session_state.text_inputs[key] = new_val
            input_values.append(new_val)

            # Lưu vào mapping_dict nếu không trùng lặp
            mapping_dict[val] = new_val

        # Kiểm tra nếu có giá trị trùng nhau
        duplicate_values = [val for val in input_values if input_values.count(val) > 1 and val != ""]
        if duplicate_values:
            has_duplicate = True
            st.warning(f"⚠ Giá trị `{', '.join(set(duplicate_values))}` đã được sử dụng nhiều lần. Vui lòng chọn số khác!")

        # Nút button bị mờ nếu có giá trị trùng lặp
        btn_disabled = has_duplicate

        if st.button(" Chuyển đổi dữ liệu", disabled=btn_disabled):
            # Lưu vào session_state
            column_info = {
                "column_name": selected_col,
                "mapping_dict": mapping_dict
            }
            st.session_state.mapping_dicts.append(column_info)

            df[selected_col] = df[selected_col].map(lambda x: mapping_dict.get(x, x))
            df[selected_col] = pd.to_numeric(df[selected_col], errors='coerce')

            # Reset text_inputs sau khi hoàn thành
            st.session_state.text_inputs.clear()

            st.session_state.df = df
            st.success(f" Đã chuyển đổi cột `{selected_col}`")

    st.dataframe(df.head())

    return df







def chuan_hoa_du_lieu(df):
    # st.subheader("📊 Chuẩn hóa dữ liệu với StandardScaler")

    # Lọc tất cả các cột số
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Tìm các cột nhị phân (chỉ chứa 0 và 1)
    binary_cols = [col for col in numerical_cols if df[col].dropna().isin([0, 1]).all()]

    # Loại bỏ cột nhị phân khỏi danh sách cần chuẩn hóa
    cols_to_scale = list(set(numerical_cols) - set(binary_cols))

    if not cols_to_scale:
        st.success(" Không có thuộc tính dạng số cần chuẩn hóa!")
        return df

    if st.button(" Thực hiện Chuẩn hóa"):
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

        # Lưu vào session_state
        st.session_state.df = df

        st.success(f" Đã chuẩn hóa các cột số (loại bỏ cột nhị phân): {', '.join(cols_to_scale)}")
        st.info(f" Giữ nguyên các cột nhị phân: {', '.join(binary_cols) if binary_cols else 'Không có'}")
        st.dataframe(df.head())

    return df

def hien_thi_ly_thuyet(df):
    st.subheader(" 10 dòng đầu của dữ liệu gốc")
    st.write(df.head(10))

                # Kiểm tra lỗi dữ liệu
    st.subheader(" Kiểm tra lỗi dữ liệu")

                # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()

                # Kiểm tra dữ liệu trùng lặp
    duplicate_count = df.duplicated().sum()

                
                
                # Kiểm tra giá trị quá lớn (outlier) bằng Z-score
    outlier_count = {
        col: (abs(zscore(df[col], nan_policy='omit')) > 3).sum()
        for col in df.select_dtypes(include=['number']).columns
    }

                # Tạo báo cáo lỗi
    error_report = pd.DataFrame({
        'Cột': df.columns,
        'Giá trị thiếu': missing_values,
        'Outlier': [outlier_count.get(col, 0) for col in df.columns]
    })

                # Hiển thị báo cáo lỗi
    st.table(error_report)

                # Hiển thị số lượng dữ liệu trùng lặp
    st.write(f" **Số lượng dòng bị trùng lặp:** {duplicate_count}")            
   
    
    st.title(" Tiền xử lý dữ liệu")

    # Hiển thị dữ liệu gốc
    
    st.subheader("1️⃣ Loại bỏ các cột không cần thiết")
    df=drop(df)
    
    st.subheader("2️⃣ Xử lý giá trị thiếu")
    df=xu_ly_gia_tri_thieu(df)

    st.subheader("3️⃣ Chuyển đổi kiểu dữ liệu")
    st.write("""
        Trong dữ liệu, có một số cột chứa giá trị dạng chữ (category). Ta cần chuyển đổi thành dạng số để mô hình có thể xử lý.
        - **Cột "Sex"**: Chuyển thành 1 (male), 0 (female).
        - **Cột "Embarked"**:   Chuyển thành 1 (Q), 2 (S), 3 (C).
        """)

    df=chuyen_doi_kieu_du_lieu(df)
    
    st.subheader("4️⃣ Chuẩn hóa dữ liệu số")
    st.write("""
        Các giá trị số có thể có khoảng giá trị khác nhau, làm ảnh hưởng đến mô hình. Ta sẽ chuẩn hóa toàn bộ về cùng một thang đo bằng StandardScaler.
        """)

    
    df=chuan_hoa_du_lieu(df)
    
def chia():
    st.subheader("Chia dữ liệu thành tập Train, Validation, và Test")
    st.write("""
    ### 📌 Chia tập dữ liệu
    Dữ liệu được chia thành ba phần để đảm bảo mô hình tổng quát tốt:
    - **70%**: để train mô hình.
    - **15%**: để validation, dùng để điều chỉnh tham số.
    - **15%**: để test, đánh giá hiệu suất thực tế.
    """)
       
    train_test_size()
    
    


def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""
    
    # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra NaN hoặc Inf
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_train.shape
    #st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1) vào X_train
    X_b = np.c_[np.ones((m, 1)), X_train]
    #st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    #st.write(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra xem gradients có NaN không
        # st.write(gradients)
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    #st.success("✅ Huấn luyện hoàn tất!")
    #st.write(f"Trọng số cuối cùng: {w.flatten()}")
    return w
def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

    # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_poly.shape
    print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    print(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01  
    print(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra nếu gradient có giá trị NaN
        if np.isnan(gradients).any():
            raise ValueError("Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    print("✅ Huấn luyện hoàn tất!")
    print(f"Trọng số cuối cùng: {w.flatten()}")
    
    return w



# Hàm chọn mô hình
def chon_mo_hinh():
    st.subheader("🔍 Chọn mô hình hồi quy")
    
    # 🔹 Khởi tạo run_name nếu chưa có
    if 'run_name' not in st.session_state:
        st.session_state['run_name'] = f"run_{np.random.randint(1000, 9999)}"
    
    model_type_V = st.radio("Chọn loại mô hình:", ["Multiple Linear Regression", "Polynomial Regression"])
    model_type = "linear" if model_type_V == "Multiple Linear Regression" else "polynomial"
    
    n_folds = st.slider("Chọn số folds (KFold Cross-Validation):", min_value=2, max_value=10, value=5)
    learning_rate = st.slider("Chọn tốc độ học (learning rate):", 
                          min_value=1e-6, max_value=0.1, value=0.01, step=1e-6, format="%.6f")

    degree = 2
    if model_type == "polynomial":
        degree = st.slider("Chọn bậc đa thức:", min_value=2, max_value=5, value=2)

    fold_mse = []
    scaler = StandardScaler()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # 🔹 Kiểm tra dữ liệu đã được chia hay chưa
    if "X_train" not in st.session_state or st.session_state["X_train"] is None:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi huấn luyện mô hình!")
        return None, None, None

    X_train, X_test = st.session_state["X_train"], st.session_state["X_test"]
    y_train, y_test = st.session_state["y_train"], st.session_state["y_test"]

    if st.button("Huấn luyện mô hình"):
        # 🔹 Kiểm tra và thiết lập mlflow_url nếu chưa có
        if "mlflow_url" not in st.session_state:
            st.session_state["mlflow_url"] = "https://dagshub.com/TonThatTruongVu/TITANIC_APP_Linear.mlflow"

        try:
            with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}_{model_type}"):

                mlflow.log_param("model_type", model_type)
                mlflow.log_param("n_folds", n_folds)
                mlflow.log_param("learning_rate", learning_rate)
                if model_type == "polynomial":
                    mlflow.log_param("degree", degree)

                for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
                    X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                    y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                    if model_type == "linear":
                        w = train_multiple_linear_regression(X_train_fold, y_train_fold, learning_rate=learning_rate)
                        w = np.array(w).reshape(-1, 1)
                        X_valid_b = np.c_[np.ones((len(X_valid), 1)), X_valid.to_numpy()]
                        y_valid_pred = X_valid_b.dot(w)
                    else:  
                        X_train_fold = scaler.fit_transform(X_train_fold)
                        w = train_polynomial_regression(X_train_fold, y_train_fold, degree, learning_rate=learning_rate)
                        w = np.array(w).reshape(-1, 1)
                        X_valid_scaled = scaler.transform(X_valid.to_numpy())
                        X_valid_poly = np.hstack([X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
                        X_valid_b = np.c_[np.ones((len(X_valid_poly), 1)), X_valid_poly]
                        y_valid_pred = X_valid_b.dot(w)

                    mse = mean_squared_error(y_valid, y_valid_pred)
                    fold_mse.append(mse)
                    mlflow.log_metric(f"mse_fold_{fold+1}", mse)
                    print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

                avg_mse = np.mean(fold_mse)

                if model_type == "linear":
                    final_w = train_multiple_linear_regression(X_train, y_train, learning_rate=learning_rate)
                    st.session_state['linear_model'] = final_w
                    X_test_b = np.c_[np.ones((len(X_test), 1)), X_test.to_numpy()]
                    y_test_pred = X_test_b.dot(final_w)
                else:
                    X_train_scaled = scaler.fit_transform(X_train)
                    final_w = train_polynomial_regression(X_train_scaled, y_train, degree, learning_rate=learning_rate)
                    st.session_state['polynomial_model'] = final_w
                    X_test_scaled = scaler.transform(X_test.to_numpy())
                    X_test_poly = np.hstack([X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
                    X_test_b = np.c_[np.ones((len(X_test_poly), 1)), X_test_poly]
                    y_test_pred = X_test_b.dot(final_w)

                test_mse = mean_squared_error(y_test, y_test_pred)

                # 🔹 **Log dữ liệu vào MLflow**
                mlflow.log_metric("avg_mse", avg_mse)
                mlflow.log_metric("test_mse", test_mse)

                st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
                st.success(f"MSE trên tập test: {test_mse:.4f}")
                st.success(f"✅ Đã log dữ liệu cho **Train_{st.session_state['run_name']}_{model_type}**!")
                st.markdown(f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")

        except Exception as e:
            st.error(f"⚠ Lỗi khi huấn luyện mô hình: {e}")

        finally:
            mlflow.end_run()

        return final_w, avg_mse, scaler

    return None, None, None


import numpy as np
import streamlit as st


from datetime import datetime
def get_mlflow_runs():
    """Lấy danh sách các runs từ MLflow và hiển thị dữ liệu mới nhất"""
    client = mlflow.tracking.MlflowClient()
    experiment_id = "0"  # Thay bằng ID thực tế của bạn
    runs = client.search_runs(experiment_id, order_by=["start_time DESC"])

    if not runs:
        return None

    # Chuyển đổi danh sách runs thành DataFrame
    run_data = []
    for run in runs:
        start_time_ms = run.info.start_time  # Thời gian dạng milliseconds
        start_time_dt = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")  # Chuyển đổi

        run_data.append({
            "Run ID": run.info.run_id,
            "Run Name": run.data.tags.get("mlflow.runName", "Unnamed Run"),
            "Start Time": start_time_dt,  # Hiển thị thời gian đẹp hơn
            "Status": run.info.status,
            "Accuracy": run.data.metrics.get("accuracy", "N/A"),
            "Test MSE": run.data.metrics.get("test_mse", "N/A"),
            "Prediction Result": run.data.params.get("prediction_result", "N/A")
        })

    return pd.DataFrame(run_data)

def test():
    """Chức năng dự đoán và hiển thị kết quả"""
    # Kiểm tra xem mô hình đã được lưu trong session_state chưa
    model_type = st.selectbox("Chọn mô hình:", ["linear", "polynomial"])

    if model_type == "linear" and "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
    elif model_type == "polynomial" and "polynomial_model" in st.session_state:
        model = st.session_state["polynomial_model"]
    else:
        st.warning("Mô hình chưa được huấn luyện.")
        return

    # Nhập các giá trị cho các cột của X_train
    X_train = st.session_state.X_train
    num_columns = len(X_train.columns)
    column_names = X_train.columns.tolist()

    st.write(f"Nhập các giá trị cho {num_columns} cột của X_train:")

    # Tạo các trường nhập liệu cho từng cột
    X_train_input = []
    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    for i, column_name in enumerate(column_names):
        mapping_dict = next(
            (col["mapping_dict"] for col in st.session_state.mapping_dicts if col["column_name"] == column_name), None
        )

        if mapping_dict:
            value = st.selectbox(f"Giá trị cột {column_name}", options=list(mapping_dict.keys()), key=f"column_{i}")
            value = int(mapping_dict[value])
        else:
            value = st.number_input(f"Giá trị cột {column_name}", key=f"column_{i}")

        X_train_input.append(value)

    # Chuyển đổi list thành array
    X_train_input = np.array(X_train_input).reshape(1, -1)

    # Chuẩn hóa dữ liệu
    X_train_input_final = X_train_input.copy()
    scaler = StandardScaler()

    for i in range(X_train_input.shape[1]):
        if X_train_input[0, i] not in [0, 1]:  # Nếu giá trị không phải 0 hoặc 1
            X_train_input_final[0, i] = scaler.fit_transform(X_train_input[:, i].reshape(-1, 1)).flatten()

    st.write("Dữ liệu sau khi xử lý:")

    if st.button("Dự đoán"):
        # Thêm cột 1 cho intercept (nếu cần)
        X_input_b = np.c_[np.ones((X_train_input_final.shape[0], 1)), X_train_input_final]
        y_pred = X_input_b.dot(model)  # Dự đoán với mô hình đã lưu

        # Hiển thị kết quả dự đoán
        st.write("SỐNG" if y_pred >= 0.5 else "CHẾT")

        # Hiển thị danh sách các runs từ MLflow
        df_runs = get_mlflow_runs()
        if df_runs is not None:
            st.subheader("📊 Các Runs Đã Lưu Trong MLflow")
            st.dataframe(df_runs)

def data():
    uploaded_file = st.file_uploader("📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, delimiter=",")
            st.success("📂 File tải lên thành công!")

            # Hiển thị lý thuyết và xử lý dữ liệu
            hien_thi_ly_thuyet(df)
        except Exception as e:
            st.error(f"❌ Lỗi : {e}")
            
import streamlit as st
import mlflow
import os


          
def chon():
    try:
                
        final_w, avg_mse, scaler = chon_mo_hinh()
    except Exception as e:
        st.error(f"Lỗi xảy ra: {e}")
def main():

# Chọn tab bằng radio button
    st.title("TITANIC APP LINEAR REGRESSION")
    option = st.radio("Chọn chức năng:", 
                  ["📘 Tiền xử lý dữ liệu", "⚙️ Huấn luyện", "🔢 Dự đoán",])
    
# Hiển thị nội dung tương ứng
    if option == "📘 Tiền xử lý dữ liệu":
        data()
    elif option == "⚙️ Huấn luyện":
        chia()
        chon()
    elif option == "🔢 Dự đoán":
        test()

    
            

        
if __name__ == "__main__":
    main()
    
        

        


            
  
