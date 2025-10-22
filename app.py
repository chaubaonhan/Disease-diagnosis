
# app.py
import streamlit as st
import pandas as pd
import os
import time
import numpy as np
import tensorflow as tf # Thêm cho Keras models
import matplotlib.pyplot as plt # Thêm cho vẽ biểu đồ
import ast
import json
import io

def set_page_style():
    """
    Hàm này chèn CSS để thay đổi màu nền và một số kiểu khác.
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #e0f7fa; /* Màu xanh dương nhạt (cyan-lighten-5) */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Cấu hình trang ---
# st.set_page_config() phải là lệnh Streamlit đầu tiên trong script của bạn.
st.set_page_config(layout="wide") # Sử dụng toàn bộ chiều rộng trang

# Áp dụng màu nền và style tùy chỉnh
set_page_style()

# Thiết lập tiêu đề cho ứng dụng
st.title("🫀 Demo dự đoán bệnh tim")
st.markdown("""
Dự đoán nguy cơ mắc bệnh tim từ dữ liệu bệnh nhân với các mô hình Machine Learning đã được tối ưu hóa, huấn luyện trên bộ dữ liệu Cleveland.
*   **Bộ dữ liệu:** PTB-XL
*   **Các mô hình:** 1D Convolution Neural Network
""")


# --- PHẦN XỬ LÝ VÀ HIỂN THỊ DỮ LIỆU ---

# Định nghĩa đường dẫn đến các file CSV
db_file = 'D:\Demo\database\ptbxl_database.csv'
scp_file = 'D:\Demo\database\scp_statements.csv'

# 1. Hiển thị dữ liệu từ ptbxl_database.csv
st.header(f"Dữ liệu từ `{db_file}`")
if os.path.exists(db_file):
    try:
        # Đọc file CSV
        # Đổi tên file merge.csv thành ptbxl_database.csv nếu cần
        df_db = pd.read_csv(db_file, index_col=0)
        # Hiển thị 10 dòng đầu tiên
        st.dataframe(df_db.head(10))
    except Exception as e:
        st.error(f"Lỗi khi đọc file {db_file}: {e}")
else:
    st.warning(f"Không tìm thấy file `{db_file}`. Vui lòng đặt file vào cùng thư mục với `app.py`.")

# 2. Hiển thị dữ liệu từ scp_statements.csv
st.header(f"Dữ liệu từ `{scp_file}`")
if os.path.exists(scp_file):
    try:
        # Đọc file CSV
        df_scp = pd.read_csv(scp_file, index_col=0)
        # Hiển thị 10 dòng đầu tiên
        st.dataframe(df_scp.head(10))
    except Exception as e:
        st.error(f"Lỗi khi đọc file {scp_file}: {e}")
else:
    st.warning(f"Không tìm thấy file `{scp_file}`. Vui lòng đặt file vào cùng thư mục với `app.py`.")

# --- Model Loading ---
# Sử dụng st.cache_resource để tải mô hình chỉ một lần
@st.cache_resource
def load_keras_models():
    model_paths = {
        "model01": r'D:\Demo\model\model01.keras',
        "model02": r'D:\Demo\model\model02.keras',
        "model03": r'D:\Demo\model\model03.keras'
    }
    loaded_models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                loaded_models[name] = tf.keras.models.load_model(path)
                st.success(f"Đã tải {name} từ `{path}`.")
            except Exception as e:
                st.error(f"Lỗi khi tải {name} từ `{path}`: {e}")
        else:
            st.warning(f"Không tìm thấy file mô hình `{path}` cho {name}.")
    return loaded_models

st.header("Thực hiện dự đoán")
NUMPY_DATA_FILE = 'D:\Demo\database\data.npz'

data = np.load(NUMPY_DATA_FILE, allow_pickle=True)

X_train = data['X_train']
Y_train = data['Y_train']
Z_train = data['Z_train']

X_valid = data['X_valid']
Y_valid = data['Y_valid']
Z_valid = data['Z_valid']

X_test = data['X_test']
Y_test = data['Y_test']
Z_test = data['Z_test']
ECG_df = pd.read_csv('D:\Demo\database\ptbxl_database.csv', index_col='ecg_id')

# Load models at the start
models = load_keras_models()
ECG_df.scp_codes = ECG_df.scp_codes.apply(lambda x: ast.literal_eval(x))
ECG_df.patient_id = ECG_df.patient_id.astype(int)
ECG_df.nurse = ECG_df.nurse.astype('Int64')
ECG_df.site = ECG_df.site.astype('Int64')
ECG_df.validated_by = ECG_df.validated_by.astype('Int64')
demo_numpy_idx=[8,1000]
# Lấy ecg_id tương ứng từ test set
ECG_test_df = ECG_df[ECG_df.strat_fold == 10].copy()
ECG_test_df = ECG_test_df.reset_index(drop=False)  # giữ ecg_id gốc
demo_indices = ECG_test_df.loc[demo_numpy_idx, 'ecg_id'].tolist()
demo_info = ECG_df.loc[demo_indices]  # tất cả cột

labels = ['NORM','MI','STTC','CD','HYP']

demo_labels = pd.DataFrame(Z_test[demo_numpy_idx], columns=labels, index=demo_indices)
demo_full = pd.concat([demo_info, demo_labels], axis=1)

# Tạo dữ liệu mẫu từ demo_full để người dùng lựa chọn
example_patients = {
    f"Bệnh nhân (ecg_id: {demo_full.index[0]})": demo_full.iloc[0].to_dict(),
    f"Bệnh nhân (ecg_id: {demo_full.index[1]})": demo_full.iloc[1].to_dict()
}

# Tạo mapping từ ecg_id của bệnh nhân mẫu đến chỉ số numpy gốc trong X_test
# Điều này cần thiết để truy cập tín hiệu ECG từ X_test nếu không có file .npy cụ thể
ecg_id_to_demo_numpy_idx = {ecg_id: original_idx for ecg_id, original_idx in zip(demo_indices, demo_numpy_idx)}

# Định nghĩa đường dẫn đến các file tín hiệu ECG cụ thể mà người dùng đã cung cấp
# VUI LÒNG ĐẢM BẢO CÁC FILE NÀY TỒN TẠI TẠI ĐƯỜNG DẪN ĐƯỢC CHỈ ĐỊNH.
DEMO_ECG_FILE_1 = r'D:\Demo\test\demo_ecg_76.npy'
DEMO_ECG_FILE_2 = r'D:\Demo\test\demo_ecg_8733.npy'

ecg_id_to_specific_file = {
    demo_full.index[0]: DEMO_ECG_FILE_1,
    demo_full.index[1]: DEMO_ECG_FILE_2
}

# Tạo hai cột cho việc chọn mẫu và nút dự đoán
pred_col1, pred_col2 = st.columns([2, 1])

with pred_col1:
    # Dropdown để chọn bệnh nhân mẫu
    selected_patient_key = st.selectbox(
        "Chọn bệnh nhân mẫu để xem thông tin",
        options=list(example_patients.keys())
    )
    # Lấy dữ liệu của bệnh nhân được chọn
    patient_data = example_patients[selected_patient_key]
    
    # Hiển thị dữ liệu của bệnh nhân đã chọn
    st.write("Thông tin bệnh nhân:")
    st.json(patient_data)

with pred_col2:
    st.write("") # Thêm khoảng trống
    st.write("Nhấn nút dưới đây để hiển thị tín hiệu ECG của bệnh nhân được chọn.")
    plot_ecg_button = st.button("🧬 Hiển thị ECG", key="plot_ecg_button") # Đổi key để tránh trùng lặp nếu có nút khác

    st.write("") # Thêm khoảng trống
    st.write("Nhấn nút dưới đây để dự đoán bệnh lý ECG.")
    predict_disease_button = st.button("🧠 Dự đoán bệnh lý", key="predict_disease_button")

if plot_ecg_button:
    st.info(f"Đã chọn phân tích cho bệnh nhân: {selected_patient_key}")

    # Trích xuất ecg_id từ selected_patient_key
    ecg_id_str = selected_patient_key.split(': ')[1].replace(')', '')
    selected_ecg_id = int(ecg_id_str)

    st.subheader(f"Biểu đồ tín hiệu ECG cho ecg_id: {selected_ecg_id}")

    ecg_signal = None
    source_info = ""

    # 1. Thử tải tín hiệu ECG từ các file .npy cụ thể mà người dùng đã cung cấp
    specific_file_path = ecg_id_to_specific_file.get(selected_ecg_id)
    if specific_file_path and os.path.exists(specific_file_path):
        try:
            ecg_signal = np.load(specific_file_path)
            source_info = f"từ file `{specific_file_path}`"
            st.success(f"Đã tải tín hiệu ECG {source_info}.")
        except Exception as e:
            st.error(f"Lỗi khi tải file ECG `{specific_file_path}`: {e}. Thử tải từ X_test.")
            ecg_signal = None # Reset signal if loading failed

    # 2. Nếu không tải được từ file cụ thể, hoặc file không tồn tại, thử tải từ X_test
    if ecg_signal is None:
        numpy_idx_for_selected_ecg = ecg_id_to_demo_numpy_idx.get(selected_ecg_id)
        if numpy_idx_for_selected_ecg is not None:
            ecg_signal = X_test[numpy_idx_for_selected_ecg]
            source_info = "từ tập dữ liệu X_test"
            st.success(f"Đã tải tín hiệu ECG {source_info}.")
        else:
            st.error("Không tìm thấy dữ liệu ECG trong X_test cho bệnh nhân này.")

    # 3. Hiển thị tín hiệu ECG nếu đã tải thành công
    if ecg_signal is not None:
        # --- Xử lý định dạng tín hiệu ECG ---
        st.write(f"Định dạng dữ liệu gốc {source_info}: `{ecg_signal.shape}`")
        
        # Nếu là 3D array (ví dụ: (1, 12, 1000)), loại bỏ chiều không cần thiết
        if ecg_signal.ndim == 3 and ecg_signal.shape[0] == 1:
            ecg_signal = ecg_signal.squeeze(0)
            st.info(f"Đã điều chỉnh định dạng thành: `{ecg_signal.shape}`")

        # Nếu định dạng là (1000, 12), chuyển vị thành (12, 1000)
        if ecg_signal.ndim == 2 and ecg_signal.shape[1] == 12:
            ecg_signal = ecg_signal.T
            st.info(f"Đã chuyển vị và điều chỉnh định dạng thành: `{ecg_signal.shape}`")

        # --- Vẽ biểu đồ sau khi đã chuẩn hóa định dạng ---
        # Kiểm tra lại định dạng cuối cùng có phải là (12, N) hay không
        if ecg_signal.ndim == 2 and ecg_signal.shape[0] == 12:
            # Chuyển đổi sang DataFrame để Streamlit có thể vẽ biểu đồ đường
            # Transpose để mỗi cột là một đạo trình, mỗi hàng là một thời điểm
            ecg_df_plot = pd.DataFrame(ecg_signal.T, columns=[f'Lead {i+1}' for i in range(ecg_signal.shape[0])])
            st.line_chart(ecg_df_plot)
        
        else:
            st.warning(f"Không thể vẽ biểu đồ. Định dạng tín hiệu ECG cuối cùng (`{ecg_signal.shape}`) không phù hợp (cần có 12 đạo trình).")
            st.write("Dữ liệu thô:", ecg_signal)
        # Nút xuất file
        output_file_path = f"demo_{selected_ecg_id}_ecg.npy"
    
    # Sử dụng st.download_button để người dùng click là tải file về
        try:
            # Chuyển dữ liệu ECG sang bytes
            ecg_bytes = io.BytesIO()
            np.save(ecg_bytes, ecg_signal)
            ecg_bytes.seek(0)  # quay về đầu file để đọc

            st.download_button(
                label=f"Tải tín hiệu ECG (demo_{selected_ecg_id}_ecg.npy)",
                data=ecg_bytes,
                file_name=f"demo_{selected_ecg_id}_ecg.npy",
                mime="application/octet-stream"
            )
        except Exception as e:
            st.error(f"Lỗi khi chuẩn bị file tải xuống: {e}")
    else:
        st.error("Không thể tải tín hiệu ECG cho bệnh nhân này từ bất kỳ nguồn nào.")

if predict_disease_button:
    if not models:
        st.error("Không có mô hình nào được tải. Vui lòng kiểm tra đường dẫn file mô hình.")
    else:
        st.info(f"Đang dự đoán bệnh lý cho bệnh nhân: {selected_patient_key}")

        # Trích xuất ecg_id từ selected_patient_key
        ecg_id_str = selected_patient_key.split(': ')[1].replace(')', '')
        selected_ecg_id = int(ecg_id_str)

        # Lấy chỉ số numpy gốc từ mapping
        original_idx_in_test_set = ecg_id_to_demo_numpy_idx.get(selected_ecg_id)

        if original_idx_in_test_set is None:
            st.error(f"Không tìm thấy dữ liệu cho ecg_id {selected_ecg_id} trong tập test để dự đoán.")
        else:
            # Chuẩn bị dữ liệu đầu vào cho các mô hình
            X_demo = X_test[original_idx_in_test_set][np.newaxis, :]
            Y_demo = Y_test[original_idx_in_test_set][np.newaxis, :]

            predictions = {}

            # Dự đoán với model01 (Tabular)
            if "model01" in models:
                try:
                    pred01 = models["model01"].predict(X_demo, verbose=0)
                    predictions["Model01 (Tabular)"] = pred01[0] * 100
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán với Model01: {e}")

            # Dự đoán với model02 (Tabular + ECG)
            if "model02" in models:
                try:
                    pred02 = models["model02"].predict([X_demo, Y_demo], verbose=0)
                    predictions["Model02 (Tabular+ECG)"] = pred02[0] * 100
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán với Model02: {e}")

            # Dự đoán với model03 (ECG)
            if "model03" in models:
                try:
                    pred03 = models["model03"].predict([X_demo, Y_demo[:, :800, :]], verbose=0)
                    predictions["Model03 (Tabular+ECG expand)"] = pred03[0] * 100
                except Exception as e:
                    st.error(f"Lỗi khi dự đoán với Model03: {e}")

            if predictions:
                st.subheader("Kết quả dự đoán từ các mô hình")

                # Vẽ biểu đồ cột
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(labels))
                width = 0.25
                model_names = list(predictions.keys())
                num_models = len(model_names)
                bar_positions = [x + (i - (num_models - 1) / 2) * width for i in range(num_models)]
                for i, model_name in enumerate(model_names):
                    ax.bar(bar_positions[i], predictions[model_name], width, label=model_name)
                ax.set_ylabel('Xác suất (%)')
                ax.set_title(f'Dự đoán bệnh lý cho bệnh nhân (ecg_id: {selected_ecg_id})')
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
                ax.legend()
                ax.set_ylim(0, 100)
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)
                plt.close(fig)

                # --- Chuẩn hóa patient_data: NaN → "Không rõ", None → "Không có", nhãn 0.0/1.0 → "?" ---
                def clean_value(val, key=None):
                    if key in ['NORM', 'MI', 'STTC', 'CD', 'HYP'] and (val == 0.0 or val == 1.0):
                        return "?"
                    elif pd.isna(val) or val == "unknown":
                        return "Không rõ"
                    elif val is None:
                        return "Không có"
                    else:
                        return val

                patient_info_cleaned = {k: clean_value(v, k) for k, v in patient_data.items()}

                # --- Tạo JSON (không có nhãn thật) ---
                json_data = {
                    "ecg_id": selected_ecg_id,
                    "patient_info": patient_info_cleaned,
                    "predictions": {model: [f"{p:.2f}%" for p in vals] for model, vals in predictions.items()}
                }

                # --- Nút download JSON ---
                st.download_button(
                    label="📥 Xuất file JSON",
                    data=json.dumps(json_data, indent=4, ensure_ascii=False),
                    file_name=f"prediction_ecg_{selected_ecg_id}.json",
                    mime="application/json"
                )
            else:
                st.warning("Không có dự đoán nào được tạo ra. Vui lòng kiểm tra các mô hình đã được tải.")
