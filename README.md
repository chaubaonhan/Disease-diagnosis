# 🫀 Demo Dự đoán Bệnh tim

Đây là một ứng dụng web được xây dựng bằng Streamlit để demo khả năng dự đoán các bệnh lý tim mạch dựa trên dữ liệu điện tâm đồ (ECG) từ bộ dữ liệu PTB-XL.
Các mô hình Machine Learning đã được tối ưu hóa, huấn luyện trên bộ dữ liệu PTB-XL.

## Tính năng

-   **Hiển thị dữ liệu**: Xem trước dữ liệu từ các file `ptbxl_database.csv` và `scp_statements.csv`.
-   **Chọn bệnh nhân mẫu**: Lựa chọn giữa hai bệnh nhân mẫu để xem thông tin chi tiết.
-   **Hiển thị tín hiệu ECG**: Vẽ biểu đồ tín hiệu ECG 12 đạo trình cho bệnh nhân được chọn.
-   **Dự đoán bệnh lý**: Sử dụng 3 mô hình Keras đã được huấn luyện để dự đoán 5 loại bệnh lý (`NORM`, `MI`, `STTC`, `CD`, `HYP`).
-   **So sánh mô hình**: Trực quan hóa kết quả dự đoán của các mô hình trên một biểu đồ cột để dễ dàng so sánh.

## Cấu trúc thư mục

```
d:\Demo/
├── app.py                 # Mã nguồn chính của ứng dụng Streamlit
├── data.npz               # Dữ liệu đã xử lý (train/valid/test splits)
├── README.md              # File hướng dẫn này
├── requirements.txt       # Các thư viện Python cần thiết
├── .venv/                 # Thư mục môi trường ảo
├── database/              # Thư mục chứa dữ liệu gốc
│   ├── ptbxl_database.csv # Dữ liệu thông tin bệnh nhân
│   └── scp_statements.csv # Chú giải các mã SCP
├── model/                 # Thư mục chứa các mô hình đã huấn luyện
│   ├── model01.keras      # Mô hình 1 (dữ liệu dạng bảng)
│   ├── model02.keras      # Mô hình 2 (dữ liệu dạng bảng + ECG đầy đủ)
│   └── model03.keras      # Mô hình 3 (dữ liệu dạng bảng + 800 mẫu ECG)
└── test/                  # Thư mục chứa dữ liệu mẫu để demo
    ├── demo_ecg_76.npy    # Tín hiệu ECG mẫu cho bệnh nhân 76
    └── demo_ecg_8733.npy  # Tín hiệu ECG mẫu cho bệnh nhân 8733
```

## Hướng dẫn Cài đặt và Chạy

### 1. Yêu cầu cài đặt

Ứng dụng này sử dụng `tensorflow.keras.utils.plot_model` (trong file `test.py`) để vẽ kiến trúc mô hình, yêu cầu cài đặt **Graphviz**.

-   **Windows**:
    1.  Tải và cài đặt Graphviz từ trang chủ chính thức.
    2.  Thêm thư mục `bin` của Graphviz (ví dụ: `C:\Program Files\Graphviz\bin`) vào biến môi trường `PATH` của hệ thống.
-   **macOS** (sử dụng Homebrew):
    ```bash
    brew install graphviz
    ```
-   **Linux** (Ubuntu/Debian):
    ```bash
    sudo apt-get update
    sudo apt-get install graphviz
    ```

### 2. Cài đặt môi trường Python

Nên sử dụng một môi trường ảo để tránh xung đột thư viện.

```bash
# 1. Di chuyển đến thư mục dự án
cd D:\Demo

# 2. Tạo môi trường ảo
python -m venv .venv

# 3. Kích hoạt môi trường ảo
# Trên Windows
.venv\Scripts\activate
# Trên macOS/Linux
# source .venv/bin/activate

# 4. Cài đặt các thư viện cần thiết từ file requirements.txt
pip install -r requirements.txt
```

### 3. Chạy ứng dụng

Sau khi cài đặt thành công, chạy lệnh sau trong terminal (với môi trường ảo đã được kích hoạt):

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trong trình duyệt web của bạn.
