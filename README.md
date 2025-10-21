# 🫀 Demo Dự đoán Bệnh tim

Đây là một ứng dụng web được xây dựng bằng Streamlit để demo khả năng dự đoán các bệnh lý tim mạch dựa trên dữ liệu điện tâm đồ (ECG) từ bộ dữ liệu PTB-XL.
Các mô hình 1D Convolution Neural Network đã được tối ưu hóa, huấn luyện trên bộ dữ liệu PTB-XL.

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
### 3. Đánh giá các mô hình

| Tên mô hình | Độ chính xác | Kiến trúc |
| :--- | :--- | :--- |
| Model 1 | 81% | [MLP (5 Lớp Dense)](model01_architecture.png) |
| Model 2 | 90% | [CNN 1D (3 Conv) + MLP (2 Dense) + 3 Dense (1000 steps)](https://github.com/chaubaonhan/Disease-diagnosis/blob/main/model02_architecture.png) |
| Model 3 | 89.6% | [CNN 1D (3 Conv) + MLP (2 Dense) + 3 Dense (800 step)](https://github.com/chaubaonhan/Disease-diagnosis/blob/main/model03_architecture.png) |

### 4. Chạy ứng dụng

Sau khi cài đặt thành công, chạy lệnh sau trong terminal (với môi trường ảo đã được kích hoạt):

```bash
streamlit run app.py
```

Ứng dụng sẽ mở trong trình duyệt web của bạn.

### 5. Giao diện web 

Phần đầu sẽ tả về database đó là thông tin bệnh nhân

<img width="1260" height="381" alt="image" src="https://github.com/user-attachments/assets/19c446ae-88a5-4f43-b57b-35e053cb6c75" />

Dữ liệu về nhãn bệnh

<img width="1262" height="380" alt="image" src="https://github.com/user-attachments/assets/6398c82b-0ff2-49dd-9971-576c3c1f63b0" />

Hình ảnh về ECG của bệnh nhân 

<img width="1249" height="292" alt="image" src="https://github.com/user-attachments/assets/ef372468-2c27-4644-8623-3682d0207711" />

Có chức năng thể hiện số không nhất thiết phải vẽ 


Cuối cùng là dự đoán của 3 mô hình 

<img width="1460" height="905" alt="00b86c8e857b59cb4c6e7e9598272294934a35304c8d637b1a9f454c" src="https://github.com/user-attachments/assets/5f35d07b-304d-4226-b4f2-0a5bc25bc9c9" />


### 6. Thông tin về Database PTL-XB 
https://drive.google.com/drive/folders/1RoHQ5ZOElYm378oMAqw7R-3PzroYV6qP?usp=sharing






