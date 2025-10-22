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
```markdown
📁 .devcontainer/
│   └── 📄 devcontainer.json        # Cấu hình môi trường phát triển tự động trong VS Code Dev Containers (Docker)
📁 database/
│   ├── 📄 data.npz                 # Dữ liệu đã được xử lý và nén (numpy array, thường là train/test/valid splits)
│   ├── 📄 merge.csv                # File tổng hợp thông tin bệnh nhân sau khi merge nhiều nguồn dữ liệu
│   ├── 📄 ptbxl_database.csv       # CSDL chính từ PTB-XL (thông tin metadata: tuổi, giới, nhãn ECG, v.v.)
│   └── 📄 scp_statements.csv       # Bảng ánh xạ giữa nhãn SCP và loại bệnh tim (NORM, MI, STTC, CD, HYP)
📁 experiment/
│   └── 📄 Heart_diagnosis.ipynb    # Notebook dùng để huấn luyện, đánh giá và trực quan hóa mô hình chẩn đoán ECG
📁 model/
│   ├── 📄 model01_architecture.png # Sơ đồ kiến trúc mô hình 01 (ví dụ CNN 1D hoặc LSTM)
│   ├── 📄 model01.keras            # Trọng số mô hình 01 đã được huấn luyện (định dạng Keras)
│   ├── 📄 model02_architecture.png # Sơ đồ kiến trúc mô hình 02 (ví dụ mô hình kết hợp meta + ECG)
│   ├── 📄 model02.keras            # Trọng số mô hình 02 đã huấn luyện
│   ├── 📄 model03_architecture.png # Sơ đồ kiến trúc mô hình 03 (phiên bản thử nghiệm hoặc cải tiến)
│   └── 📄 model03.keras            # Trọng số mô hình 03
📁 test/
│   ├── 📄 demo_ecg_76.npy          # Dữ liệu ECG mẫu (id = 76) dùng cho demo/predict
│   ├── 📄 demo_ecg_8733.npy        # Dữ liệu ECG mẫu khác (id = 8733)
│   ├── 📄 prediction_ecg_76.json   # Kết quả dự đoán của mô hình cho demo_ecg_76
│   └── 📄 prediction_ecg_8733.json # Kết quả dự đoán của mô hình cho demo_ecg_8733
📄 .gitattributes                   # Thiết lập thuộc tính Git (ví dụ: xử lý dòng, LFS, text/binary)
📄 app.py                           # Ứng dụng Streamlit hiển thị giao diện chẩn đoán ECG và kết quả dự đoán
📄 README.md                        # Tài liệu mô tả dự án (giới thiệu, hướng dẫn cài đặt, demo, v.v.)
📄 requirements.txt                 # Danh sách thư viện Python cần cài đặt để chạy dự án
```
<a href="https://githubtree.mgks.dev/repo/chaubaonhan/Disease-diagnosis/main/" target="_blank">Hướng dẫn cách tạo cấu trúc thư mục</a>

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

<img width="1362" height="596" alt="image" src="https://github.com/user-attachments/assets/6bd51b8c-c2ed-4f09-9198-7534b70476a2" />


Có chức năng thể hiện số không nhất thiết phải vẽ 

<img width="1343" height="461" alt="image" src="https://github.com/user-attachments/assets/fd7c6fc8-790a-41b8-bb23-d18316573337" />


Cuối cùng là dự đoán của 3 mô hình 

<img width="1460" height="905" alt="00b86c8e857b59cb4c6e7e9598272294934a35304c8d637b1a9f454c" src="https://github.com/user-attachments/assets/5f35d07b-304d-4226-b4f2-0a5bc25bc9c9" />

[Mô phỏng chuẩn đoán của bác sĩ](https://disease-diagnosis-kqfcwdbwukt6jd2jsvrv97.streamlit.app/)

### 6. Thông tin về Database PTL-XB 
https://drive.google.com/drive/folders/1RoHQ5ZOElYm378oMAqw7R-3PzroYV6qP?usp=sharing












