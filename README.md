# 🧠 Hash Algorithm Predictor using Deep Learning

Dự án này xây dựng một hệ thống học sâu để dự đoán **thuật toán băm tối ưu** dựa trên các đặc trưng thống kê của dữ liệu như:
- Loại dữ liệu (Data Type)
- Kích thước (Data Size)
- Entropy
- Phân bố byte (Byte Distribution)
- Độ phức tạp (Complexity Score)
- Tỷ lệ va chạm (Collision Rate)

## 🚀 Mục tiêu
Xác định thuật toán băm tối ưu nhất cho mỗi mẫu dữ liệu (SHA-256, SHA3-512, Keccak-256, BLAKE3) nhằm:
- Tối ưu hiệu suất bảo mật
- Giảm tỷ lệ va chạm
- Phù hợp với tính chất dữ liệu đầu vào

---

## 🛠️ Công nghệ sử dụng
- Python 3.x
- TensorFlow / Keras
- Pandas, NumPy
- Scikit-learn

---

## 🗂 Cấu trúc dự án
```
.
├── hash_algorithm_dataset.csv        # Dataset huấn luyện
├── hash_algorithm_test_dataset.csv   # Dataset kiểm thử
├── hash_algorithm_model.h5           # Mô hình học sâu đã huấn luyện
├── model_train.py                    # Script huấn luyện mô hình
├── predict.py                        # Script dự đoán đầu vào mới
├── README.md                         # Tài liệu dự án
```
🔍 Dự đoán với mô hình đã huấn luyện
Chạy script predict.py để đưa ra gợi ý:


```
python predict.py
```

Ví dụ: Dự đoán thuật toán băm phù hợp với dữ liệu mới có entropy 4.5, complexity 0.75...

🔹 Thuật toán băm được gợi ý: SHA3-512
📈 Độ chính xác mô hình
Mô hình đạt độ chính xác ~95% trên tập kiểm tra (tuỳ thuộc vào dữ liệu đầu vào và huấn luyện).

💾 Lưu ý
Cần đảm bảo Byte Distribution được parse đúng kiểu (list of float).

File mô hình .h5 cần được đặt đúng đường dẫn nếu sử dụng Google Drive hoặc môi trường đám mây.

📌 TODO
 Tối ưu hóa mô hình bằng kỹ thuật GridSearch/RandomSearch

 Tạo giao diện web với Flask hoặc Streamlit

 Hỗ trợ thêm thuật toán băm như Whirlpool, MD6

📚 Tài liệu tham khảo
Keras Documentation

NIST: Hash Functions

BLAKE3 Spec

👤 Tác giả
github.com/MinhKhongCau