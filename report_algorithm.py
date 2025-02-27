# Test model 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import classification_report, confusion_matrix

# Load mô hình đã huấn luyện
model_path = "/content/drive/MyDrive/dataset/hash_algorithm_model.h5"
model = tf.keras.models.load_model(model_path)

# Load tập kiểm tra
test_data_path = "/content/drive/MyDrive/dataset/hash_algorithm_test_dataset.csv"
df_test = pd.read_csv(test_data_path)

df_test["Byte Distribution"] = df_test["Byte Distribution"].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

# Mã hóa cột "Optimal Hash Algorithm" thành số
hash_algorithms = ["SHA3-512", "BLAKE3", "SHA-256", "Keccak-256"]
df_test["Optimal Hash Algorithm"] = df_test["Optimal Hash Algorithm"].apply(lambda x: hash_algorithms.index(x))

# Mã hóa cột "Datatype" thành số
data_type = ["Random Bytes", "Blockchain TX", "JSON", "Image", "Text"]
df_test["Data Type"] = df_test["Data Type"].apply(lambda x: data_type.index(x) if x in data_type else -1)


# Chuẩn bị dữ liệu đầu vào
X_test = df_test.drop(columns=["Optimal Hash Algorithm"])
y_test = df_test["Optimal Hash Algorithm"]

X_test["Byte Distribution"] = X_test["Byte Distribution"].apply(lambda x: np.array(x) if isinstance(x, list) else np.zeros(256))

byte_distribution_expanded = pd.DataFrame(X_test["Byte Distribution"].to_list(), index=X_test.index)
X_test = X_test.drop(columns=["Byte Distribution"]).join(byte_distribution_expanded)

X_test.columns = X_test.columns.astype(str)

# Chuẩn hóa dữ liệu
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Dự đoán trên tập kiểm tra
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# In báo cáo hiệu suất
print("📊 Báo cáo đánh giá mô hình:")
print(classification_report(y_test, y_pred, target_names=["SHA3-512", "BLAKE3", "SHA-256", "Keccak-256"]))

# Biểu đồ "Độ chính xác dự đoán theo thuật toán băm"
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Tính số lượng mẫu đúng trên tổng số mẫu của mỗi thuật toán
accuracy_per_class = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

# Vẽ biểu đồ đường
plt.figure(figsize=(8, 5))
plt.plot(hash_algorithms, accuracy_per_class, marker='o', linestyle='-', color='b', label="Độ chính xác")

# Định dạng biểu đồ
plt.xlabel("Thuật toán băm")
plt.ylabel("Độ chính xác")
plt.title("Độ chính xác dự đoán theo thuật toán băm")
plt.ylim(0, 1)  # Giá trị trong khoảng 0-1
plt.grid(True)
plt.legend()
plt.show()

# Biểu đồ "Tần suất dự đoán theo thuật toán băm"
import collections

# Đếm số lượng dự đoán mỗi thuật toán
pred_counts = collections.Counter(y_pred)

# Chuyển đổi thành danh sách theo thứ tự thuật toán băm
pred_frequencies = [pred_counts[i] for i in range(len(hash_algorithms))]

# Vẽ biểu đồ đường
plt.figure(figsize=(8, 5))
plt.plot(hash_algorithms, pred_frequencies, marker='s', linestyle='-', color='g', label="Tần suất dự đoán")

# Định dạng biểu đồ
plt.xlabel("Thuật toán băm")
plt.ylabel("Số lần được dự đoán")
plt.title("Tần suất dự đoán theo thuật toán băm")
plt.grid(True)
plt.legend()
plt.show()
