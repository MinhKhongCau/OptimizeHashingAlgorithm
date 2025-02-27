import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "hash_algorithm_dataset.csv"

# Đọc dataset
df = pd.read_csv(file_path)

# Mã hóa cột "Data Type" thành số
label_encoder = LabelEncoder()
df["Data Type"] = label_encoder.fit_transform(df["Data Type"])

# Chuyển đổi cột "Byte Distribution" từ chuỗi thành mảng số
df["Byte Distribution"] = df["Byte Distribution"].apply(eval)
byte_distribution_cols = pd.DataFrame(df["Byte Distribution"].tolist())
df = pd.concat([df, byte_distribution_cols], axis=1).drop(columns=["Byte Distribution"])

# Mã hóa cột "Optimal Hash Algorithm" thành số
hash_algorithms = ["SHA3-512", "BLAKE3", "SHA-256", "Keccak-256"]
df["Optimal Hash Algorithm"] = df["Optimal Hash Algorithm"].apply(lambda x: hash_algorithms.index(x))

# Chia dữ liệu thành tập huấn luyện và kiểm tra
X = df.drop(columns=["Optimal Hash Algorithm"])
y = df["Optimal Hash Algorithm"]

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Xây dựng mô hình học sâu
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(4, activation='softmax')  # 4 thuật toán băm
])

# Compile mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Đánh giá mô hình
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Độ chính xác mô hình: {accuracy * 100:.2f}%")

# Lưu mô hình
model.save("hash_algorithm_model.h5")
print("✅ Mô hình đã được lưu thành 'hash_algorithm_model.h5'")
