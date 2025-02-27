import numpy as np
import pandas as pd

# Số lượng dữ liệu mẫu
num_samples = 100000

# Loại dữ liệu có thể có
data_types = ["Text", "Image", "JSON", "Blockchain TX"]

# Tạo dữ liệu ngẫu nhiên
np.random.seed(42)

dataset = {
    "Data Type": np.random.choice(data_types, num_samples),
    "Data Size (Bytes)": np.random.randint(100, 10000, num_samples),
    "Entropy": np.random.uniform(3.0, 8.0, num_samples),
    "Byte Distribution": [np.random.dirichlet(np.ones(10), size=1)[0].tolist() for _ in range(num_samples)],
    "Complexity Score": np.random.uniform(0.5, 1.0, num_samples),
    "Collision Rate": np.random.uniform(0.00001, 0.001, num_samples),
}

# Chọn thuật toán băm tối ưu dựa trên entropy và complexity score
def choose_hash_algorithm(entropy, complexity):
    if entropy > 6.5 and complexity > 0.85:
        return "BLAKE3"
    elif entropy > 5.5:
        return "SHA3-512"
    elif complexity > 0.75:
        return "Keccak-256"
    else:
        return "SHA-256"

dataset["Optimal Hash Algorithm"] = [
    choose_hash_algorithm(entropy, complexity)
    for entropy, complexity in zip(dataset["Entropy"], dataset["Complexity Score"])
]

# Chuyển dataset thành DataFrame
df = pd.DataFrame(dataset)

# Lưu thành CSV
df.to_csv("hash_algorithm_dataset.csv", index=False)

print("✅ Dataset đã tạo và lưu vào 'hash_algorithm_test_dataset.csv'")
