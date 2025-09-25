# Để đọc file CSV
import pandas as pd

# Để chia dữ liệu
from sklearn.model_selection import train_test_split

# Để xử lý nhãn và chuẩn hóa dữ liệu
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Để huấn luyện mô hình
from sklearn.linear_model import LogisticRegression

# Để đánh giá
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Vẽ mô hình
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc dữ liệu
file_name = 'student.csv'
df = pd.read_csv(file_name)

# Hiển thị dữ liệu
print({file_name})
df.head()

# Sửa tên cột bị dính ký tự lạ
df.columns = df.columns.str.strip()

# Chuyển đổi 2 cột grade về dạng số (nếu có lỗi thì thành NaN)
for col in ["Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Điền NaN bằng trung bình cột
df.fillna(df.mean(numeric_only=True), inplace=True)

# Tách các đặc trưng (X) và biến mục tiêu (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Chuyển đổi tất cả các cột trong X sang kiểu số (float)
X = X.astype(float)

# Mã hóa biến mục tiêu 'Target' từ dạng chữ ('Dropout', 'Enrolled', 'Graduate')
# thành dạng số (0, 1, 2) để mô hình có thể học.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# In ra các lớp và cách chúng được mã hóa để tham khảo
class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

# Chia dữ liệu thành 80% cho huấn luyện và 20% cho kiểm thử.
# random_state=42 để đảm bảo kết quả có thể tái lập.
# stratify=y_encoded để giữ nguyên tỉ lệ phân bổ của các lớp trong cả tập train và test.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Chuẩn hóa dữ liệu.
# Dùng StandardScaler để đưa tất cả các đặc trưng về cùng một thang đo (trung bình 0, phương sai 1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tăng max_iter lên 1000 để đảm bảo thuật toán có đủ số lần lặp để hội tụ.
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Dùng mô hình đã huấn luyện để dự đoán trên tập kiểm thử.
y_pred = model.predict(X_test_scaled)

# Tính toán và in ra độ chính xác.
accuracy = accuracy_score(y_test, y_pred)

# In báo cáo phân loại chi tiết bao gồm precision, recall, f1-score cho từng lớp.
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

print(f"Độ chính xác (Accuracy) của mô hình: {accuracy:.2%}")
print("\nClassification Report:")
print(report)

# Ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)

# Vẽ contour + dữ liệu thực
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.title("Confusion Matrix Logistic Regression")
plt.show()
