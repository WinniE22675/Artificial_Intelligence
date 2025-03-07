import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout
from keras.api.optimizers import Adam
import shap
import seaborn as sns
from sklearn.decomposition import PCA

# โหลดข้อมูล
path = "Project_AI/updated_pollution_dataset.csv"
data = pd.read_csv(path)

# แปลงค่า Label (Air Quality) เป็นตัวเลข
label_encoder = LabelEncoder()
data['Air Quality'] = label_encoder.fit_transform(data['Air Quality'])
# เก็บชื่อคลาส Good Hazardous Moderate Poor ลง class_names
class_names = label_encoder.classes_ 

# แยก Features และ Target
X = data.drop(columns=['Air Quality']).values  # เอาแค่ส่วน Features
y = data['Air Quality'].values  # เอาแค่ Target

# แบ่งข้อมูลเป็น Train/Test
# ทดสอบ train 90% test 10%, train 80% test 20%, train 70% test 30%
# ผลที่ดีสุดเป็น train 80% test 20% epochs=50 batch_size=32
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

# มาตรฐานข้อมูล (Standardization)
# ทำให้ค่าของแต่ละ feature อยู่ในช่วงที่เหมาะสม ลดปัญหา bias
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# สร้าง Neural Network model (FNN)
# ทดสอบ Dense 128 64, 64 32, 32 16
# 128 64 เยอะไป ผลลัพธ์เกิด Overfitting, 32 16 รู้สึกน้อยไปและถ้าเทียบกับ 64 32 ได้ผลลัพธ์ดีกว่า
# Dense ที่ใช้เป็น 64 กับ 32
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden Layer 1
    Dropout(0.3),  # ลด Overfitting โดยสุ่มปิด 30% ของ Neurons
    Dense(32, activation='relu'),  # Hidden Layer 2
    Dense(len(np.unique(y)), activation='softmax')  # Output Layer ใช้ Softmax สำหรับ Multi-class Classification ให้ค่าอยู่ในช่วง 0-1
])

# Compile model
# sparse_categorical_crossentropy เป็น loss function ที่เหมาะกับ multi-class classification และ Target ที่เรากำหนดเป็นตัวเลข
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
# ทดสอบ epochs 30, 50, 70, 100 และ batch_size 16, 32, 64
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# ประเมินผลลัพธ์
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss * 100:.2f}%')
print(f'Accuracy: {accuracy * 100:.2f}%')

# ทดสอบโมเดล
# ใช้ np.argmax เลือกคลาสที่โมเดลพยากรณ์ว่ามีความเป็นไปได้สูงสุด (ค่ามากที่สุดของในแถว axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)

# แสดงผลตรวจสอบความแม่นยำของโมเดล
# ใช้วัด ประสิทธิภาพของโมเดล
# Precision >> ความแม่นยำของแต่ละคลาส (เช่น ค่าที่โมเดลทำนายว่า "Poor" จริง ๆ ถูกต้องแค่ไหน)
# Recall >> ความสามารถในการจับค่าของแต่ละคลาส (ค่าจริงของ "Poor" ถูกจับได้ทั้งหมดไหม)
# F1-score >> ค่าเฉลี่ยของ Precision และ Recall
# Support >> จำนวนตัวอย่างของแต่ละคลาส
# Macro Average >> ค่าเฉลี่ยของแต่ละคลาส (ไม่สนใจขนาดคลาส)
# Weighted Average >> ค่าเฉลี่ยแบบ ถ่วงน้ำหนักตามขนาดคลาส
print(classification_report(y_test, y_pred, target_names=class_names))

'''

# แสดง Confusion Matrix
# เพื่อทดสอบว่าโมเดลสามารถใช้งานได้จริงแค่ไหน โดยแปลงค่าผลลัพธ์กลับเป็นหมวดหมู่ (Good, Moderate, Poor, Hazardous)
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

'''

# Plot Decision Boundary (ใช้ PCA ลดมิติ)
pc1_label = "Principal Component 1 (PC1)"
pc2_label = "Principal Component 2 (PC2)"
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test)
xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100),
                     np.linspace(X_pca[:, 1].min(), X_pca[:, 1].max(), 100))
Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=[class_names[i] for i in y_test], palette='Set1')
plt.xlabel(pc1_label)
plt.ylabel(pc2_label)
plt.title("Decision Boundary")
plt.show()
print(pd.DataFrame(pca.components_, columns=data.drop(columns=['Air Quality']).columns, index=['PC1', 'PC2']))


'''
# เช็ค Overfitting หรือ Underfitting
# Accuracy ระหว่าง Train กับ Test
# Overfitting >> Train Accuracy สูง แต่ Test Accuracy ต่ำมาก 
# Underfitting >> Train & Test Accuracy ต่ำ
train_acc = model.evaluate(X_train, y_train, verbose=0)[1]  # [loss, accuracy] >> [1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[1]     # [loss, accuracy] >> [1]
print(f'Training Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}')

# Overfitting >> Training loss ต่ำ แต่ Validation loss สูง
# Underfitting >> ทั้ง Training และ Validation loss สูง
# ดึงค่าจาก history
loss = history.history['loss']               # Training Loss
val_loss = history.history['val_loss']       # Validation Loss
epochs = range(1, len(loss) + 1)             # จำนวน Epochs
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.show()


# Feature Importance using SHAP
# ใช้โมเดลที่ฝึกแล้วมาสร้าง Explainer
explainer = shap.Explainer(model, X_train)

# ดูว่าแต่ละ Feature มีผลต่อการพยากรณ์มากแค่ไหน
shap_values = explainer(X_test)

# Plot กราฟผลรวม
# สีแดง: ดันค่าพยากรณ์ให้สูง
# สีฟ้า: ลดค่าพยากรณ์
# Feature สำคัญ >> จุดเยอะและกระจายมาก
# Feature มีผลน้อย >> จุดแคบและเรียงกัน
# # แสดง Feature Importance ของทุกคลาสในกราฟเดียว
# shap.summary_plot(shap_values, features=X_test, feature_names=data.drop(columns=['Air Quality']).columns)

# แสดง Feature Importance ตาม Class
for i, class_name in enumerate(class_names):
    print(f"SHAP Feature Importance for {class_name}")
    shap.summary_plot(shap_values[:, :, i], features=X_test, feature_names=data.drop(columns=['Air Quality']).columns)


# ข้อมูลใหม่ (สมมติค่าจาก Sensor)
new_data = np.array([[26.2,49.9,0.5,2,22.9,2.3,0.94,17.9,581]])  # ค่า Feature เช่น อุณหภูมิ, ความชื้น ฯลฯ

# ทำ Standardization เหมือนกับที่ใช้ตอน Train
new_data_scaled = scaler.transform(new_data)

# ทำนายค่า Air Quality
prediction = np.argmax(model.predict(new_data_scaled), axis=1)
predicted_label = class_names[prediction[0]]  # แปลงกลับเป็นชื่อคลาส

'''

# คำถาม ?
# 1. ทำไมต้องเลือก ReLU แล้วถ้าเกิดได้ค่าติดลบต้องทำยังไง แก้ยังไง
# Ans: ให้ค่า Gradient เป็นค่าติดลบเล็ก ๆ แทนศูนย์ เช่น 
#   - Leaky ReLU f(x)=max(αx,x)
#   - Parametric ReLU (PReLU) คล้าย Leaky ReLU แต่ให้ α เรียนรู้ค่าเองจากข้อมูล
# =================================================================================================
# 2. Decision Boundary ในโค้ด แกน x และ แกน y คืออะไร?
# Ans: ใช้ PCA มาลดมิติ โดยคิดคำนวณจากทุก Features เลยทำให้แกน X และ Y เป็น
#
# xx (แกน X) → PC1 ได้ค่ามาจากการรวมกันของฟีเจอร์ทั้งหมด (Temperature,Humidity,PM2.5,PM10,NO2,SO2,CO,Proximity_to_Industrial_Areas,Population_Density)
# เป็นทิศทางที่ข้อมูลมีความกระจายมากที่สุด
# yy (แกน Y) → PC2 ก็เป็นการรวมกันอีกแบบของฟีเจอร์ทั้งหมด (ได้จากนำค่าของฟีเจอร์ทั้งหมดมาคำนวณ)
# ใช้อธิบายความแตกต่างที่เหลืออยู่
#
# คำนวณ PCA
# 1. ทำ Standardization (ทำให้ค่าเฉลี่ยเป็น 0 และสเกลเป็น 1)
# 2. จาก Covariance Matrix → หา Eigenvalues, Eigenvectors
# 3. Eigenvectors คือค่าที่กำหนดว่าแต่ละ Feature มีผลต่อ PC1, PC2 แค่ไหน
# 4. เราเลือก PC1, PC2 ตาม Explained Variance
# 5. ค่าในตาราง บอกว่าแต่ละ Feature มีอิทธิพลต่อ PC1, PC2 มากแค่ไหน
# จาก code ที่ได้
# PC1 = มลพิษโดยรวม, PC2 = ความเข้มข้นของ PM2.5/PM10
# =================================================================================================
# 3. SHAP ทำงานยังไง? SHAP values คืออะไร?
# Ans: 
#   1. ใช้ SHAP Explainer กับโมเดลที่ Train แล้ว
#   2. คำนวณค่า SHAP values
#       เปรียบเทียบโมเดลที่ มี และ ไม่มี Feature นั้น ๆ
#   คำนวณว่า Feature แต่ละตัวมีผลต่อค่าพยากรณ์ของโมเดลแค่ไหน
#   3. Summary Plot แสดงผลกระทบของแต่ละ Feature
#       แกน Y >> Feature ต่าง ๆ ที่ใช้ในโมเดล
#       แกน X >> SHAP Value ที่แสดงผลกระทบของ Feature ต่อโมเดล
#       จุด (Dot) >> ค่าของแต่ละตัวอย่างข้อมูล (Instance) ในชุดข้อมูล
#       แดง (High Value) = ค่าของ Feature สูง
#       ฟ้า (Low Value) = ค่าของ Feature ต่ำ
#       จุดที่กระจายไปทางขวา (SHAP Value บวก) แสดงว่า Feature นั้นมีแนวโน้มเพิ่มค่าพยากรณ์
#       จุดที่กระจายไปทางซ้าย (SHAP Value ลบ) แสดงว่า Feature นั้นมีแนวโน้มลดค่าพยากรณ์
#       ถ้าสีแดงอยู่ทางขวา >> ถ้าค่าของ Feature สูง จะเพิ่มค่าพยากรณ์
#       ถ้าสีฟ้าอยู่ทางขวา >> ถ้าค่าของ Feature ต่ำ จะเพิ่มค่าพยากรณ์แทน
#
#   SHAP Value คำนวณจาก Shapley Value (แนวคิดจากทฤษฎีเกม) โดยใช้หลักการว่า
#   "ถ้าตัด Feature ออกไป ผลลัพธ์โมเดลจะเปลี่ยนไปแค่ไหน?"
#   ใช้สูตรคำนวณ SHAP Value
#   1. รันโมเดลโดยใช้ทุก Feature → ได้ค่าพยากรณ์ f(F) 
#   2. เลือก Subset ของ Feature แบบสุ่ม เช่น S={PM2.5,CO}
#   3. รันโมเดลใหม่เฉพาะ Feature ใน S
#   4. เพิ่ม Feature ใหม่ i เข้าไปใน S แล้วรันโมเดลอีกครั้ง >> ได้ค่าพยากรณ์ f(S∪{i})
#   5. คำนวณผลกระทบของ Feature นั้น i โดยใช้ f(S∪{i})−f(S)
#   6. ทำซ้ำกับทุก Subset ที่เป็นไปได้ และถ่วงน้ำหนักตามสูตร
#   7. สรุปเป็นค่าเฉลี่ยของผลกระทบ >> ได้ค่า SHAP Value ของ Feature นั้น
# =================================================================================================