import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np 
import matplotlib.pyplot as plt

import keras.api.models as mod
import keras.api.layers as lay

pitch = 40
step = 1
N = 500
n_train = int(N*0.7)

def gen_data(x):
    return (x%pitch)/pitch

t = np.arange(1, N+1)
# y = [gen_data(i) for i in t]
y = np.sin(0.05*t*10) + 0.8 * np.random.rand(N)
y = np.array(y)

# plt.figure()
# plt.plot(y)
# plt.show()

def convertToMatrix(data, step=2):
    X, Y =[],[]
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d,])
        Y.append(data[d,])
        
    return np.array(X), np.array(Y)

train, test =  y[0:n_train],y[n_train:N]

x_train, y_train = convertToMatrix(train,step)
x_test, y_test = convertToMatrix(test,step)

print("Dimension (Before) : ", train.shape, test.shape)
print("Dimension (After)  : ", x_train.shape, x_test.shape)

model = mod.Sequential()
model.add(lay.SimpleRNN(units = 32,
                        input_shape=(step,1),
                        activation="relu"))
model.add(lay.Dense(units = 1))

model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1)

# plt.plot(hist.history['loss'])
# plt.show()

y_pred = model.predict(x_test)

plt.figure(figsize=(10, 6))

plt.plot(y_test, label="Original", color="blue")
plt.plot(y_pred, label="Predict", linestyle="--", color="red")
plt.title("Original and Predicted Data")
plt.xlabel("Time")
plt.ylabel("Values")
plt.legend()
plt.grid(True)
plt.show()
