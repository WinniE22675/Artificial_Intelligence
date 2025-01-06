import keras.api.models as mod
import keras.api.layers as lay

model = mod.Sequential()
model.add(lay.SimpleRNN(units = 1,
                        input_shape = (1,1),
                        activation='relu'))

model.summary()
model.save("RNN.h5")