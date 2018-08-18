from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json



x = np.load('x_input.npy')
y = np.load('y.npy')
train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)


train_y = np_utils.to_categorical(train_y)
test_y = np_utils.to_categorical(test_y)
print(train_y)
model = Sequential()
model.add(Dense(128, input_shape=(62,)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics=["accuracy"])
history = model.fit(train_X, train_y, validation_split=0.2, epochs=19, batch_size=82, verbose=0)
loss,accuracy = model.evaluate(test_X, test_y, verbose=0)
print("Accuracy = {:.2f}".format(accuracy))
print(history.history.keys())
# summarize history for accuracy
plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')


# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model! ^_^")



