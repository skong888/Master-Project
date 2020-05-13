import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Lambda
from tensorflow.keras.layers import MaxPooling2D, Conv2D, AveragePooling2D, SeparableConv2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

print ("tensorflow test")
print ("test starting at ", datetime.datetime.now())

# File Names (need to change / for \\ on laptop)
model_File = os.path.join(os.getcwd(), 'Models/')
fTime = time.strftime("%b-%d-%H%M", time.localtime())
file_Name = os.path.join(model_File, fTime+'-Separable1/')
model_Name = 'separable.h5'

# Test Info
batch_size = 2
num_classes = 10
epochs = 100

# Setup Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print ("Shape of data :", x_train.shape)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Building network
def cnn():
	cnn_model = Sequential()
	# Layer 1
	cnn_model.add(SeparableConv2D(32, (3, 3), activation='relu', input_shape=x_train.shape[1:]))
	cnn_model.add(SeparableConv2D(32, (3, 3), activation='relu'))
	cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	cnn_model.add(Dropout(0.25))
	# Layer 2
	cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
	cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
	cnn_model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	cnn_model.add(Dropout(0.25))
	# Layer 3
	cnn_model.add(Flatten())
	cnn_model.add(Dense(512))
	cnn_model.add(Activation('relu'))
	cnn_model.add(Dropout(0.5))
	cnn_model.add(Dense(num_classes))
	cnn_model.add(Activation('softmax'))
	return cnn_model


model = cnn()
model.summary()

opt = SGD(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

start_Time = time.time()

# Training model
training = model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_split=0.2,
	shuffle=True,
	verbose=1)

training_Time = time.time()

# Creating Folder
if not os.path.exists(file_Name):
	os.mkdir(file_Name)
# Acc History
plt.plot(training.history['acc'])  # -------------- to change for 2.0
plt.plot(training.history['val_acc'])  # -------------- to change for 2.0
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(file_Name+'acc_test.png')
plt.close()
# Loss History
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(file_Name+'loss_test.png')
plt.close()
# Predict Test

scores = model.evaluate(x_test, y_test, verbose=2)
end_Time = time.time()

print('total training time:', (training_Time - start_Time)/60, "min")
print('Test time:', end_Time - training_Time)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# Saving Model
model_Path = os.path.join(file_Name, model_Name)
model.save(model_Path)
weight_Path = os.path.join(file_Name, 'weight_' + model_Name)
model.save_weights(weight_Path)
model_Size = os.path.getsize(model_Path)
print('Model size:', model_Size)
with open(file_Name + 'architecture.json', 'w')as f:
	f.write(model.to_json())

# Saving info in log
with open(file_Name + 'Log.txt', 'w+') as log:
	log.write('\n File created ' + fTime + '\n')
	model.summary(print_fn=lambda x: log.write(x + '\n'))
	log.write('\n')
	log.write('Batch size:' + str(batch_size) + '\n')
	log.write('Number of epochs:' + str(epochs) + '\n')
	log.write('total training time:' + str((training_Time - start_Time)/60) + ' minutes \n')
	log.write('Test time:' + str(end_Time - training_Time) + ' seconds \n')
	log.write('Test loss:' + str(scores[0]) + '\n')
	log.write('Test accuracy:' + str(scores[1]*100) + ' % \n')
	log.write('Model size:' + str(model_Size/1000000) + 'MB \n')
	log.close

print("test done at ", datetime.datetime.now())
