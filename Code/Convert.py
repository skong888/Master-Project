import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.datasets import cifar10
import os
import time
import numpy as np
import h5py
#import tensorflow.tflite_runtime.interpreter as tflite
tf.enable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
tf.executing_eagerly()

# Setup Data
num_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
mnist_train, _ = cifar10.load_data()
images = tf.cast(mnist_train[0], tf.float32)/255.0
#images = tf.cast(mnist_train[0], tf.uint8)

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]


model_File = os.path.join(os.getcwd(), 'Models/Jan-29-1238-Perforated/')
model_Path = os.path.join(model_File, 'Pruned.h5')
model = load_model(model_Path)
print("model successfully loaded")
model.summary()

#converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter = tf.lite.TFLiteConverter.from_keras_model_file(model_Path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
  for input_value in mnist_ds.take(100):
    yield [input_value]

converter.representative_dataset = representative_data_gen


converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()
open(model_File + "prunedlite.tflite", "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


x_train = x_train.astype('uint8')
x_test = x_test.astype('uint8')
#test = tf.cast(x_test, tf.uint8)
test = x_test.reshape(10000, 1, 32, 32, 3)
output_data = np.zeros([10000, 10])

print("start test")
print(input_details)
print(output_details)
start_time = time.time()
for i in range(10000):
    inf_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], test[i])
    interpreter.invoke()
    output_data[i] = interpreter.get_tensor(output_details[0]['index'])
    print("img #", i, "/10000 - ", "time: ", time.time() - inf_time, end="\r")
    #print(output_data)
print("\nend time: ", time.time()-start_time)

label = np.rint(output_data/255)
n = 0
for i in range(10000):
    #print(y_test[i], label[i])
    if (y_test[i] == label[i]).all():
        n += 1
print(n/10000)


