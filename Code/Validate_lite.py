import data
import numpy as np
import tensorflow as tf
import time
import tempfile
import zipfile
import os
#import tflite_runtime.interpreter as tflite

(x_train, y_train), (x_test, y_test) = data.load_data()


print("Train data: ", x_train.shape)
print("Train labels: ", y_train.shape)
print("Test data: ", x_test.shape)
print("Test labels: ", y_test.shape)

model_File = "./coralmodels/perforated/"
model_Name = "prunedlite"
model_Path = os.path.join(model_File, model_Name+'.tflite')


interpreter = tf.lite.Interpreter(model_path=model_Path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test = x_test.reshape(10000, 1, 32, 32, 3)
output_data = np.zeros([10000, 10])

print(y_test.shape)
print(output_data.shape)
print("start test")

N = 100
with open(model_File + model_Name + 'vLog.txt', 'a+') as log:
    log.write('Time\tPer\n')
    for k in range(N):
        print(k+1, '/', N)
        start_time = time.time()
        inf = np.zeros(10000)
        for i in range(10000):
            inf_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test[i])
            interpreter.invoke()
            output_data[i] = interpreter.get_tensor(output_details[0]['index'])
            inf_stop = time.time()-inf_time
            inf[i] = inf_stop
            print("img #", i, "/10000 - ", "time: ", inf_stop, end="\r")
        stop_time = time.time()-start_time
        print("\nend time: ", stop_time)
        label = np.rint(output_data / 255)

        n = 0
        for i in range(10000):
            # print(y_test[i], label[i])
            if (y_test[i] == label[i]).all():
                n += 1

        print("accuracy :", n / 10000)
        log.write('{0:.4f}\t{1:.5f} \n'.format(stop_time, np.mean(inf)))

    log.close()
