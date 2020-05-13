import data
import numpy as np
#import tensorflow as tf
import time
import tempfile
import zipfile
import os
import tflite_runtime.interpreter as tflite

(x_train, y_train), (x_test, y_test) = data.load_data()


print("Train data: ", x_train.shape)
print("Train labels: ", y_train.shape)
print("Test data: ", x_test.shape)
print("Test labels: ", y_test.shape)

model_File = "./Models/Jan-29-1238-Perforated/"
model_Path = os.path.join(model_File, 'lite.tflite')
pruned_model_Path = os.path.join(model_File, 'prunedlite.tflite')

interpreter = tflite.Interpreter(model_path=model_Path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

pruned_interpreter = tflite.Interpreter(model_path=pruned_model_Path)
pruned_interpreter.allocate_tensors()
pruned_input_details = pruned_interpreter.get_input_details()
pruned_output_details = pruned_interpreter.get_output_details()

test = x_test.reshape(10000, 1, 32, 32, 3)
output_data = np.zeros([10000, 10])
pruned_output_data = np.zeros([10000, 10])

print(y_test.shape)
print(output_data.shape)
print("start test")
start_time = time.time()
for i in range(10000):
    inf_time = time.time()
    interpreter.set_tensor(input_details[0]['index'], test[i])
    interpreter.invoke()
    output_data[i] = interpreter.get_tensor(output_details[0]['index'])
    inf_stop = time.time()-inf_time
    print("img #", i, "/10000 - ", "time: ", inf_stop, end="\r")
    #print(output_data)
stop_time = time.time()-start_time
print("\nend time: ", stop_time)


pruned_start_time = time.time()
for i in range(10000):
    pruned_inf_time = time.time()
    pruned_interpreter.set_tensor(pruned_input_details[0]['index'], test[i])
    pruned_interpreter.invoke()
    pruned_output_data[i] = pruned_interpreter.get_tensor(output_details[0]['index'])
    pruned_inf_stop = time.time()-pruned_inf_time
    print("pruned img #", i, "/10000 - ", "time: ", pruned_inf_stop, end="\r")
    #print(output_data)
pruned_stop_time = time.time()-pruned_start_time
print("\nend time: ", pruned_stop_time)

label = np.rint(output_data/255)
pruned_label = np.rint(pruned_output_data/255)
n = 0
m = 0

for i in range(10000):
    #print(y_test[i], label[i])
    if (y_test[i] == label[i]).all():
        n += 1
    if (y_test[i] == pruned_label[i]).all():
        m += 1

print("accuracy :", n/10000)
print("pruned accuracy :", m/10000)


size=(os.path.getsize(model_Path) / float(2**20))
print('Model size: %.2f Mb' % size)
_,zip = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(model_Path)
model_Size = (os.path.getsize(zip) / float(2**20))
print("Size of the model after compression: %.2f Mb"
      % model_Size)


pruned_size = (os.path.getsize(pruned_model_Path) / float(2**20))
print('Pruned Model size: %.2f Mb' % pruned_size)
_,zip = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(model_Path)
pruned_model_Size = (os.path.getsize(zip) / float(2**20))
print("Size of the pruned model after compression: %.2f Mb"
      % pruned_model_Size)



fTime = time.strftime("%d-%b-%H%M", time.localtime())
#file_Name = os.path.join('./Models/', fTime+'-tflite/')
#if not os.path.exists(file_Name):
#	os.mkdir(file_Name)model_Path = os.path.join(model_File, 'perforated.h5')
log = open(model_File + 'liteLog.txt', 'w+')
log.write('\n File created ' + fTime + '\n')
log.write('-------BASE-------\n')
log.write('accuracy :' + str(n/10000) + '\n')
log.write('time :' + str(stop_time)+'\n')
log.write('time for single image :' + str(inf_stop)+'\n')
log.write('Model size: %.2f Mb \n' % size)
log.write('Compressed Model size: %.2f Mb \n' % model_Size)


log.write('\n\n------PRUNED------\n')
log.write('accuracy :' + str(m/10000) + '\n')
log.write('time :' + str(pruned_stop_time)+'\n')
log.write('time for single image :' + str(pruned_inf_stop)+'\n')
log.write('Pruned Model size: %.2f Mb \n' % pruned_size)
log.write('Compressed Pruned Model size: %.2f Mb \n' % pruned_model_Size)


