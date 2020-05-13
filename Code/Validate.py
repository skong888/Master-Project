import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model

from tensorflow.keras.datasets import cifar10


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model_File = os.path.join(os.getcwd(), 'Models/Separable/')
model_name = ('Pruned')
model_Path = os.path.join(model_File, model_name+'.h5')
model = load_model(model_Path)
print("model successfully loaded")
model.summary()

num_classes = 10

fTime = time.strftime("%b-%d-%H%M", time.localtime())

# Setup Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Shape of data :", x_train.shape)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

N = 100

with open(model_File + model_name +'_vLog.txt', 'a+') as log:
    #log.write('Time\tAcc \tLoss\n')
    log.write('Time \n')
    for i in range(N):
        # Old Score
        print('-'*30)
        print(i+1, '/', N)
        start_time = time.time()
        score = model.evaluate(x_test, y_test, verbose=2)
        test_time = time.time() - start_time
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        print('Test time:', test_time)
        #log.write('{0:.4f}\t{1:.4f}\t{2:.4f} \n'.format(test_time, score[1], score[0]))
        log.write('{0:.4f}\n'.format(test_time))
            #log.write('\nValidation ' + model_name + 'date:' + fTime + '\n')
            #log.write('Test time:' + str(test_time) + ' seconds ')
            #log.write('Test loss:' + str(score[0]) + '\n')
            #log.write('Test accuracy:' + str(score[1] * 100) + ' % \n')
            #log.write('__________________________________________________\n')

    log.close


