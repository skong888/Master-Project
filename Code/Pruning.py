import os
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Lambda
from tensorflow.keras.layers import MaxPooling2D, Conv2D, AveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import zipfile
import tensorflow_model_optimization.sparsity.keras as tfmot

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

model_File = os.path.join(os.getcwd(), 'Models/Feb-10-1846-intbaseline/')
model_Path = os.path.join(model_File, 'baseline.h5')
model = load_model(model_Path)
print("model successfully loaded")
model.summary()

epochs = 20
batch_size = 2
num_classes = 10

# Setup Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print("Shape of data :", x_train.shape)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Old Score
old_score = model.evaluate(x_test, y_test, verbose=2)
print('-'*30)
print('Test loss:', old_score[0])
print('Test accuracy:', old_score[1])
print('-'*30)

num_train_samples = x_train.shape[0]
end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
print('End step: ' + str(end_step))

new_pruning_params = {
      'pruning_schedule': tfmot.PolynomialDecay(initial_sparsity=0.50,
                                                   final_sparsity=0.90,
                                                   begin_step=0,
                                                   end_step=end_step,
                                                   frequency=100)
}

new_pruned_model = tfmot.prune_low_magnitude(model, **new_pruning_params)
new_pruned_model.summary()
opt = SGD(lr=0.001)
new_pruned_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

callbacks = [
    tfmot.UpdatePruningStep()
]
start_Time = time.time()
# Training model
training = new_pruned_model.fit(x_train, y_train,
	batch_size=batch_size,
	epochs=epochs,
	validation_split=0.2,
	shuffle=True,
    callbacks=callbacks,
	verbose=1)
training_Time = time.time()

scores = new_pruned_model.evaluate(x_test, y_test, verbose=2)
end_Time = time.time()

print('total training time:', (training_Time - start_Time)/60, "min")
print('Test time:', end_Time - training_Time)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

final_model = tfmot.strip_pruning(new_pruned_model)
final_model.summary()
final_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
final_scores = final_model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', final_scores[0])
print('Test accuracy:', final_scores[1])


# File Names (need to change / for \\ on laptop)
#model_File = os.path.join(os.getcwd(), 'Models/')
fTime = time.strftime("%d-%b-%H%M", time.localtime())
#file_Name = os.path.join(model_File, fTime+'-GES843-Pruning-Perforated/')
file_Name = model_File
model_Name = 'Pruned.h5'

if not os.path.exists(file_Name):
	os.mkdir(file_Name)
# Acc History
plt.plot(training.history['acc'])  # -------------- to change for 2.0
plt.plot(training.history['val_acc'])  # -------------- to change for 2.0
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(file_Name+'pruned_acc_test.png')
plt.close()
# Loss History
plt.plot(training.history['loss'])
plt.plot(training.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(file_Name+'pruned_loss_test.png')
plt.close()

# Saving Model
new_model_Path = os.path.join(file_Name, model_Name)
final_model.save(new_model_Path)
new_weight_Path = os.path.join(file_Name, 'weight_' + model_Name)
final_model.save_weights(new_weight_Path)
new_model_Size = os.path.getsize(new_model_Path)
model_Size = os.path.getsize(model_Path)
print('Model size: %.2f Mb' % (os.path.getsize(model_Path) / float(2**20)))
print('Pruned Model size: %.2f Mb' % (os.path.getsize(new_model_Path) / float(2**20)))
with open(file_Name + 'pruned_architecture.json', 'w')as f:
	f.write(final_model.to_json())

_,zip3 = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip3, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(new_model_Path)
print("Size of the pruned model after compression: %.2f Mb"
      % (os.path.getsize(zip3) / float(2**20)))

_,zip = tempfile.mkstemp('.zip')
with zipfile.ZipFile(zip, 'w', compression=zipfile.ZIP_DEFLATED) as f:
  f.write(model_Path)
print("Size of the model after compression: %.2f Mb"
      % (os.path.getsize(zip) / float(2**20)))

# Saving info in log
with open(file_Name + 'Log.txt', 'a+') as log:
    log.write('\n ---------------------------------------------- \n')
    log.write('\n Pruned File created ' + fTime + '\n')
    #final_model.summary(print_fn=lambda x: log.write(x + '\n'))
    log.write('\n')
    log.write('Batch size:' + str(batch_size) + '\n')
    log.write('Number of epochs:' + str(epochs) + '\n')
    log.write('total training time:' + str((training_Time - start_Time)/60) + ' minutes \n')
    log.write('Test time:' + str(end_Time - training_Time) + ' seconds \n')
    log.write('Test loss:' + str(scores[0]) + '\n')
    log.write('Test accuracy:' + str(scores[1]*100) + ' % \n')
    log.write('Model size: %.2f Mb \n' % (os.path.getsize(model_Path) / float(2**20)))
    log.write('Pruned Model size: %.2f Mb \n' % (os.path.getsize(new_model_Path) / float(2**20)))
    log.write('Compressed Model size: %.2f Mb \n' % (os.path.getsize(zip) / float(2**20)))
    log.write('Compressed Pruned Model size: %.2f Mb \n' % (os.path.getsize(zip3) / float(2**20)))
    log.close

print("test done at ", datetime.datetime.now())

