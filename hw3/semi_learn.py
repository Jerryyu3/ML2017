import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adadelta, adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator
import sys
import csv
#import datetime

class History(Callback):
  def on_train_begin(self,logs={}):
    self.tr_losses=[];
    self.val_losses=[];
    self.tr_accs=[];
    self.val_accs=[];
  def on_epoch_end(self,epoch,logs={}):
    self.tr_losses.append(logs.get('loss'));
    self.val_losses.append(logs.get('val_loss'));
    self.tr_accs.append(logs.get('acc'));
    self.val_accs.append(logs.get('val_acc'));

def load_data():
  X = []; Y = [];
  with open(sys.argv[1],'r') as file:
    for row in csv.DictReader(file):
      f = row['feature'];
      f = f.split();f = list(map(float,f));
      X.append(f);Y.append(float(row['label']));
  X = np.asarray(X); Y = np.asarray(Y); #Y = [Y]; Y = np.transpose(Y);
  Y = np_utils.to_categorical(Y,7);
  X = X/255.0;
  #Label = (np.delete(np.genfromtxt(sys.argv[1], delimiter=','),0,0))[:,0];
  #X = (np.delete(np.genfromtxt(sys.argv[1]),0,0))#[:,1:];
  return X,Y

def dump_history(store_path,logs):
  with open(os.path.join(store_path,'train_loss'),'a')as f:
    for loss in logs.tr_losses: f.write('{}\n'.format(loss));
  with open(os.path.join(store_path,'train_accuracy'),'a')as f:
    for acc in logs.tr_accs: f.write('{}\n'.format(acc));
  with open(os.path.join(store_path,'valid_loss'),'a')as f:
    for loss in logs.val_losses: f.write('{}\n'.format(loss));
  with open(os.path.join(store_path,'valid_accuracy'),'a')as f:
    for acc in logs.val_accs: f.write('{}\n'.format(acc));
  

def train(x_train,y_train):

  base_dir = (os.path.dirname(os.path.realpath(__file__)));
  exp_dir = os.path.join(base_dir,'exp');
  dir_cnt = 0; epoch = 300;
  log_path = "epoch{}".format(str(epoch)); log_path += "_";
  store_path = os.path.join(exp_dir,log_path+str(dir_cnt));

  while(dir_cnt<30):
    if not os.path.isdir(store_path):
      os.mkdir(store_path);
      break;
    else:
      dir_cnt += 1;
      store_path = os.path.join(exp_dir,log_path+str(dir_cnt));

  model = Sequential();

  model.add(Conv2D(64,(5,5),border_mode='valid',input_shape=(48,48,1)));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  model.add(keras.layers.convolutional.ZeroPadding2D(padding=2)); 
  model.add(MaxPooling2D(pool_size=(5,5),strides=(2,2)));

  model.add(keras.layers.convolutional.ZeroPadding2D(padding=1));  
  model.add(Conv2D(64,(3,3)));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  #model.add(MaxPooling2D(2,2));
  model.add(keras.layers.convolutional.ZeroPadding2D(padding=1));
  model.add(Conv2D(64,(3,3)));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3,3),strides=(2,2)));

  model.add(keras.layers.convolutional.ZeroPadding2D(padding=1));  
  model.add(Conv2D(128,(3,3)));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  model.add(keras.layers.convolutional.ZeroPadding2D(padding=1));
  model.add(Conv2D(128,(3,3)));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));

  model.add(keras.layers.convolutional.ZeroPadding2D(padding=1));  
  model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3,3),strides=(2,2)));

  model.add(Flatten());
  model.add(Dense(units=512));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  model.add(Dropout(0.3));
  model.add(Dense(units=512));
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zero', weights=None));
  model.add(Dropout(0.3));
  #model.add(Dense(units=200,activation="relu"));
  #model.add(Dropout(0.5));
  model.add(Dense(units=7,activation="softmax"));
  model.summary();
  
  train_bound = int(x_train.shape[0]*9/10);
  label_bound = int(x_train.shape[0]/2);

  x_train1_ori = x_train[0:label_bound,:]; y_train1_ori = y_train[0:label_bound,:];
  x_train1 = x_train[0:label_bound,:]; y_train1 = y_train[0:label_bound,:];
  x_train2 = x_train[label_bound:train_bound,:];
  x_val = x_train[train_bound:,:]; y_val = y_train[train_bound:,:];

  x_train1_ori = x_train1_ori.reshape(x_train1_ori.shape[0],48,48,1);
  x_train1 = x_train1.reshape(x_train1.shape[0],48,48,1);
  x_train2 = x_train2.reshape(x_train2.shape[0],48,48,1);
  x_val = x_val.reshape(x_val.shape[0],48,48,1);

  #ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08);
  #model.compile(loss="categorical_crossentropy",optimizer=ada,metrics=['accuracy']); 
  model.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy']);
  acc = 0;

  #early = EarlyStopping(monitor='val_loss', patience=20);
  history = History();
  #model.fit(x_train1,y_train1,batch_size=64,epochs=epoch,validation_data=(x_val,y_val),callbacks=[early]);
  #dump_history(store_path,history);
  #model.save(os.path.join(store_path,'model.h5'));

  #model.fit(x_train1,y_train1,validation_split=0.1,batch_size=50,epochs=20,callbacks=[early_stopping]);

  filepath = os.path.join(store_path,'model.{epoch:03d}-{val_acc:.4f}.h5');
  #checkpointer = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='auto')

  datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

  datagen.fit(x_train1);

  iteration = 0;
  while(iteration<epoch):
    model.fit_generator(datagen.flow(x_train1,y_train1,batch_size=64),steps_per_epoch=int(x_train1.shape[0]/64),nb_epoch=1,validation_data=(x_val,y_val),callbacks=[history])#,callbacks=[checkpointer])
    dump_history(store_path,history);
    prob = model.predict(x_train2,batch_size=64);
    label = prob.argmax(axis=1); label.astype(int);
    #print(np.sum(prob[:,label][0]>0.45));
    #print(np.shape(x_train2[prob[:,label][0]>0.45]));
    new_train = (x_train2[prob[:,label][0]>0.45]); new_label = [label[prob[:,label][0]>0.45]];
    new_label = np.transpose(new_label);
    new_label = np_utils.to_categorical(new_label,7);
    #print(np.shape(y_train1_ori),np.shape(new_label));

    if (np.shape(new_train)[0]>0):
      x_train1 = np.concatenate((x_train1_ori,new_train),axis=0); 
      y_train1 = np.concatenate((y_train1_ori,new_label),axis=0);
    #print(np.shape(x_train1),np.shape(y_train1));
    iteration += 1;
    model.save(os.path.join(store_path,'model.h5'));

  #score = model.evaluate(x_val,y_val);
  #print ('\nVal acc:',score[1]);

def main():
  x_train,y_train = load_data();
  model = train(x_train,y_train);

if __name__=='__main__':
  main()
