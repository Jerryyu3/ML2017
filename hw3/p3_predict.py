import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, adam
from keras.utils import np_utils
import sys
import csv

def load_test():
  X = []; Y = [];
  with open(sys.argv[1],'r') as file:
    for row in csv.DictReader(file):
      f = row['feature'];
      f = f.split();f = list(map(float,f));
      X.append(f);Y.append(int(row['id']));
  X = np.asarray(X); Y = np.asarray(Y); #Y = [Y]; Y = np.transpose(Y);
  #Y = np_utils.to_categorical(Y,7);
  X = X/255.0;    
  #Label = (np.delete(np.genfromtxt(sys.argv[1], delimiter=','),0,0))[:,0];
  #X = (np.delete(np.genfromtxt(sys.argv[1]),0,0))#[:,1:];
  return X,Y

def predict(x_test,idx):

  model = load_model(sys.argv[2]);

  x_test = x_test.reshape(x_test.shape[0],48,48,1);

  classes = model.predict(x_test,batch_size = 64);
  label = classes.argmax(axis=1); 
  ansy = np.concatenate(([idx],[label]),axis=0); ansy.astype(int); ansy = np.transpose(ansy);
  print (ansy)
  ansy = ansy.tolist();

  with open(sys.argv[3],'w')as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg);

def main():
  x_test,idx = load_test();
  predict(x_test,idx);

if __name__=='__main__':
  main()
