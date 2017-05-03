import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def main():
  base_dir = (os.path.dirname(os.path.realpath(__file__)))
  hist_dir = os.path.join(base_dir,'exp',sys.argv[1]);
  train_loss_path = os.path.join(hist_dir,'train_loss');
  train_acc_path = os.path.join(hist_dir,'train_accuracy');
  valid_loss_path = os.path.join(hist_dir,'valid_loss');
  valid_acc_path = os.path.join(hist_dir,'valid_accuracy');
  loss_fig = os.path.join(hist_dir,'loss.png');
  acc_fig = os.path.join(hist_dir,'acc.png');

  t = np.arange(0,400,1);
  train_loss = np.genfromtxt(train_loss_path);
  valid_loss = np.genfromtxt(valid_loss_path);
  train_acc = np.genfromtxt(train_acc_path);
  valid_acc = np.genfromtxt(valid_acc_path);

  plt.figure()
  plt.plot(t,train_loss,label="train"); plt.plot(t,valid_loss,label="valid");

  plt.xlabel('# of epochs');
  plt.ylabel('Loss');
  plt.title('Loss of training set and validation set');
  plt.grid(True);
  plt.legend();
  plt.savefig(loss_fig)

  plt.figure()
  plt.plot(t,train_acc,label="train"); plt.plot(t,valid_acc,label="valid");

  plt.xlabel('# of epochs');
  plt.ylabel('Accuracy');
  plt.title('Accuracy of training set and validation set');
  plt.grid(True);
  plt.legend();
  plt.savefig(acc_fig)



if __name__=='__main__':
  main(); 
