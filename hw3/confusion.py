from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
from keras.utils import np_utils
import itertools

base_dir = (os.path.dirname(os.path.realpath(__file__)))
exp_dir = os.path.join(base_dir,'exp')
store_path = "epoch1000_0";

def load_data(filepath):
    X = []; Y = [];
    with open(filepath,'r')as file:
      for row in csv.DictReader(file):
          f = row['feature']; f = f.split();
          f = list(map(float,f));
          X.append(f); Y.append(float(row['label']));
    X = np.asarray(X); Y = np.asarray(Y); Y = [Y]
    Y = np.transpose(Y);
    #Y = np_utils.to_categorical(Y,7);
    X = X/255.0;
    #X = [ X[i,:].reshape((1,48,48,1)) for i in range(X.shape[0])];

    return X,Y;

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    fig = plt.figure()
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    fig.savefig('./exp/confusion');

def main():
    model_path = os.path.join(exp_dir,store_path,'model.647-0.6646.h5')
    emotion_classifier = load_model(model_path)
    np.set_printoptions(precision=2)
    dev_feats,te_labels = load_data('train.csv');
    train_bound = int(dev_feats.shape[0]*9/10);
    dev_feats = dev_feats[train_bound:,:]; te_labels = te_labels[train_bound:,:];
    dev_feats = dev_feats.reshape(dev_feats.shape[0],48,48,1);

    print (np.shape(dev_feats));
    predictions = emotion_classifier.predict_classes(dev_feats,batch_size=64)
    conf_mat = confusion_matrix(te_labels,predictions)

    plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])


if __name__ == "__main__":
    main();
