import os
import sys
import csv
import argparse
from keras.models import load_model
from termcolor import colored,cprint
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

base_dir = (os.path.dirname(os.path.realpath(__file__)))

img_dir = os.path.join(base_dir, 'image')
if not os.path.exists(img_dir):
  os.makedirs(img_dir)

cmap_dir = os.path.join(img_dir, 'cmap')
if not os.path.exists(cmap_dir):
    os.makedirs(cmap_dir)

partial_see_dir = os.path.join(img_dir,'partial_see')
if not os.path.exists(partial_see_dir):
    os.makedirs(partial_see_dir)

model_dir = os.path.join(base_dir, 'exp')

def load_data(filepath):
    X = []; Y = [];
    with open(filepath,'r')as file:
      for row in csv.DictReader(file):
          f = row['feature']; f = f.split();
          f = list(map(float,f));
          X.append(f); Y.append(float(row['label']));
    X = np.asarray(X); Y = np.asarray(Y);
    #Y = np_utils.to_categorical(Y,7);
    X = X/255.0; 
    X = [ X[i,:].reshape((1,48,48,1)) for i in range(X.shape[0])];
    print (np.shape(X))
    return X,Y;

def main():
    parser = argparse.ArgumentParser(prog='plot_saliency.py',
            description='ML-Assignment3 visualize attention heat map.')
    parser.add_argument('--epoch', type=int, metavar='<#epoch>', default=80)
    args = parser.parse_args()
    model_name = "epoch%s_0/model.647-0.6646.h5" %str(args.epoch)
    model_path = os.path.join(model_dir, model_name)
    emotion_classifier = load_model(model_path)
    print(colored("Loaded model from {}".format(model_name), 'yellow', attrs=['bold']))

    #private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    #private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
    #                   for i in range(len(private_pixels)) ]

    pixels,_ = load_data("./train.csv");
    input_img = emotion_classifier.input
    #img_ids = ["image ids from which you want to make heatmaps"]
    img_ids = [28602,28606,28648,28650];

    for idx in img_ids:
        val_proba = emotion_classifier.predict(pixels[idx])
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        grads /= (K.sqrt(K.mean(K.square(grads)))+1e-5);
        fn = K.function([input_img, K.learning_phase()], [grads])

        
        heatmap = np.zeros((1,1,48,48,1));
        '''
        Implement your heatmap processing here!
        hint: Do some normalization or smoothening on grads
        '''
        for i in range(20):
            grads = fn([pixels[idx], 1]);
            heatmap += grads;

        plt.figure()
        plt.imshow(pixels[idx].reshape(48,48),cmap='gray')
        plt.colorbar();
        plt.tight_layout();
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}_original.png'.format(idx)),dpi=100);

        heatmap = heatmap.reshape(48,48);
        thres = 0.7
        see = pixels[idx].reshape(48, 48)
        see[np.where(heatmap <= thres)] = np.mean(see)


        plt.figure()
        plt.imshow(heatmap, cmap=plt.cm.jet)
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(cmap_dir, '{}_heat.png'.format(idx)), dpi=100)

        plt.figure()
        plt.imshow(see,cmap='gray')
        plt.colorbar()
        plt.tight_layout()
        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join(partial_see_dir, '{}_part.png'.format(idx)), dpi=100)

if __name__ == "__main__":
    main()
