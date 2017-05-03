import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import numpy as np
import csv

def load_data(filepath):
    X = []; Y = [];
    with open(filepath,'r')as file:
      for row in csv.DictReader(file):
        f = row['feature']; f = f.split();
        f = list(map(float,f));
        X.append(f); Y.append(float(row['label']));
    X = np.asarray(X); Y = np.asarray(Y);
    X = X/255.0;
    X = [ X[i,:].reshape((1,48,48,1)) for i in range(X.shape[0])];
    return X,Y

vis_dir = os.path.dirname(os.path.realpath(__file__));

def main():
    #emotion_classifier = load_model('./exp/epoch1000_0/model.647-0.6646.h5')
    emotion_classifier = load_model('./exp/epoch300_1/model.h5')
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])

    input_img = emotion_classifier.input
    name_ls = ["conv2d_3"];
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

    #private_pixels = load_pickle('../fer2013/privateTest_pixels.pkl')
    #private_pixels = [ np.fromstring(private_pixels[i], dtype=float, sep=' ').reshape((1, 48, 48, 1)) 
    #                   for i in range(len(private_pixels)) ]

    pixels,Y = load_data('./train.csv');
    store_path = 'epoch300_1_conv2d3_shock'
    choose_id = 28600
    photo = pixels[choose_id]
    for cnt, fn in enumerate(collect_layers):
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        nb_filter = im[0].shape[3]
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.tight_layout()
        fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join(vis_dir, store_path)
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'layer{}'.format(cnt)))
if __name__=="__main__":
    main()
