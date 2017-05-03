import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K

import numpy as np

NUM_STEPS = 80;
RECORD_FREQ = 20;
def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func, filter_images):
    """
    Implement this function!
    """
    for i in range(RECORD_FREQ):
      target, grads = iter_func([input_image_data]);
      input_image_data += grads;
    filter_images[num_step].append([input_image_data[0].reshape((48,48)),target]);
    return filter_images,input_image_data

filter_dir = os.path.dirname(os.path.realpath(__file__));

def main():
    emotion_classifier = load_model("./exp/epoch1000_0/model.647-0.6646.h5")
    #emotion_classifier = load_model("./exp/epoch300_1/model.h5")
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
    input_img = emotion_classifier.input
    store_path = 'epoch1000_0';

    nb_filter = 64;
    name_ls = ["conv2d_2","conv2d_3"]
    collect_layers = [ layer_dict[name].output for name in name_ls ]

    for cnt, c in enumerate(collect_layers):
        filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
        for filter_idx in range(nb_filter):
            input_img_data = np.random.random((1, 48, 48, 1)) # random noise
            target = K.mean(c[:, :, :, filter_idx])
            grads = normalize(K.gradients(target, input_img)[0])
            iterate = K.function([input_img], [target, grads])

            ###
            #"You need to implement it."
            for it in range(NUM_STEPS//RECORD_FREQ):
                filter_imgs,input_img_data = grad_ascent(it, input_img_data, iterate, filter_imgs)
            ###
        
        for it in range(NUM_STEPS//RECORD_FREQ):
            fig = plt.figure(figsize=(14, 8))
            for i in range(nb_filter):
                ax = fig.add_subplot(nb_filter/16, 16, i+1)
                ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
                plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
                plt.tight_layout()
            fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], (it+1)*RECORD_FREQ))
            img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
            fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))
        
if __name__ == "__main__":
    main()
