import numpy as np
from sklearn.svm import LinearSVR as SVR
from gen import get_eigenvalues
import sys
import os
from PIL import Image

# Train a linear SVR

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)
svr = SVR(C=1)
svr.fit(X, y)


# svr.get_params() to save the parameters
# svr.set_params() to restore the parameters

# predict
#testdata = np.load(sys.argv[1]);
test_X = []
data_dir = "./hand";
for filename in os.listdir(data_dir):
  loadfile = Image.open(os.path.join(data_dir,filename),'r');loadfile.load();
  loadfile = loadfile.resize((int(loadfile.size[0]/8),int(loadfile.size[1]/8)));
  data = np.asarray(loadfile,dtype="float32"); data = np.reshape(data,(1,64*60));
  test_X.append(data);

test_X = np.asarray(test_X); test_X = np.reshape(test_X,(481,64*60));

vs = get_eigenvalues(test_X,1)

#test_X = np.array(test_X)

pred_y = svr.predict(vs);pred_y = pred_y.astype(int);
print (pred_y);
