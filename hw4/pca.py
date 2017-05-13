import os 
from PIL import Image
from os.path import splitext
import numpy as np
import matplotlib.pyplot as plt


def main():
    train_faces = [];
    test_faces = [];
    data_dir = "./CMUface";
    i = 0;
    for filename in os.listdir(data_dir):
        print ("Loading: %s" % filename);
        loadfile = Image.open(os.path.join(data_dir,filename),'r');
        loadfile.load();
        data = np.asarray(loadfile,dtype="float32"); data = np.reshape(data,(1,64*64));
        if(i%75<10 and i/75<10):
          train_faces.append(data);
        else:
          test_faces.append(data);
        i+=1;
    train_faces = np.asarray(train_faces); train_faces = np.reshape(train_faces,(100,64*64));
    test_faces = np.asarray(test_faces); test_faces = np.reshape(test_faces,(75*13-100,64*64));
    ins = 100; dim = 64*64;
    meanImage = np.sum(train_faces,axis=0)/ins; #Calculate Mean faces
    temp_faces = train_faces - meanImage;
    Gram = np.dot(np.transpose(temp_faces),temp_faces);
    eigv, eigfaces = np.linalg.eigh(Gram);
    #plot_p1_1(eigfaces[:,64*64-9:],meanImage); # for q1_1
    #plot_p1_2(train_faces,eigfaces[:,64*64-5:],meanImage);
    p1_3(train_faces,eigfaces[:,64*64-100:],meanImage)

def p1_3(train_faces,eigfaces,meanImage):
    k = 0;
    while(k<=100):
        coeff = np.dot(np.transpose(eigfaces[:,99-k:]),np.transpose(train_faces-meanImage));
        proj_faces = np.dot(eigfaces[:,99-k:],coeff) + np.transpose([meanImage]);
        proj_faces = np.transpose(proj_faces);
        error = np.sqrt(np.mean(np.power(proj_faces-train_faces,2)))/256;
        k+=1;
        if(k%10==0): print ("The error of reconstruction of top "+str(k)+" eigenfaces "+str(error));
        if(error<0.01):
           print ("The error of reconstruction of top "+str(k)+" eigenfaces "+str(error));
           print("The smallest k such that the error is less than 1% is "+str(k));break;

def plot_p1_2(train_faces,eigfaces,meanImage):
    coeff = np.dot(np.transpose(eigfaces),np.transpose(train_faces-meanImage));
    proj_faces = np.dot(eigfaces,coeff) + np.transpose([meanImage]);

    fig = plt.figure(figsize=(14, 8))
    for i in range(100):
        ax = fig.add_subplot(10, 10, i+1);
        ax.imshow(np.reshape(train_faces[i,:],(64,64)),cmap='gray');
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout();
    fig.suptitle("Original faces");
    img_path = "./pca_p1_2_ori.png"; fig.savefig(img_path);
    fig = plt.figure(figsize=(14, 8))
    for i in range(100):
        ax = fig.add_subplot(10, 10, i+1);
        ax.imshow(np.reshape(proj_faces[:,i],(64,64)),cmap='gray');
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout();
    fig.suptitle("Projected faces");
    img_path = "./pca_p1_2_proj.png"; fig.savefig(img_path);
   

def plot_p1_1(eigfaces,meanImage):
    fig = plt.figure(figsize=(14, 8));
    i = 8;
    while(i>=0):#for i in range(9):
        ax = fig.add_subplot(3, 3, 9-i);
        ax.imshow(np.reshape(eigfaces[:,i],(64,64)),cmap='gray');
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout();
        i-=1;
    fig.suptitle("Top 9 eigenfaces");
    img_path = "./pca_p1_1.png"; fig.savefig(img_path);

    fig = plt.figure(figsize=(14, 8))
    plt.imshow(np.reshape(meanImage,(64,64)),cmap='gray');
    plt.tight_layout();
    plt.draw(); fig.suptitle("MeanImage");
    img_path = "./pca_p1_1_mean.png"; fig.savefig(img_path);

if __name__ == "__main__":
    main();
