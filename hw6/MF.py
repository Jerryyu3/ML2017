import numpy as np
import csv
import sys
import keras
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.layers import Flatten, Merge, Input, Concatenate
from keras.layers.merge import Dot, Add
from keras.models import Sequential, load_model, Model
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

def load_data(path):
    movie_dict = dict();
    movie_content = []
    #users = []
    user_dict = dict()
    Rating = []
    movie_id = []
    user_id = []

    with open(path + 'movies.csv', 'r') as movieFile:
        lines = movieFile.readlines();
        i = 0;
        for line in lines:
            if(i==0): 
                i+=1; continue;
            content = line.split('::');
            content[2] = content[2].split('\n')[0]
            movie_dict[int(content[0])] = content[2].replace('|', ' ')
            movie_content.append(content[2].replace('|', ' '))
    with open(path + 'users.csv', 'r') as userFile:
        lines = userFile.readlines();
        i = 0;
        for line in lines:
            if(i==0):
                i+=1; continue;
            con = line.split('::')
            con[4] = con[4].split('\n')[0]
            con[4] = con[4].split('-')[0]
            if con[1] == 'F': con[1] = 1
            else: con[1] = 0
            user_dict[int(con[0])] = [float(con[1]), float(con[2]), float(con[3]), float(con[4])]
    with open(path + 'train.csv', 'r') as trainFile:
        lines = trainFile.readlines();
        i = 0;
        for line in lines:
            if(i==0):
                i+=1; continue;
            TrainDataID, UserID, MovieID, R = line.split(',')
            #print(XD)
            #TrainDataID = XD[0]; UserID = XD[1]; MovieID = XD[2]; R = XD[3];
            R = int(R.split('\n')[0])
            movie_id.append(int(MovieID))
            user_id.append(int(UserID))
            Rating.append(R)

    #users = np.array(users)
    Rating = np.array(Rating)
    user_id = np.array(user_id)
    movie_id = np.array(movie_id)

    #print (Rating, user_id, movie_id);

    return user_dict, movie_dict, movie_content, Rating, movie_id, user_id

def split_data(X,Y,Z,split_ratio):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X_data = X[indices]
    Y_data = Y[indices]
    Z_data = Z[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]
    Z_train = Z_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]
    Z_val = Z_data[:num_validation_sample]

    return (X_train,Y_train,Z_train),(X_val,Y_val,Z_val)

def load_test(path):
    movie_id = []
    user_id = []
    with open(path + 'test.csv', 'r') as testFile:
        lines = testFile.readlines();
        i = 0;
        for line in lines:
            if(i==0):
                i+=1; continue;
            TestDataID, UserID, MovieID = line.split(',')
            MovieID = int(MovieID.split('\n')[0])
            movie_id.append(int(MovieID))
            user_id.append(int(UserID))
    test_user_id = np.array(user_id)
    test_movie_id = np.array(movie_id)

    return test_user_id, test_movie_id;


def saveResult(path, result):
    MeanRating = 3.581712; StdRating = 1.116898
    with open(path, 'w')as output:
        output.write("TestDataID,Rating\n")
        for idx, data in enumerate(result):
            ans = data[0]
            #ans = (data[0]*StdRating+MeanRating)
            output.write(str(idx+1) + ',' + str(ans) + '\n')


def get_model(n_users, n_items, latent_dim=10):
    user_input = Input(shape=[1]);
    item_input = Input(shape=[1]);
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input);
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input);
    item_vec = Flatten()(item_vec)
    user_bias = Embedding(n_users, 1, embeddings_initializer='zeros')(user_input);
    user_bias = Flatten()(user_bias)
    item_bias = Embedding(n_items, 1, embeddings_initializer='zeros')(item_input);
    item_bias = Flatten()(item_bias)
    r_hat = Dot(axes=1)([user_vec, item_vec]);
    r_hat = Add()([r_hat, user_bias, item_bias]);
    model = keras.models.Model([user_input, item_input], r_hat);
    model.summary(); 
    model.compile(loss='mse', optimizer='adamax');
    return model

def nn_model(n_users, n_items, latent_dim=10):
    user_input = Input(shape=[1]);
    item_input = Input(shape=[1]);
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer='random_normal')(user_input);
    user_vec = Flatten()(user_vec)
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer='random_normal')(item_input);
    item_vec = Flatten()(item_vec)
    merge_vec = Concatenate()([user_vec, item_vec])
    print (merge_vec.shape)
    hidden = Dense(150, activation='relu')(merge_vec);
    hidden = Dense(50, activation='relu')(hidden);
    output = Dense(1)(hidden)
    model = keras.models.Model([user_input, item_input], output);
    model.summary();
    model.compile(loss='mse', optimizer='adamax');
    return model

def main():

    Training = int(sys.argv[3]);
    if Training == 1:
        user_dict, movie_dict, movie_content, Rating, movie_id, user_id = load_data(sys.argv[1]);
        #MeanRating = np.mean(Rating); StdRating = np.std(Rating);
        #Rating = (Rating-MeanRating)/StdRating;
        model = nn_model(max(user_dict.keys()),max(movie_dict.keys()),13);
        (user_train,movie_train,Rating_train),(user_val,movie_val,Rating_val) = split_data(user_id,movie_id,Rating,0.1)
        filepath = './model_nobias/Model.{epoch:02d}-{val_loss:.4f}_dim13_nn.hdf5'
        callbacks = [EarlyStopping('val_loss', patience=10), ModelCheckpoint(filepath, save_best_only=False)]
        #model.fit([user_train, movie_train], Rating_train, epochs=200, validation_data=([user_val,movie_val],Rating_val), verbose=1, callbacks=callbacks)

    elif Training == 2:
        test_user_id, test_movie_id = load_test(sys.argv[1]);
        model = load_model("./model/Model.77-0.7637_dim8.hdf5")
        result = model.predict([test_user_id, test_movie_id])
        saveResult(sys.argv[2], result)
    else:
        test_user_id, test_movie_id = load_test(sys.argv[1]);
        model = load_model("./model/Model.42-0.7631_dim13.hdf5")
        result = model.predict([test_user_id, test_movie_id])
        model = load_model("./model/Model.62-0.7711_dim10.hdf5")
        result += model.predict([test_user_id, test_movie_id])
        model = load_model("./model/Model.77-0.7637_dim8.hdf5")
        result += model.predict([test_user_id, test_movie_id])
        #model = load_model("./model/Model.45-0.7718_dim15.hdf5")
        #result += model.predict([test_user_id, test_movie_id])
        #model = load_model("./model/Model.152-0.7767_dim5.hdf5")
        #result += model.predict([test_user_id, test_movie_id])
        result /= 3;
        saveResult(sys.argv[2], result)

if __name__ == '__main__':
    main();
