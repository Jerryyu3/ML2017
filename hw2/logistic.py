import sys
import numpy as np
import csv

def read_fea(filename):
  with open(filename,'r') as file:
    #kkk = file.read();
    #kk = kkk.split(',','\n');
    #print (kk)
    sentences = file.readlines();
    i = 0;
    for sentence in sentences:
      if(i==0): i+=1;continue;
      words = sentence.split(',');
      if(i==1): total = np.array([map(float,words)]);i+=1;continue;
      total = np.concatenate((total,np.array([map(float,words)])),axis=0);
      i+=1;
      #print (i);
    #print (total);
    return total;

def read_train_ans():
  with open(sys.argv[2],'r') as file2:
    kkk = file2.read();
    kkk = kkk.split();
    Y = np.array([map(float,kkk)]);
    Y = np.transpose(Y);
    return Y;

def load_data():
  X_train = np.delete(np.genfromtxt(sys.argv[1], delimiter=','), 0, 0)
  Y_train = [np.genfromtxt(sys.argv[2], delimiter=',')];
  X_test = np.delete(np.genfromtxt(sys.argv[3], delimiter=','), 0, 0)
  Y_train = np.transpose(Y_train)
  return X_train, Y_train, X_test

def sigmoid(z):
  return 1/(1+np.exp(-z));

def computeMean_Var(X):
  length = (np.shape(X)[1]);
  table = np.zeros((2,6));
  table[0][0] = np.mean(X[:,0]); table[1][0] = np.std(X[:,0]);
  table[0][1] = np.mean(X[:,1]); table[1][1] = np.std(X[:,1]);
  table[0][3] = np.mean(X[:,3]); table[1][3] = np.std(X[:,3]);
  table[0][4] = np.mean(X[:,4]); table[1][4] = np.std(X[:,4]);
  table[0][5] = np.mean(X[:,5]); table[1][5] = np.std(X[:,5]);
  return table;

def computeMean_Var2(X):
  length = (np.shape(X)[1]);
  table = np.zeros((2,length));
  for i in range(length):
    table[0][i] = np.mean(X[:,i]); table[1][i] = np.std(X[:,i]);
  return table;

def normalize(lX,table):
  length = np.shape(table)[1];
  for i in range(length):
    if(table[1,i]>0): lX[:,i] = (lX[:,i]-table[0,i])/table[1,i];

def main():
  #train_fea = read_fea(sys.argv[1]);
  #train_ans = read_train_ans();
  #test_fea = read_fea(sys.argv[3]);
  train_fea,train_ans,test_fea = load_data();
  #with open("check.csv",'w')as check:
  #  fff = train_fea[train_ans[0:999,0]>0.5,:].tolist();
  #  w2 = csv.writer(check); w2.writerows(fff);
  #print (np.shape(train_fea),np.shape(train_ans),np.shape(test_fea));
  if(sys.argv[5]=="1"): logistic(train_fea,train_ans,test_fea);
  elif(sys.argv[5]=="2"): generative(train_fea,train_ans,test_fea);
  elif(sys.argv[5]=="3"): best(train_fea,train_ans,test_fea);

def logistic(train_fea,train_ans,test_fea):
  W = np.zeros((106+1,1));
  table = computeMean_Var(np.concatenate((train_fea,test_fea),axis=0));
  normalize(train_fea,table);normalize(test_fea,table);
  train_fea = np.insert(train_fea,0,1,axis=1);
  test_fea = np.insert(test_fea,0,1,axis=1);

  eta = 0.5; epoch = 0;
  totalgra = 0; conti = 0;  prevcost = 999; 
  #train_ans = (train_ans[0:9,:]);
  while(epoch<3000):
    gradient = -np.transpose([np.mean((train_ans-sigmoid(np.dot(train_fea,W)))*train_fea,axis=0)]);
    if(epoch!=0): W = W - eta*gradient/np.sqrt(totalgra);
    else: W = W - eta*gradient;
    totalgra += np.power(gradient,2);

    y = (sigmoid(np.dot(train_fea,W))>0.5); y = y.astype(int)
    #print (np.dot(train_fea,W));
    cost = np.mean(np.abs(train_ans-y))
    print (cost);
    if(cost>=prevcost):conti += 1;
    else: conti = 0;
    prevcost = cost;
    epoch +=1;
    if (conti>=30): break;
   
  ansy = (sigmoid(np.dot(test_fea,W))>0.5); ansy = ansy.astype(int);
  print (np.dot(test_fea,W))
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD);
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();

  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg);

def best(train_fea,train_ans,test_fea):
  table = computeMean_Var(np.concatenate((train_fea,test_fea),axis=0));
  normalize(train_fea,table);normalize(test_fea,table);

  train_1 = train_fea[train_ans[:,0]==1,:]; train_0 = train_fea[train_ans[:,0]==0,:];
  Mean_train1 = np.transpose([np.mean(train_1,axis=0)]);
  Mean_train0 = np.transpose([np.mean(train_0,axis=0)]);
  
  #selectfea = (np.absolute(Mean_train1-Mean_train0)>0.0001);
  #train_fea = (train_fea[:,selectfea[:,0]]);
  #test_fea = test_fea[:,selectfea[:,0]];
   
  train_fea = np.insert(train_fea,0,1,axis=1);
  test_fea = np.insert(test_fea,0,1,axis=1);

  W = np.zeros((np.shape(train_fea)[1],1));
  eta = 1; epoch = 0;
  totalgra = 0; conti = 0;  prevcost = 999; 
  while(epoch<3000):
    gradient = -np.transpose([np.mean((train_ans-sigmoid(np.dot(train_fea,W)))*train_fea,axis=0)]);
    if(epoch!=0): W = W - eta*gradient/np.sqrt(totalgra);
    else: W = W - eta*gradient;
    totalgra += np.power(gradient,2);

    y = (sigmoid(np.dot(train_fea,W))>0.5); y = y.astype(int)
    cost = np.mean(np.abs(train_ans-y))
    print (cost);
    if(cost>=prevcost):conti += 1;
    else: conti = 0;
    prevcost = cost;
    epoch +=1;
    if (conti>=10): break;
   
  ansy = (sigmoid(np.dot(test_fea,W))>0.5); ansy = ansy.astype(int);
  print (np.dot(test_fea,W))
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD);
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();

  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg); 
  
def generative(train_fea,train_ans,test_fea):

  table = computeMean_Var(train_fea);normalize(train_fea,table);normalize(test_fea,table);
  train_1 = train_fea[train_ans[:,0]==1,:]; train_0 = train_fea[train_ans[:,0]==0,:];
  Mean_train1 = np.transpose([np.mean(train_1,axis=0)]); Cov_train1 = np.cov(np.transpose(train_1));
  Mean_train0 = np.transpose([np.mean(train_0,axis=0)]); Cov_train0 = np.cov(np.transpose(train_0));
  #with open("check.csv",'w')as check:
  #  fff = np.transpose(Mean_train1).tolist(); print(fff)
  #  fff += (np.transpose(Mean_train0).tolist()); print(fff)
  #  w2 = csv.writer(check); w2.writerows(fff);#print (np.shape(Mean_train1),np.shape(Cov_train1));
  
  N1 = np.shape(train_1)[0]; N0 = np.shape(train_0)[0]; P1 = float(N1)/float(N1 + N0); P0 = float(N0)/float(N1+N0);
  Cov_train = P1*Cov_train1 + P0*Cov_train0; INV_Cov = np.linalg.inv(Cov_train);
  
  W = np.dot(np.transpose(Mean_train1-Mean_train0),INV_Cov); W = np.transpose(W);
  #print (INV_Cov/np.linalg.norm(INV_Cov));
  INV_Cov = INV_Cov/np.linalg.norm(INV_Cov,axis=0);
  b = -np.dot(np.dot(np.transpose(Mean_train1),INV_Cov),Mean_train1)/2+np.dot(np.dot(np.transpose(Mean_train0),INV_Cov),Mean_train0)/2+np.log(float(N1)/float(N0));
  
  y = (sigmoid(np.dot(train_fea,W)+b)>0.5); y = y.astype(int)
  cost = np.mean(np.abs(train_ans-y))
  print (cost);
   
  ansy = (sigmoid(np.dot(test_fea,W)+b)>0.5); ansy = ansy.astype(int);
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD);
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();

  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg); 
  
if __name__=='__main__':
  main();
