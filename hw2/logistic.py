import sys
import numpy as np
import csv

def read_fea(filename):
  with open(filename,'r') as file:
    sentences = file.readlines();
    i = 0;
    for sentence in sentences:
      if(i==0): i+=1;continue;
      words = sentence.split(',');
      if(i==1): total = np.array([map(float,words)]);i+=1;continue;
      total = np.concatenate((total,np.array([map(float,words)])),axis=0);
      i+=1;
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
  if(sys.argv[5]=="1"): logistic(train_fea,train_ans,test_fea);
  elif(sys.argv[5]=="2"): generative(train_fea,train_ans,test_fea);
  elif(sys.argv[5]=="3"): 
    yyy = generative(train_fea,train_ans,test_fea);
    train_fea,train_ans,test_fea = load_data();   
    best(train_fea,train_ans,test_fea,yyy);

def shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])

def logistic(train_fea,train_ans,test_fea):
  W = np.zeros((106+1,1));
  table = computeMean_Var(np.concatenate((train_fea,test_fea),axis=0));
  normalize(train_fea,table);normalize(test_fea,table);
  train_fea = np.insert(train_fea,0,1,axis=1);
  test_fea = np.insert(test_fea,0,1,axis=1);

  eta = 1; epoch = 0;
  totalgra = 0; conti = 0;  prevcost = 999;
  batch_size = 200;
  batch_num = int(np.floor(train_fea.shape[0]/batch_size));
 
  while(epoch<2000):
    train_fea,train_ans = shuffle(train_fea,train_ans);
    #for idx in range(batch_num): 
    #gradient = -np.transpose([np.mean((train_ans[idx*batch_size:(idx+1)*batch_size,:]-sigmoid(np.dot(train_fea[idx*batch_size:(idx+1)*batch_size,:],W)))*train_fea[idx*batch_size:(idx+1)*batch_size,:],axis=0)]);
    gradient = -np.transpose([np.mean((train_ans-sigmoid(np.dot(train_fea,W)))*train_fea,axis=0)]);

    if(epoch!=0): W = W - eta*gradient/np.sqrt(totalgra);
    else: W = W - eta*gradient;
    totalgra += np.power(gradient,2);

    y = (sigmoid(np.dot(train_fea,W))>0.5); y = y.astype(int)
    cost = np.mean(np.abs(train_ans-y))
    if(cost>=prevcost):conti += 1;
    else: conti = 0;
    prevcost = cost;
    epoch +=1;
    if (conti>10): break;

    if(epoch%10==0):
      print ("Training..., Epoch:"+str(epoch));

  print("End Training!!")
   
  ansy = (sigmoid(np.dot(test_fea,W))>0.5); ansy = ansy.astype(int);
  #print (np.dot(test_fea,W))
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD);
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();
  W2 = W.tolist();
  #with open("logis_model2.csv",'w') as mfile:
  #  strg = [];
  #  strg.append(W2);
  #  w = csv.writer(mfile);
  #  w.writerows(strg); 

  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg);

def calcost(u,X,Y,s,idx,thr):
  predict = np.zeros(np.shape(Y));
  if(s==1): select = (X[:,idx]>thr);
  else: select = (X[:,idx]<thr);
  predict[select,0] = 1;
  cost = (np.sum(np.abs(predict-Y)*u)/np.sum(u));
  return cost;

def decision_stump(u,X,Y,M1,M0):
  s = 1; idx = 0; thr = 0; cost = 1;
  for i in range(0,106):
    if(i==0 or i==1 or i==3 or i==4 or i==5):
      tempcost = calcost(u,X,Y,1,i,-999);
      if(tempcost<cost):
	cost = tempcost; s = 1; idx = i; thr = (-999);
      tempcost = calcost(u,X,Y,-1,i,-999);
      if(tempcost<cost):
	cost = tempcost; s = -1; idx = i; thr = (-999);
      diff = np.abs(M1[i]-M0[i])/5;
      lower = np.minimum(M1[i],M0[i])-2*diff;  
      for j in range(0,10):
	tempcost = calcost(u,X,Y,1,i,lower+j*diff)
	if(tempcost<cost):
	  cost = tempcost; s = 1; idx = i; thr = lower+j*diff;
	tempcost = calcost(u,X,Y,-1,i,lower+j*diff)
	if(tempcost<cost):
	  cost = tempcost; s = -1; idx = i; thr = lower+j*diff;
    else:
      tempcost = calcost(u,X,Y,1,i,-0.5);
      if(tempcost<cost):
	cost = tempcost; s = 1; idx = i; thr = (-0.5);
      tempcost = calcost(u,X,Y,1,i,0.5);
      if(tempcost<cost):
	cost = tempcost; s = 1; idx = i; thr = 0.5;
      tempcost = calcost(u,X,Y,-1,i,0.5);
      if(tempcost<cost):
	cost = tempcost; s = -1; idx = i; thr = 0.5;
      tempcost = calcost(u,X,Y,-1,i,-1.5);
      if(tempcost<cost):
	cost = tempcost; s = -1; idx = i; thr = (-1.5);
  return s,idx,thr,cost

def best(train_fea,train_ans,test_fea,yyy):
  table = computeMean_Var(np.concatenate((train_fea,test_fea),axis=0));
  normalize(train_fea,table);normalize(test_fea,table);

  train_1 = train_fea[train_ans[:,0]==1,:]; train_0 = train_fea[train_ans[:,0]==0,:];
  #ans_1 = train_ans[train_ans[:,0]==1,:];
  #train_fea = np.concatenate((train_fea,train_1),axis=0); train_ans = np.concatenate((train_ans,ans_1),axis=0);

  Mean_train1 = np.transpose([np.mean(train_1,axis=0)]);
  Mean_train0 = np.transpose([np.mean(train_0,axis=0)]);
  Mean_train = np.transpose([np.mean(train_fea,axis=0)]); 
  #Cov = np.cov(np.transpose(np.concatenate((train_fea,train_ans),axis=1))); print (Mean_train,Cov[:,106]);
  
  epoch = 0; u = np.ones(np.shape(train_ans))/(np.shape(train_ans)[0]);  
  alpha = []; totals = []; totalidx = []; totalthr = [];
  while(epoch<400):
    s,idx,thr,cost = decision_stump(u,train_fea,train_ans,Mean_train1,Mean_train0);
    #print (s,idx,thr,cost);
    predict = np.zeros(np.shape(train_ans));
    if(s==1): select = (train_fea[:,idx]>thr);
    else: select = (train_fea[:,idx]<thr);
    predict[select,0] = 1;
    bigger = (predict!=train_ans); smaller = (predict==train_ans); ratio = np.sqrt((1-cost)/cost);
    u[bigger[:,0],0] *= ratio; u[smaller[:,0],0] /= ratio;
    alpha.append(np.log(ratio)); totals.append(s);totalidx.append(idx);totalthr.append(thr);
    epoch += 1;

    if(epoch%50==0): 
      print("Training..., epoch:"+str(epoch));    
      alpha2 = np.array([alpha]); 
      num = 0; Xy = np.zeros((np.shape(train_fea)[0],1));
      while(num<np.shape(alpha2)[1]):
        tempy = -np.ones((np.shape(train_fea)[0],1));
        if(totals[num]==1): select = (train_fea[:,totalidx[num]]>totalthr[num]);
        else: select = (train_fea[:,totalidx[num]]<totalthr[num]);
        tempy[select,0] = 1; tempy = tempy.astype(float);
        tempy*=(alpha2[0,num]); 
        Xy+=tempy;
        num+=1; 
      Xy = (Xy>0); Xy = Xy.astype(int); 
      cost = np.mean(np.abs(Xy-train_ans)); 
      print ("Cost:"+str(cost))
  print ("Training Finish!");
  '''
  selectfea = np.zeros((np.shape(train_fea)[1],1));
  N1 = np.shape(train_1)[0]; N0 = np.shape(train_0)[0]; P1_0 = float(N1)/float(N0);
  for i in range(0,106):
    if(i<=5):
      selectfea[i,0] = 1; continue;
    one_1 = np.count_nonzero(train_1[:,i]);one_0 = np.count_nonzero(train_0[:,i]);one1_0 = float(one_1)/float(one_0);
    if (P1_0>one1_0): 
      if((P1_0-one1_0)/P1_0>0.05): selectfea[i,0] = 1;
    else: 
      if((one1_0-P1_0)/(1-P1_0)>0.05): selectfea[i,0] = 1;
  selectfea = selectfea.astype(bool);
  '''
  #selectfea = (np.absolute(Mean_train1-Mean_train0)>0.0001);
  #train_fea = (train_fea[:,selectfea[:,0]]);
  #test_fea = test_fea[:,selectfea[:,0]];
  #selectfea = np.where(Mean_train1==0);

  num = 0; ansy2 = np.zeros((np.shape(test_fea)[0],1));
  while(num<np.shape(alpha2)[1]):
    tempy = -np.ones((np.shape(test_fea)[0],1));
    if(totals[num]==1): select = (test_fea[:,totalidx[num]]>totalthr[num]);
    else: select = (test_fea[:,totalidx[num]]<totalthr[num]);
    tempy[select,0] = 1; tempy = tempy.astype(float);
    tempy*=(alpha2[0,num]);
    ansy2+=tempy;
    num+=1;
  ansy2 = (ansy2>0); ansy2 = ansy2.astype(int);
  with open("model.csv",'w') as mfile:
    strg = [];
    strg.append(alpha);strg.append(totals);strg.append(totalidx);strg.append(totalthr);
    w = csv.writer(mfile);
    w.writerows(strg); 
 
  '''
  train_fea = np.insert(train_fea,0,1,axis=1);
  test_fea = np.insert(test_fea,0,1,axis=1);
  W = np.zeros((np.shape(train_fea)[1],1));
  eta = 1; epoch = 0;
  totalgra = 0; conti = 0;  prevcost = 1; 
  while(epoch<3000):
    gradient = -np.transpose([np.mean((train_ans-sigmoid(np.dot(train_fea,W)))*train_fea,axis=0)]);
    if(epoch!=0): W = W - eta*gradient/np.sqrt(totalgra);
    else: W = W - eta*gradient;
    totalgra += (np.power(gradient,2));

    y = (sigmoid(np.dot(train_fea,W))>0.5); y = y.astype(int)
    cost = np.mean(np.abs(train_ans-y))
    print (cost);
    if(cost>=prevcost):conti += 1;
    else: conti = 0;
    prevcost = cost;
    epoch +=1;
    if (conti>=10): break;
   
  ansy = (sigmoid(np.dot(test_fea,W))>0.5); ansy = ansy.astype(int); 
  '''
  #ansy = ((ansy+ansy2+yyy)>1.5); ansy = ansy.astype(int);
  ansy = ansy2;
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD);
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();
  
  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg); 
  

def generative(train_fea,train_ans,test_fea):

  table = computeMean_Var2(train_fea);normalize(train_fea,table);normalize(test_fea,table);
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

  INV_Cov = INV_Cov/np.linalg.norm(INV_Cov,axis=0);
  b = -np.dot(np.dot(np.transpose(Mean_train1),INV_Cov),Mean_train1)/2+np.dot(np.dot(np.transpose(Mean_train0),INV_Cov),Mean_train0)/2+np.log(float(N1)/float(N0));
  
  y = (sigmoid(np.dot(train_fea,W)+b)>0.5); y = y.astype(int)
  cost = np.mean(np.abs(train_ans-y))
  print ("Generating the generative model...");
   
  ansy = (sigmoid(np.dot(test_fea,W)+b)>0.5); ansy = ansy.astype(int); 
  XDD = [np.arange(1,np.shape(ansy)[0]+1)]; XDD = np.transpose(XDD); yyy = ansy;
  ansy = np.concatenate((XDD,ansy),axis=1); ansy = ansy.tolist();

  print ("Predict the answer!!");

  with open(sys.argv[4],'w') as wfile:
    strg = [["id","label"]]+ansy;
    w = csv.writer(wfile);
    w.writerows(strg);
  return yyy 
  
if __name__=='__main__':
  main();
