import os
import sys
import numpy as np

def readcsv():
  with open("train.csv",'r') as file:
    sentences = file.readlines();
    i = 0;
    total = np.array([]);
    for sentence in sentences:
      if(i==0):
        i+=1;continue;
      words = sentence.split(',');
      if(i%18==1):
        data = np.array([map(float,words[3:])]);i+=1;
        continue;
      if(i%18!=11 and i%18!=14):
        data=np.concatenate((data,np.array([map(float,words[3:])])),axis=0);
      if(i==18):
        total=data;
      elif(i%18==0):
        total = np.concatenate((total,data),axis=1);
      i+=1;
        #print total;break;
  return total;

def readtest():
  with open("test_X.csv",'r') as file2:
   sentences = file2.readlines();i=0; name = list();
   for sentence in sentences:
    word = sentence.split(',');
    if(i%18==0):
      data = np.array([map(float,word[2:])]);i+=1;continue;
    if(i%18!=10 and i%18!=13):
      data = np.concatenate((data,np.array([map(float,word[2:])])),axis=1);
    if(i==17):
      total = data;name.append(word[0]);
    elif(i%18==17):
      total = np.concatenate((total,data),axis=0);name.append(word[0]);
    i+=1;
  total = np.insert(total,0,1,axis=1); 
  return name,total;

def main():
  traindata = readcsv();
  W = np.zeros(145);
  for i in range(5751):
    if(i==0):
      X = [(np.reshape(traindata[:,i:i+9],144))];Y = [traindata[9,i+9]];
    else:
      X = np.concatenate((X,[(np.reshape(traindata[:,i:i+9],144))]),axis=0);
      Y = np.concatenate((Y,[traindata[9,i+9]]),axis=0);
  X = np.insert(X,0,1,axis=1);
  Y = np.transpose([Y]);
  B = np.linalg.pinv(X);
  eta = 0.01;
  while(true):
    randchoose = np.random.randperm(5751);
    gradient = 
    W = W - eta
  #W = np.dot(B,Y);
  name,test_data = readtest();
  ansy = np.dot(test_data,W);
  with open("PM_2_5.csv",'w') as file3:
    strg = "id,value\n";
    for j in range(240):
      strg+=name[j];
      strg+=",";strg+=str(ansy[j,0]);strg+="\n";
    file3.write(strg);

if __name__=='__main__':
  main()  
