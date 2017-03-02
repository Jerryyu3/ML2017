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
      if(i%18==(int(sys.argv[1])+1)):
        #data = np.array([map(float,words[3:])]);
        data = np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)]);
        i+=1;
        continue;
      if((i%18<=4 and i%18>=3) or (i%18<=16 and i%18>=13) or (i%18==10)):
        #data=np.concatenate((data,np.array([map(float,words[3:])])),axis=0);
        data=np.concatenate((data,np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)])),axis=0);
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
    if(i%18==int(sys.argv[1])):
      data = np.array([map(float,word[2:])]);data = data [:,4:9]; 
      dataa = np.array([np.power(map(float,word[2:]),2)]);dataa = dataa[:,4:9];
      data = np.concatenate((data,dataa),axis=1);
      i+=1;continue;
    if((i%18>=2 and i%18<=3)or(i%18>=12 and i%18<=15)or(i%18==9)):
      data2 = np.array([map(float,word[2:])]);data2 = data2[:,4:9];
      data22 = np.array([np.power(map(float,word[2:]),2)]);data22 = data22[:,4:9];
      data2 = np.concatenate((data2,data22),axis=1);
      data = np.concatenate((data,data2),axis=1);
    if(i==17):
      total = data;name.append(word[0]);
    elif(i%18==17):
      total = np.concatenate((total,data),axis=0);name.append(word[0]);
    i+=1;
  total = np.insert(total,0,1,axis=1); 
  return name,total;

def main():
  traindata = readcsv();
  #W = np.zeros(145);
  for i in range(5751):
    if(i==0):
      X = [(np.reshape(traindata[:,i+4:i+9],80))];Y = [traindata[6,i+9]];
    elif(i<5751):
      X = np.concatenate((X,[(np.reshape(traindata[:,i+4:i+9],80))]),axis=0);
      Y = np.concatenate((Y,[traindata[6,i+9]]),axis=0);
    elif(i==5600):
      valX = [(np.reshape(traindata[:,i+4:i+9],80))];valY = [traindata[6,i+9]];
    else:
      valX = np.concatenate((valX,[(np.reshape(traindata[:,i+4:i+9],80))]),axis=0);
      valY = np.concatenate((valY,[traindata[6,i+9]]),axis=0);
 
  X = np.insert(X,0,1,axis=1);
  #valX = np.insert(valX,0,1,axis=1);
  Y = np.transpose([Y]);
  print Y
  lamda = 50;
  B = np.dot(np.linalg.inv((np.dot(np.transpose(X),X)+lamda*np.eye(81))),np.transpose(X));
  #valY = np.transpose([valY]);
  #B = np.linalg.pinv(X);
  #eta = 0.01;
  #while(true):
  #  randchoose = np.random.randperm(5751);
  #  gradient = 
  #  W = W - eta
  W = np.dot(B,Y);
  #print W
  #ansvaly = np.dot(valX,W);
  #cost = np.sum(np.power(np.subtract(ansvaly,valY),2));
  #print cost;
  
  name,test_data = readtest();
  print np.shape(test_data);
  ansy = np.dot(test_data,W);  
  with open("PM_2_5.csv",'w') as file3:
    strg = "id,value\n";
    for j in range(240):
      strg+=name[j];
      strg+=",";strg+=str(ansy[j,0]);strg+="\n";
    file3.write(strg);
  

if __name__=='__main__':
  main()  
