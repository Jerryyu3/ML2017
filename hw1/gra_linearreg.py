#import os
import sys
import numpy as np

def readcsv():
  with open(sys.argv[1],'r') as file:
    sentences = file.readlines();
    i = 0;
    total = np.array([]);
    for sentence in sentences:
      if(i==0):
        i+=1;continue;
      words = sentence.split(',');
      if(i%18==(int(sys.argv[2])+1)):
        data = np.array([map(float,words[3:])]);
        #data = np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)]); 
        i+=1;
        continue;
      elif(i%18!=11):#if((i%18<=4 and i%18>=3) or (i%18<=16 and i%18>=13) or (i%18==10)):
        data=np.concatenate((data,np.array([map(float,words[3:])])),axis=0);
        #data=np.concatenate((data,np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)])),axis=0);
      if(i==18):
        total=data;
      elif(i%18==0):
        total = np.concatenate((total,data),axis=1);
      i+=1;
      #if(i==37):
      #  print total;#break;
  return total;

def readtest():
  with open("test_X.csv",'r') as file2:
    sentences = file2.readlines();i=0; name = list();
    for sentence in sentences:
      word = sentence.split(',');
      if(i%18==int(sys.argv[2])):
        data = np.array([map(float,word[2:])]);#data = data [:,4:9]; 
        #dataa = np.array([np.power(map(float,word[2:]),2)]);#dataa = dataa[:,4:9];
        #data = np.concatenate((data,dataa),axis=1);
        i+=1;continue;
      elif(i%18!=10):#if((i%18>=2 and i%18<=3)or(i%18>=12 and i%18<=15)or(i%18==9)):
        data2 = np.array([map(float,word[2:])]);#data2 = data2[:,4:9];
        #data22 = np.array([np.power(map(float,word[2:]),2)]);#data22 = data22[:,4:9];
        #data2 = np.concatenate((data2,data22),axis=1);
        data = np.concatenate((data,data2),axis=1);
      if(i==17):
        total = data;name.append(word[0]);
      elif(i%18==17):
        total = np.concatenate((total,data),axis=0);name.append(word[0]);
      i+=1;
  total = np.insert(total,0,1,axis=1); 
  return name,total;

def computeMean_Var(X):
  length = (np.shape(X)[1]);
  table = np.zeros((2,length));
  for i in range(length):
    table[0][i] = np.mean(X[:,i]);table[1][i] = np.std(X[:,i]);
  return table;

def normalize(lX,table):
  number = np.shape(lX)[0];length = np.shape(lX)[1]; 
  for i in range(length):
    #for j in range(number):
    if(table[1,i]>0):
      lX[:,i] = (lX[:,i]-table[0,i])/table[1,i];

def main():
  traindata = readcsv();
  W = np.zeros((154,1));
  for i in range(5751):
    if(i==0):
      X = [(np.reshape(traindata[:,i:i+9],153))];Y = [traindata[9,i+9]]
    elif(i<5600):
      X = np.concatenate((X,[(np.reshape(traindata[:,i:i+9],153))]),axis=0);
      Y = np.concatenate((Y,[traindata[9,i+9]]),axis=0);
    elif(i==5600):
      valX = [(np.reshape(traindata[:,i:i+9],153))];valY = [traindata[9,i+9]];
    else:
      valX = np.concatenate((valX,[(np.reshape(traindata[:,i:i+9],153))]),axis=0);
      valY = np.concatenate((valY,[traindata[9,i+9]]),axis=0);
 
  X = np.insert(X,0,1,axis=1);
  #print (np.mean(X[:,1]),np.var(X[:,1]),(X[:,1]-np.mean(X[:,1]))/np.var(X[:,1]));
  #table = computeMean_Var(X); #normalize(X,table);
  valX = np.insert(valX,0,1,axis=1);
  Y = np.transpose([Y]);
  #print (X)
  lamda = 5;
  B = np.dot(np.linalg.inv((np.dot(np.transpose(X),X)+lamda*np.eye(154))),np.transpose(X));
  valY = np.transpose([valY]);
  #B = np.linalg.pinv(X);
  '''
  eta = 0.01; epoch = 0;
  while(epoch<1):
    #randchoose = np.random.randperm(5600);
    gradient = np.zeros((307,1));
    for i in range(5600):
	#print np.shape([X[i,:]])
	gradient += 2*np.transpose((Y[i]-(np.dot([X[i,:]],W)))*([-X[i,:]]));
	#if i==0:
	#print (Y[i]-(np.dot([X[i,:]],W)),[-X[i,:]],gradient);
	#print (sum(gradient));
	#raw_input();
	#W = W - eta * gradient;
    gradient/=5600;
    W = W - eta * gradient;
    #print (W);
    #raw_input();
    epoch += 1;
  #print (W)
  '''
  W = np.dot(B,Y);
  print (W)
  #normalize(valX,table);
  ansvaly = np.dot(valX,W);
  cost = np.sqrt(np.sum(np.power(np.subtract(ansvaly,valY),2)));
  print cost;
  
  name,test_data = readtest(); #normalize(test_data,table);
  #print (np.shape(test_data));
  #print (test_data);
  ansy = np.dot(test_data,W);  
  print (ansy);
  with open("PM_2_5.csv",'w') as file3:
    strg = "id,value\n";
    for j in range(240):
      strg+=name[j];
      strg+=",";strg+=str(ansy[j,0]);strg+="\n";
    file3.write(strg);
  

if __name__=='__main__':
  main()  
