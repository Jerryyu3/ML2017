import random
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
      if(i%18==(int(sys.argv[4])+1)):
        #data = np.array([map(float,words[3:])]);
        #data = np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)]); 
        data = np.array([map(float,words[3:]),np.power(map(float,words[3:]),2),np.power(map(float,words[3:]),3)]);
        i+=1;
        continue;
      elif(i%18!=11):#if((i%18<=4 and i%18>=3) or (i%18<=16 and i%18>=13) or (i%18==10)):
        #data=np.concatenate((data,np.array([map(float,words[3:])])),axis=0);
        #data=np.concatenate((data,np.array([map(float,words[3:]),np.power(map(float,words[3:]),2)])),axis=0);
        data=np.concatenate((data,np.array([map(float,words[3:]),np.power(map(float,words[3:]),2),np.power(map(float,words[3:]),3)])),axis=0);
      '''
      else:
	for jjj in range(3,len(words)):
	  if(words[jjj]=="NR"):words[jjj]="0";
	  elif(words[jjj]=="NR\r\n"):words[jjj]="0\r\n";
	data=np.concatenate((data,np.array([map(float,words[3:]),np.power(map(float,words[3:]),2),np.power(map(float,words[3:]),3)])),axis=0); 
      '''
      if(i==18):
        total=data;
      elif(i%18==0):
        total = np.concatenate((total,data),axis=1);
      i+=1;
      #if(i==37):
      #  print total;#break;
  return total;

def readtest():
  with open(sys.argv[2],'r') as file2:
    sentences = file2.readlines();i=0; name = list();
    for sentence in sentences:
      word = sentence.split(',');
      if(i%18==0):#int(sys.argv[4])):
        data = np.array([map(float,word[2:])]);#data = data [:,4:9]; 
        dataa = np.array([np.power(map(float,word[2:]),2)]);#dataa = dataa[:,4:9];
        data = np.concatenate((data,dataa),axis=1);
	#data3 = np.array([np.power(map(float,word[2:]),3)]);
	#data = np.concatenate((data,data3),axis=1);
        i+=1;continue; 
      #elif(i%18==10):
      #for jjj in range(2,len(word)):
      #	  if(word[jjj]=="NR"):word[jjj]="0";
      #	  elif(word[jjj]=="NR\n"):word[jjj]="0\n";
      #	data2 = np.array([map(float,word[2:])]);
      #  data22 = np.array([np.power(map(float,word[2:]),2)]);
      #  data2 = np.concatenate((data2,data22),axis=1);
      #	data23 = np.array([np.power(map(float,word[2:]),3)]);
      #	data2 = np.concatenate((data2,data23),axis=1);
      #  data = np.concatenate((data,data2),axis=1);
      elif(i%18==1 or i%18==8 or i%18==9 or i%18==13):
	data2 = np.array([map(float,word[2:])]);#data2 = data2[:,4:9];
        data22 = np.array([np.power(map(float,word[2:]),2)]);#data22 = data22[:,4:9];
        data2 = np.concatenate((data2,data22),axis=1);
	data23 = np.array([np.power(map(float,word[2:]),3)]);
	data2 = np.concatenate((data2,data23),axis=1);
        data = np.concatenate((data,data2),axis=1);
      #elif(i%18==0):#if((i%18>=2 and i%18<=3)or(i%18>=12 and i%18<=15)or(i%18==9)):
      #  data2 = np.array([map(float,word[2:])]);#data2 = data2[:,4:9];
      #  data22 = np.array([np.power(map(float,word[2:]),2)]);#data22 = data22[:,4:9];
      #  data2 = np.concatenate((data2,data22),axis=1);
      #  data = np.concatenate((data,data2),axis=1);
      elif(i%18==3 or i%18==5 or i%18==6 or i%18==7 or i%18==11 or i%18==12 or i%18==14 or i%18==15):#if((i%18>=2 and i%18<=3)or(i%18>=12 and i%18<=15)or(i%18==9)):
        data2 = np.array([map(float,word[2:])]);
        #data22 = np.array([np.power(map(float,word[2:]),2)]);
        #data2 = np.concatenate((data2,data22),axis=1);
        data = np.concatenate((data,data2),axis=1);
      elif(i%18==2):
	data2 = np.array([np.power(map(float,word[2:]),1)]);
	data23 = np.array([np.power(map(float,word[2:]),3)]);
	data2 = np.concatenate((data2,data23),axis=1);
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
  size_train = np.shape(traindata); 
  #for i in range(size_train[0]):
  #  XDD = np.sum(traindata[i,:]*traindata[27,:])/(np.linalg.norm(traindata[i,:]))/np.linalg.norm(traindata[27,:]);
  #  print (i,XDD);
  #select = np.array([0,1,24,25]);
  #select = np.array([0,8,9,11,13,14,15,16]);
  #select = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]);
  #select = np.array([0,1,16,17,18,19,22,23,26,27,28,29,30,31,32,33]);
  #select = np.array([0,1,2,3,4,6,10,11,12,14,16,17,18,19,20,21,22,24,25,26,28,30,32]);
  select = np.array([0,1,3,4,5,6,8,9,15,18,21,24,25,26,27,28,29,30,33,36,37,38,39,42]);
  #select = np.array([0,8,9,11,]);
  #select = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]);
  #select = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]);
  selectsize = 24;
  print ("choosing features...");
  W = np.ones((selectsize*9+1,1));
  #print (traindata[[1,3,5],:]);
  for i in range(size_train[1]-9):
    if(traindata[27,i+9]==-1 and i<size_train[1]-10):
    	traindata[27,i+9] = np.minimum(0,(traindata[27,i+8]+traindata[27,i+10])/2);
    if(i==0):
      X = [(np.reshape(traindata[select,i:i+9],selectsize*9))];Y = [traindata[27,i+9]]
    elif(i%480<471):#i<size_train[1]-160):
      X = np.concatenate((X,[(np.reshape(traindata[select,i:i+9],selectsize*9))]),axis=0);
      Y = np.concatenate((Y,[traindata[27,i+9]]),axis=0);
    elif(i%480==440):#i==size_train[1]-160):
      valX = [(np.reshape(traindata[select,i:i+9],selectsize*9))];valY = [traindata[27,i+9]];
    elif(i%480<=470):
      valX = np.concatenate((valX,[(np.reshape(traindata[select,i:i+9],selectsize*9))]),axis=0);
      valY = np.concatenate((valY,[traindata[27,i+9]]),axis=0);
 
  X = np.insert(X,0,1,axis=1);
  #print (np.mean(X[:,1]),np.var(X[:,1]),(X[:,1]-np.mean(X[:,1]))/np.var(X[:,1]));
  table = computeMean_Var(X); normalize(X,table);
  Y = np.transpose([Y]);
  
  lamda = 0;
  B = np.dot(np.linalg.inv((np.dot(np.transpose(X),X)+lamda*np.eye(selectsize*9+1))),np.transpose(X));
  #valX = np.insert(valX,0,1,axis=1); #validation 
  #valY = np.transpose([valY]); #validation
  #normalize(valX,table); 
  
  eta = 0.03; epoch = 0; 
  totalgra = 0; 
  conti = 0; prevcost = 999999999;
  '''
  while(epoch<20000):
    #randchoose = np.random.randperm(5600);
    gradient = np.zeros((selectsize*9+1,1));
    #print np.mean((Y-(np.dot(X,W)))*(-X),axis=0)   
    gradient += (2*np.transpose([np.mean((Y-(np.dot(X,W)))*(-X),axis=0)])+2*lamda*(W));
    #totalgra += np.power(gradient,2);
    W = W - eta * gradient#/np.sqrt(totalgra);

    ansvaly = np.dot(valX,W);
    cost = np.sqrt(np.mean(np.power(np.subtract(ansvaly,valY),2)));
    if(cost>prevcost): conti+=1;
    else: conti=0;
    prevcost = cost;
    print cost;
    epoch += 1;
    if conti>=2: break;
  #print (W)
  '''
  W = np.dot(B,Y);
  print ("Training finished!!");
  #validation
  #ansvaly = np.dot(valX,W);
  #cost = np.sqrt(np.mean(np.power(np.subtract(ansvaly,valY),2))); 
  #print (cost);
  #with open("model.csv",'w') as wfile:
  #  str1 = "w\n";
  #  for j in range(np.shape(W)[0]):
  #	str1+=str(W[j,0]);str1+="\n";
  #  wfile.write(str1);

  name,test_data = readtest(); 
  normalize(test_data,table);
  #print (np.shape(test_data));
  #print (test_data);
  ansy = np.dot(test_data,W);  
  #print (ansy);
  with open(sys.argv[3],'w') as file3:
    strg = "id,value\n";
    for j in range(240):
      strg+=name[j];
      strg+=",";strg+=str(ansy[j,0]);strg+="\n";
    file3.write(strg);
  

if __name__=='__main__':
  main()  
