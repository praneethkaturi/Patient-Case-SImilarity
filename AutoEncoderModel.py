#!/usr/bin/env python
# coding: utf-8

# In[293]:


from google.colab import drive
drive.mount('/content/gdrive', force_remount = True)


# In[ ]:


import pandas as pd
import numpy as np
import keras
from keras import Input
from keras.layers import Dense
from keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


demo = pd.read_csv("/content/gdrive/My Drive/Smart India Hackathon/THE DATASET/Demographics.csv")
case = pd.read_csv("/content/gdrive/My Drive/Smart India Hackathon/THE DATASET/Case_Documents.csv")


# In[334]:


case.head()


# In[ ]:


df = pd.read_csv("/content/gdrive/My Drive/Smart India Hackathon/THE DATASET/Demographics.csv")


# In[ ]:


#demographics preprocessing
to_datetime = ['DateOfBirth', 'DateOfAdmission', 'DateOfDischarge']

for feature in to_datetime:
  demo[feature] = pd.to_datetime(demo[feature], format = '%Y/%m/%d')
  
demo['DurationOfStay'] = demo['DateOfDischarge'] - demo['DateOfAdmission']
demo['Age'] = demo['DateOfAdmission'] - demo.DateOfBirth
demo['Age'] = np.array([[demo['Age'][i].days/365]for i in range(demo.shape[0])])
demo['Age'] = pd.to_numeric(demo['Age'])
demo['Age'] = np.ceil(demo['Age'])
demo.drop(labels= ['DateOfDischarge', 'DateOfAdmission'], axis = 1, inplace= True)
#demo = pd.get_dummies(demo, columns= ['DischargeDisposition', 'ServiceLine', 'PatientClass', 'Gender'])
demo.drop(labels= 'DateOfBirth', inplace= True, axis = 1)
demo['DurationOfStay'] = np.array([[demo['DurationOfStay'][i].seconds]for i in range(demo.shape[0])])


# In[ ]:


demo = pd.get_dummies(demo, columns= ['DischargeDisposition', 'ServiceLine', 'PatientClass', 'Gender'])


# In[337]:


case.head()


# In[ ]:


uni = case.DocType.unique()
for new in uni:
  demo[new] = np.zeros(demo.shape[0])


# In[ ]:


cases = case.CaseId.unique()


# In[ ]:


list(case[case.CaseId == 5795].DocType.unique())


# In[ ]:


docType = {}
for new in cases:
  unique = case[case.CaseId == new].DocType.unique()
  docType[new] = list(unique)


# In[302]:


demo.head()


# In[ ]:


ind = demo[demo.CaseId == 1].index


# In[ ]:


demo.loc[ind, ['FIST/SINUS', 'ServiceLine']]


# In[ ]:


len(cases)


# In[ ]:


np.ones(4).shape


# In[ ]:


for new in cases:
  get = docType[new]
  ind = demo[demo.CaseId == new].index
  for feat in get: 
    demo.loc[ind, feat] = 1


# In[ ]:


demo.to_csv('/content/gdrive/My Drive/Demo_processed.csv')


# In[ ]:


demo.drop(labels= 'CaseId', inplace= True, axis = 1) ##dont remove now


# In[332]:


demo.drop(labels= a, inplace= True, axis = 1)


# In[ ]:


#normalizing the dataset
ss = StandardScaler()
ss.fit(demo.values)
X = ss.transform(demo.values)


# In[ ]:


def enc_dec(input_shape):
  
  input_ = Input(shape=(513,))
  encoded = Dense(1024, activation='relu')(input_)
  encoded = Dense(512, activation='relu')(encoded)
  encoded = Dense(256, activation='relu')(encoded)
  encoded = Dense(512, activation='sigmoid')(encoded)

  decoded = Dense(256, activation='relu')(encoded)
  decoded = Dense(512, activation='relu')(decoded)
  decoded = Dense(512, activation='relu')(decoded)
  decoded = Dense(513)(decoded)

  autoencoder = Model(input_, decoded)
  encoder = Model(input_, encoded)
  
  return autoencoder, encoder


# In[ ]:


input_shape = (X.shape[1],)
autoencoder, encoder = enc_dec(input_shape)


# In[ ]:


autoencoder.compile(loss=  'mse', optimizer = 'adam', metrics = ['accuracy'])


# In[316]:


train_history = autoencoder.fit(X, X, epochs=100, batch_size=512, validation_split= 0.1)


# In[ ]:


enc = encoder.predict(X)
sim = cosine_similarity(enc, enc)
X_dec = autoencoder.predict(X)


# In[ ]:


import pickle
import pandas as pd
import csv
import time
import datetime
import os
import glob

class Patient:
  def fetchDocuments(self):
    column_names = []
    documents = []
    f = open( "/content/gdrive/My Drive/Smart India Hackathon/THE DATASET/Case_Documents.csv" , "r" )
    csvFile = csv.reader( f )
    for row in csvFile:
      if len( column_names ) == 0:
        column_names = row
        continue
      if int(row[0]) == int(self.caseID):
        documents.append( {
          'CaseId' : int( row[0] ) ,
          'DocName' : row[1] ,
          'DocType' : row[2] ,
          'DateOfService' : datetime.datetime.strptime( row[3] , "%Y-%m-%d %H:%M:%S" ),
          'PhysicianID' : int( row[4] ),
          'Speciality' : row[5]
        } )

    f.close()

    return documents
  
  def combineDocuments(self):
    ret = ""
    for item in self.documents:
      ret = ret + str( item['DocType'] ) + " "
      f = open( "/content/gdrive/My Drive/Smart India Hackathon/data/Raw_Texts/temp/"+item['DocName'] , "r" )
      rd = f.read()
      rd = rd.split("\n")
      ret = ret + ( " ".join( rd ) ).strip() + " "
      f.close()
    return ret

  def combineDocumentsModified(self):
    ret = ""
    iden = str( self.documents[0]['DocName'] )[:6]
    files = []
    for file in glob.glob("/content/gdrive/My Drive/Smart India Hackathon/data/Raw_Texts/temp/*.*"):
      if iden in file:
        files.append( file )
    for file in files:
      print( file )
      f = open( file , "r" )
      rd = f.read()
      rd = rd.split("\n")
      ret = ret + " ".join( rd ) + " "
    return ret.strip()
    
  def fetchSpecialities(self):
    ret = set()
    for item in self.documents:
      ret.add( str(item['Speciality']).lower() )
    return ret

  def documentContentDict(self):
    ret = []
    for item in self.documents:
      f = open( "/content/gdrive/My Drive/Smart India Hackathon/data/Raw_Texts/temp/"+item['DocName'] , "r" )
      rd = f.read()
      rd = rd.split("\n")
      ret.append( { 'DocType' : item['DocType'] , 'DocContent' : ( " ".join( rd ) ).strip() } )
      f.close()
    return ret

  def documentContentDictModified(self):
    ret = []
    iden = str( self.documents[0]['DocName'] )[:6]
    files = []
    for file in glob.glob("/content/gdrive/My Drive/Smart India Hackathon/data/Raw_Texts/temp/*.*"):
      if iden in file:
        files.append( file )
    for file in files:
      f = open( file , "r" )
      rd = f.read()
      rd = rd.split("\n")
      ret.append( ( " ".join( rd ) ).strip() )
    return ret

  def fetchXmlDocumentFnamesModified(self):
    ret = []
    iden = str( self.documents[0]['DocName'] )[:6]
    files = []
    for file in glob.glob("/content/gdrive/My Drive/Smart India Hackathon/data/NER_XML/*.xml"):
      if iden in file:
        files.append( file )
    return files
  
  def __init__(self, caseID, dob, gender, dateOfAdmission, dateOfDischarge, patientClass, statusID, serviceLine, payerID, dischargeDisposition, pointOfCare):
    self.caseID = caseID
    self.dob = dob
    self.gender = gender
    self.dateOfAdmission = dateOfAdmission
    self.dateOfDischarge = dateOfDischarge
    self.patientClass = patientClass
    self.statusID = statusID
    self.serviceLine = serviceLine
    self.payerID = payerID
    self.dischargeDisposition = dischargeDisposition
    self.pointOfCare = pointOfCare
    self.documents = self.fetchDocuments()
    self.age = int( ( self.dob - datetime.datetime.now() ).days / 365.25 )
    self.duration = int( ( self.dateOfDischarge - self.dateOfAdmission ).days )
    self.specialities = self.fetchSpecialities()


# In[ ]:


from PatientRecord import *
fileName = 'mGD'
fileObject = open('/content/gdrive/My Drive/' + fileName, 'rb')
db = pickle.load(fileObject)    
x = db['Patients']
fileObject.close()


# In[ ]:


doc = []
for i in range(len(x)):
  new = x[i].documentContentList
  for instance in new:
    doc.append(instance)


# In[154]:


len(doc)


# In[201]:


mapp[6]


# In[ ]:


d = 0
ax = []
for count in mapp:
  if(count == 0):
    pass
  else:
    for i in range(count):
      ax.append(d)
  d += 1


# In[208]:


len(ax)


# In[ ]:


for i in range(0):
  print('5')


# In[130]:


mapp


# In[ ]:


mapp = []
for i in range(len(x)):
  mapp.append(len(x[i].documentContentList))


# In[ ]:


embedd_df = pd.read_table('/content/gdrive/My Drive/Smart India Hackathon/THE DATASET/glove.6B.50d.txt', sep= ' ', index_col= 0,  header= None, quoting= csv.QUOTE_NONE)


# In[ ]:


embedd_df.drop(labels= 14798, inplace = True, axis = 0)
words_glove = embedd_df.index


# In[ ]:


def word_to_vec(word):
    return np.array(embedd_df.loc[word]).reshape((50, ))


# In[28]:


np.zeros((50, )).shape


# In[ ]:


docs = []


# In[ ]:


i = 0
vocab = {}
voc = []
for instance in doc:
  for word in instance:
     voc.append(word)
uni = sorted(set(voc))
for word in uni:
  vocab[word] = i
  i += 1


# In[ ]:


docs = []
for instance in doc:
  new = []
  for words in instance:
    new.append(words)
  docs.append(new)


# In[60]:


max(map(lambda x: len(x), docs))


# In[ ]:


PAD = '#PAD'


# In[ ]:


vocab[PAD] = 160535


# In[ ]:


X_ind = np.zeros((len(docs), 1000)) + vocab[PAD]


# In[ ]:


def sent_to_ind(sentence, vocab):
  ret = np.zeros((1, 1000)) + vocab[PAD]
  if(len(sentence) >= 1000):
    for i in range(1000):
      ret[:, i] = vocab[sentence[i]]
  else:
    for i in range(len(sentence)):
      ret[:,i] = vocab[sentence[i]]
    
  return ret


# In[ ]:


for i in range(len(X_ind)):
  X_ind[i,:] = sent_to_ind(docs[i], vocab)


# In[ ]:


from keras.models import Model
from keras.layers import Embedding
import keras
from keras import Input


# In[ ]:


from keras.layers import Dense


# In[ ]:


vocab_length = len(vocab)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


ss = StandardScaler()
ss.fit(X_ind)
X_ind_norm = ss.transform(X_ind)


# In[ ]:


def enc_dec(input_shape, vocab_length, emb_dim = 50, max_len = 1000):
  sentence_indices = Input(input_shape)
  encode = keras.layers.Dense(units = 950)(sentence_indices)
  encode = keras.layers.Dense(units = 900)(encode)
  decode = keras.layers.Dense(units = 950)(encode)
  decode = keras.layers.Dense(units = 1000)(decode)
  
  model = Model(input= sentence_indices, output = decode)
  encoder = Model(input = sentence_indices, output = encode)
  
  return encoder, model


# In[279]:


input_shape = (1000,)
encoder, model = enc_dec(input_shape, vocab_length)


# In[ ]:


model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])


# In[281]:


model.fit(X_ind_norm, X_ind_norm, batch_size = 512, epochs = 100, validation_split = 0.1)


# In[ ]:


encoded = encoder.predict(X_ind_norm)


# In[283]:


encoded.shape


# In[ ]:


data = pd.DataFrame(encoded)


# In[214]:


encoded.shape


# In[ ]:


data['CaseId'] = np.array(ax)


# In[286]:


data.head()


# ddd

# In[ ]:





# In[ ]:


new_data = data.groupby('CaseId').mean()
new_data['CaseId'] = new_data.index


# In[ ]:


new_data_val = new_data.values


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity


# In[318]:


encoded.shape


# In[319]:


enc.shape


# In[ ]:


case_id = new_data.CaseId


# In[ ]:


set_1 = set(range(enc.shape[0]))
set_2 = set(new_data.CaseId.unique())


# In[ ]:


a  = set_1 - set_2


# In[ ]:


a = list(a)


# In[ ]:


enc_removed = np.delete(enc, a, axis=0)


# In[345]:


enc_removed.shape


# In[ ]:


new_data_values = new_data.values


# In[ ]:


all_enc = np.concatenate((enc_removed, new_data_values), axis = 1)


# In[352]:


all_enc.shape


# In[ ]:


ss = StandardScaler()
ss.fit(all_enc)
all_enc_norm = ss.transform(all_enc)


# In[ ]:


def all_enc_dec(input_shape):
  
  input_ = Input(shape=(1413,))
  encoded = Dense(1054, activation='relu')(input_)
  encoded = Dense(1000, activation='relu')(encoded)
  encoded = Dense(900, activation='relu')(encoded)
  encoded = Dense(890, activation='sigmoid')(encoded)

  decoded = Dense(900, activation='relu')(encoded)
  decoded = Dense(1000, activation='relu')(decoded)
  decoded = Dense(1054, activation='relu')(decoded)
  decoded = Dense(1413)(decoded)

  autoencoder = Model(input_, decoded)
  encoder = Model(input_, encoded)
  
  return autoencoder, encoder


# In[ ]:


auto_perm_enc, all_enc_perm = all_enc_dec((1413,))


# In[ ]:


auto_perm_enc.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])


# In[407]:


auto_perm_enc.fit(all_enc_norm, all_enc_norm, epochs = 200, batch_size = 2048)


# In[ ]:


new_enc = all_enc_perm.predict(all_enc_norm)


# In[ ]:


sim = cosine_similarity(new_enc, new_enc)


# In[ ]:


similarity = pd.DataFrame(sim)


# In[ ]:


similarity.index = case_id


# In[417]:


similarity.head()


# In[ ]:


7691   7047


# In[ ]:


similarity.columns = case_id


# In[ ]:


eva = pd.read_csv('/content/gdrive/My Drive/Similarity_Evaluation.csv')


# In[ ]:


case1, case2 = eva['CaseId1'], eva['CaseId2']


# In[ ]:


sim  = similarity.loc[eva['CaseId1'], eva['CaseId2']]


# In[ ]:


ev = list([case1, case2])


# In[ ]:


a = eva[['CaseId1', 'CaseId2']].values


# In[ ]:


sim = []
for i,j in a:
  sim.append(similarity.loc[i, j]) 


# In[ ]:


eva['Similarity'] = np.array(sim)


# In[ ]:


eva.drop(labels= 'sim', axis = 1, inplace= True)


# In[ ]:


pd.set_option('display.max_rows', 300)


# In[449]:


eva


# In[ ]:




