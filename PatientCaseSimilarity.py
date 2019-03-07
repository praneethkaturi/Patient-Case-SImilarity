#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import glob, os
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import xmltodict
import pickle
ps = PorterStemmer()
lm = WordNetLemmatizer()

dbfile = open('globalDump', 'rb')      
db = pickle.load(dbfile)
dout = {}
for keys in db:
    dout[keys] = db[keys]


# In[6]:


import xml.etree.ElementTree as ET


def getLevel( root ):
    if len( root ) == 0:
        return set()
    else:
        ret = set()
        for child in root:
            if 'certainty' in child.attrib and int(child.attrib['certainty']) >= 0:
                ret.add( child.attrib['value'].lower().strip() )
                if 'cui' in child.attrib:
                    for ele in child.attrib['cui'].split(";"):
                        ret.add( ele.lower().strip() )
            ans = getLevel( child )
            ret = ret.union( ans )
        return ret



Patients = dout['Patients']

for i in range(len(Patients)):
    base_path = ".'/"
    pat = Patients[i]
    pat.documentContentList = []
    if not os.path.isdir( base_path ):
        continue
        
    for item in pat.documents:
        try:
            tree = ET.parse( base_path + item['DocName'] + ".xml" )
            root = tree.getroot()
            ans = getLevel( root )
            pat.documentContentList.append( ans )
        except:
            continue
Patients[0].documentContentList


# In[1]:


import pickle
from PatientRecord import *
dbf = open("mGD","rb")
db = pickle.load(dbf)
dbf.close()


# In[2]:


Patients = db['Patients']
corpus = [['']]*( len(Patients) )
for i in range(len(Patients)):
    if len( Patients[i].documentContentList ) > 0:
        ss = Patients[i].documentContentList[0]
        for j in range(1,len(Patients[i].documentContentList)):
            for ele in Patients[i].documentContentList[j]:
                ss.add( ele )
        corpus[i] = list( ss )
corpus[0]


# In[3]:


import numpy as np
from gensim import corpora
from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.similarities import MatrixSimilarity
               
sims = {'files': {}}
dictionary = corpora.Dictionary(corpus)
corpus_gensim = [dictionary.doc2bow(doc) for doc in corpus]
tfidf = TfidfModel(corpus_gensim)
corpus_tfidf = tfidf[corpus_gensim]


# In[4]:


import pandas as pd
dat2 = pd.read_csv("./RG2/Case_Documents.csv")
for index,row in dat2.iterrows():
    corpus[ row['CaseId'] ].append( row['DocType'].lower().strip() )
corpus[5795]


# In[5]:


data = pd.read_csv('Demo_processed.csv')
from sklearn import preprocessing
import pandas as pd
cols = list(data.columns.values)[1:3]
x = data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
data = data.drop( columns=[0] )


# In[6]:


for i in range(len(cols)):
    dictionary.add_documents([ [cols[i]] ])
    curr_id = dictionary.doc2idx([ cols[i] ])[0]
    print(i)
    for j in range( len(Patients) ):
        corpus_tfidf[j].append( ( curr_id , data.at[ j , i+1 ] ) )


# In[7]:


lsi = LsiModel(corpus_tfidf, id2word=dictionary)
lsi_index = MatrixSimilarity(lsi[corpus_tfidf])
sims['files']['LSI'] = np.array([lsi_index[lsi[corpus_tfidf[i]]] for i in range(len(corpus))])


# In[8]:


sims


# In[9]:


ind=np.unravel_index(np.argmax(sims['files']['LSI'],axis=None), sims['files']['LSI'].shape)


# In[10]:


k = sims['files']['LSI']


# In[11]:


k[6153][857]


# In[12]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(k)
k2 = scaler.transform( k )
k2


# In[13]:


k2[6153][857]


# In[14]:


k2[56][3573]


# In[15]:


k2[1][175]


# In[16]:


k2[47][7691]


# In[17]:


k2[34][2374]


# In[18]:


k2[6153][5483]


# In[110]:


data.head()


# In[26]:


import pandas as pd
result = pd.read_csv("Similarity_Evaluation.csv")
res_ls = []
for index,row in result.iterrows():
    res_ls.append( k2[ int(row['CaseId1']) ][ int(row['CaseId2']) ] )
result.drop( columns=['Similarity'] )
result['Similarity'] = res_ls
result.to_csv("Similarity_Evaluation_.csv")
result


# In[ ]:




