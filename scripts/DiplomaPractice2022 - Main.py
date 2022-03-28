#!/usr/bin/env python
# coding: utf-8

# ## Расположение проекта:

# In[1]:


proj_path = "C:\\Users\\VG\\Saved Games\\Desktop\\dippractice\\"
proj_path_b = proj_path.replace('\\','/')


# ## Проверка используемого окружения

# In[ ]:


# Текущее окружение: 

import pkg_resources
thisenv = []
for pkg in pkg_resources.Environment().__iter__():
    thisenv.append(pkg + "==" + pkg_resources.get_distribution(pkg).version)


# In[ ]:


# Окружение совпадает с requirement-summarization ?

reqenv = []
with open(proj_path + "requirements_summarization.txt") as f:
    lines = f.readlines()
    for l in lines:
        reqenv.append(l.lower().replace('\n',''))
    
print("Лишние: \n")
for pkg in thisenv:
    if pkg not in reqenv:
        print(pkg)
print("-----------")

print("Недостающие: \n")
for pkg in reqenv:
    if pkg not in thisenv:
        print(pkg)
print("-----------")


# ## Необходимые библиотеки:

# In[2]:


import os
import enum
import math
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as text
from sentence_transformers import SentenceTransformer
import umap
from sklearn import decomposition

#from sklearn.preprocessing import StandardScaler


# ## Подготовка датасета
# Преобразование исходных танных в виде нескольких таблиц dataframe так, чтобы одно предложение = один сэмпл
# 

# In[22]:


dataset_name = "putinspeech"
dataset_path = "data\\politics\\ru\\putinspeech\\"
dataset_path_b = dataset_path.replace('\\','/')

df_commentsinfo = pd.read_pickle(proj_path_b + dataset_path_b + "putinspeech_dataset.pkl")
df_commentsinfo.head(3)


# In[4]:


class TextToSentenceSplitMethods(enum.Enum):
    Naive = "naive"
    KohenSchroeder = "kohenschroeder"
    
class SplittedSentences:
    def __init__(self, dfsentences, methodd):
        self.sentences = dfsentences
        self.method = methodd


# In[23]:


get_ipython().run_cell_magic('time', '', '# Определенные методом Naive предложения:\nif "CommentTextSplitNV" in df_commentsinfo.columns:        \n    dflist=[]\n    for index, row in df_commentsinfo.iterrows():\n        for sentence in row["CommentTextSplitNV"]:\n            dflist.append(pd.DataFrame([[index,sentence]],columns=["CommentID","CommentSentence"]))\n\n    df_commentstext_NV = pd.concat(dflist,ignore_index=True) \n    commentsNV = SplittedSentences(df_commentstext_NV, TextToSentenceSplitMethods.Naive)\n    del df_commentstext_NV')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# Определенные методом Kohen-Schroeder предложения:\n#  - работает только если комментарии на английском языке -\nif "CommentTextSplitKS" in df_commentsinfo.columns:        \n    dflist=[]\n    for index, row in df_commentsinfo.iterrows():\n        for sentence in row["CommentTextSplitKS"]:\n            dflist.append(pd.DataFrame([[index,sentence]],columns=["CommentID","CommentSentence"]))\n\n    df_commentstext_KS = pd.concat(dflist,ignore_index=True)\n    commentsKS = SplittedSentences(df_commentstext_KS, TextToSentenceSplitMethods.KohenSchroeder)\n    del df_commentstext_KS')


# In[24]:


commentsNV.sentences.head(10)


# In[25]:


print(commentsNV.sentences.count())
print(commentsKS.sentences.count())


# ## Sentence Encoders

# In[7]:


class SentenceEncodersTypes(enum.Enum):
    SBERT = "SBERT"
    CMLM = "CMLM"
    LaBSE = "LaBSE"
    
class SentencePreprocessorsTypes(enum.Enum):
    BERT = "BERT"
    CMLM = "CMLM"
    
class SentenceEncoder:
    def __init__(self, encoderr, enctypee, modelnamee):
        self.encoder = encoderr
        self.enctype = enctypee
        self.modelname = modelnamee

class SentencePreprocessor:
    def __init__(self, preprocessorr, preprocessortypee, modelnamee):
        self.preprocessor = preprocessorr
        self.preprocessortype = preprocessortypee
        self.modelname = modelnamee


# In[8]:


sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']


# SBERT

# In[ ]:


ParaphraseMiniLML6v2_encoder = SentenceTransformer(proj_path + "stmodels\\SBERT\\models\\paraphraseMiniLML6v2\\model\\")
ParaphraseMiniLML6v2 = SentenceEncoder(ParaphraseMiniLML6v2_encoder, SentenceEncodersTypes.SBERT, "ParaphraseMiniLML6v2")

tf.constant(ParaphraseMiniLML6v2.encoder.encode(sentences))


# In[9]:


from transformers import AutoModel, AutoTokenizer
RuBERT180M_model = AutoModel.from_pretrained(proj_path + "stmodels\\SBERT\\models\\RuBERT180M\\model\\")
RuBERT180M_tokenizer = AutoTokenizer.from_pretrained(proj_path + "stmodels\\SBERT\\models\\RuBERT180M\\tokenizer\\")

class RuBERT180MEncoder :
    def __init__(self, modell, tokenizerr):
        self.model = modell
        self.tokenizer = tokenizerr
        
    def encode(self, sentencess):     
        arrs_to_concat = []

        for sent in sentencess:
            temp = self.tokenizer(sent, return_tensors="pt")
            for key in temp:
                if temp[key].size(dim=1) > 512:
                    temp[key] = temp[key].resize_(1,512)
            embedding = self.model(**temp, output_hidden_states=False).pooler_output
            arrs_to_concat.append(embedding.detach().numpy())
        
        return np.concatenate(arrs_to_concat, axis=0)

RuBERT180M_encoder = RuBERT180MEncoder(RuBERT180M_model,RuBERT180M_tokenizer)
RuBERT180M = SentenceEncoder(RuBERT180M_encoder, SentenceEncodersTypes.SBERT, "RuBERT180M")

tf.constant(RuBERT180M.encoder.encode(sentences))


# CMLM (1 GB)

# In[10]:


SentEvalV1_encoder = tf.keras.models.load_model(proj_path + "stmodels\\CMLM\\CMLMsenteval\\models\\CMLMsentevalV1\\model\\")
SentEvalV1 = SentenceEncoder(SentEvalV1_encoder, SentenceEncodersTypes.CMLM, "SentEvalV1")

ENuncasedV3_preprocessor = tf.keras.models.load_model(proj_path + "stmodels\\preprocessors\\BERTenUncasedV3\\model\\")
ENuncasedV3 = SentencePreprocessor(ENuncasedV3_preprocessor, SentencePreprocessorsTypes.BERT, "ENuncasedV3")

SentEvalV1.encoder(ENuncasedV3.preprocessor(tf.constant(sentences)))["default"]


# LaBSE (3 GB)

# In[11]:


LaBSEV2_encoder=tf.keras.models.load_model(proj_path + "stmodels\\LaBSE\\models\\LaBSEV2\\model\\")
LaBSEV2 = SentenceEncoder(LaBSEV2_encoder, SentenceEncodersTypes.LaBSE, "LaBSEV2")

MultilingualV2_preprocessor = tf.keras.models.load_model(proj_path + "stmodels\\preprocessors\\CMLMmultilingualV2\\model\\")
MultilingualV2 = SentencePreprocessor(MultilingualV2_preprocessor, SentencePreprocessorsTypes.CMLM, "MultilingualV2")

LaBSEV2.encoder(MultilingualV2.preprocessor(tf.constant(sentences)))["default"]


# In[ ]:


# Когда я скачиваю LaBSE-V2 encoder вручную с сайта, то
# при попытке чтения выдает ошибку KeyError: 'name' 
# Обходной путь:
# Этот скрипт скачивает LaBSE-V2 encoder как hub.KerasLayer и создает
# на его основе простую tf.keras.Model чтобы его обернуть и затем сохранить в файл, я
# был вынужден использовать Model поскольку нет встроенной функции для сохранения самого hub.KerasLayer 
#-------------------------------------------------------------------------------------------------------#

# import tensorflow_hub as hub
# LaBSEV2_encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2", trainable=False)

# см. TF Functional API

# inp1 = tf.keras.Input(shape=(None,), dtype=tf.dtypes.int32, name='input_mask')
# inp2 = tf.keras.Input(shape=(None,), dtype=tf.dtypes.int32, name='input_word_ids')
# inp3 = tf.keras.Input(shape=(None,), dtype=tf.dtypes.int32, name='input_type_ids')
# 
# LaBSEV2_encoder({'input_mask': inp1, 'input_word_ids': inp2, 'input_type_ids': inp3}, False)
# model = tf.keras.Model(inputs=[inp1, inp2, inp3], outputs=LaBSEV2_encoder.output)
# model.save(proj_path + "stmodels\\LaBSE\\models\\LaBSEV2\\model\\")


# Векторизация предложений и запись в соответствующую директорию

# In[12]:


def myGetBatchEmbeddings(dfsentences_batch, encoderr, preprocessorr = None):
    embeddings = tf.constant([])
    if encoderr.enctype == SentenceEncodersTypes.SBERT:
        sentences = list(dfsentences_batch.values)
        embeddings = tf.constant(encoderr.encoder.encode(sentences))   # hahaha
            
    if encoderr.enctype == SentenceEncodersTypes.CMLM or encoderr.enctype == SentenceEncodersTypes.LaBSE:        
        if preprocessorr == None:
            raise KeyError("Preprocessor is required for encoder type : " + encoderr.enctype.value + ", please pass parameter: preprocessorr = <class: SentencePreprocessor>.")
        else:
            sentences = tf.constant(dfsentences_batch.values)
            embeddings = encoderr.encoder(preprocessorr.preprocessor(tf.constant(sentences)))["default"]
            
    return embeddings.numpy()


# In[13]:


def mySaveEmbeddings(splittedsentences, encoder, preprocessor = None, usebatches = True, batch_size = 20):
    
    if usebatches == True:
        prep_modelname = ''
        if preprocessor != None:
            prep_modelname = preprocessor.modelname 
        
        embedd_dir = proj_path + dataset_path + "embeddings\\" + splittedsentences.method.value + '\\' + encoder.modelname + prep_modelname + '\\'
        batches_dir = embedd_dir + "batches\\"
        
                   
        embedd_name = dataset_name + '_' + splittedsentences.method.value + '_' + encoder.modelname + prep_modelname
        
        if os.path.isdir(batches_dir) == False:
            print("Creating directory: " +  batches_dir)
            os.makedirs(batches_dir)
        else:
            print("Clearing directory before loading: " + batches_dir)
            for filename in os.listdir(batches_dir):
                temp = batches_dir + filename
                if os.path.isfile(temp) or os.path.islink(temp):
                        os.unlink(temp)
            print("Done")
        
        ind_from = 0
        ind_to = 0
        l = splittedsentences.sentences.shape[0]

        print("Loading batches to: " + batches_dir)
        batchh_id = 0
        padzeroleft_batchhid = int(math.log10(int(l/batch_size + 1))) + 1
        while ind_to != l:
            ind_to = ind_from + batch_size
            if ind_to > l:
                ind_to = l

            batchh = splittedsentences.sentences.loc[splittedsentences.sentences.index[ind_from : ind_to], "CommentSentence"]
            batch_name = embedd_name + '_' + "batchid" + str(batchh_id).zfill(padzeroleft_batchhid) + '_' + str(ind_from) + '_' + str(ind_to) + ".npy"
            ind_from = ind_to
            batchh_id = batchh_id + 1
            
            batchh_embedd = myGetBatchEmbeddings(batchh, encoder, preprocessorr=preprocessor)
            np.save(batches_dir + batch_name, batchh_embedd)
            print("Done: " + str(ind_from) + " out of " + str(l) + "  " + str(int(ind_from/l*100)) + '%')
        print("Done")
        
        print("Reading and concatenating batches: ")
        arrs_to_concat = []
        for filename in os.listdir(batches_dir):
            temp = batches_dir + filename
            arrs_to_concat.append(np.load(temp))
        
        res = np.concatenate(arrs_to_concat, axis = 0)
        print("Done")
              
        np.save(embedd_dir + embedd_name + ".npy",res)
        print("Result saved as: " + embedd_dir + embedd_name + ".npy")
        
    else:
        raise KeyError("usebatches = False NOT IMPLEMENTED" )


# In[26]:


mySaveEmbeddings(commentsNV,RuBERT180M,preprocessor=None,usebatches=True,batch_size=20)
mySaveEmbeddings(commentsNV,LaBSEV2,preprocessor=MultilingualV2,usebatches=True,batch_size=20)
mySaveEmbeddings(commentsNV,SentEvalV1,preprocessor=ENuncasedV3,usebatches=True,batch_size=20)


# ## Data Dimensionality Reducers

# UMAP

# In[ ]:


UMAP_reducer = umap.UMAP(a=None, angular_rp_forest=False, b=None,
     force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
     local_connectivity=1.0, low_memory=False, metric='euclidean',
     metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
     n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
     output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
     set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
     target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
     transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)


# PCA

# In[ ]:


PCA_reducer = decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=42)


# In[ ]:




