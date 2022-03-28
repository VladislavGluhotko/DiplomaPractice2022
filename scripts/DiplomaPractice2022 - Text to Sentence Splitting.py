#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ! pip install sentence-splitter


# In[ ]:


import pandas as pd
import re
from sentence_splitter import SentenceSplitter, split_text_into_sentences


# In[ ]:


df = pd.read_csv('C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/immigrantsputin/immigrantsputin.csv')


# In[ ]:


for index,row in df.head(100).iterrows():
    print(index,row["CommentText"])


# In[ ]:


digits = "0123456789"
lettersRU = "йцукенгшщзхъфывапролджэячсмитьбюЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ"
lettersEN = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
delimiters = " ,.:;'!?()-/|\[]{}"
othersymbols = "+=<>@#&$%*~№"

mindigits_uni = min([ord(ch) for ch in digits])
maxdigits_uni = max([ord(ch) for ch in digits])
minlettersRU_uni = min([ord(ch) for ch in lettersRU])
maxlettersRU_uni = max([ord(ch) for ch in lettersRU])
minlettersEN_uni = min([ord(ch) for ch in lettersEN])
maxlettersEN_uni = max([ord(ch) for ch in lettersEN])
mindelimiters_uni = min([ord(ch) for ch in delimiters])
maxdelimiters_uni = max([ord(ch) for ch in delimiters])
minothersymbols_uni = min([ord(ch) for ch in othersymbols])
maxothersymbols_uni = max([ord(ch) for ch in othersymbols]) 


def myStringToPureText_RuEn(stringg):  
    res = ""
    for ch in stringg:
        if  mindigits_uni <= ord(ch) <= maxdigits_uni or minlettersRU_uni <= ord(ch) <= maxlettersRU_uni or minlettersEN_uni <= ord(ch) <= maxlettersEN_uni or mindelimiters_uni <= ord(ch) <= maxdelimiters_uni or minothersymbols_uni <= ord(ch) <= maxothersymbols_uni:
            res = res + ch
    return res


# In[ ]:


replacementlistRU = dict()
replacementlistRU['\n'] = '. '; replacementlistRU['\r'] = '';
replacementlistRU["(("] = '. '; replacementlistRU["))"] = '. '; 
replacementlistRU['+'] = " плюс "; replacementlistRU['='] = " равно ";  replacementlistRU['/'] = " или ";

replacementlistEN = dict()
replacementlistEN['\n'] = '. '; replacementlistEN['\r'] = '';
replacementlistEN['+'] = " plus "; replacementlistEN['='] = " equals ";  replacementlistEN['/'] = " or ";
replacementlistEN['&'] = " and ";

naivepattern = re.compile(r'([а-яa-zА-ЯA-Z](( т.п.| т.д.| пр. | г. | e.g. | Mr. | Mrs. | i.e. )|[^?!.\(]|\([^\)]*\))*[.?!])')

def myTextToSentencesSplitter(comtext, method = "naive", lang = "en"):
    res = []
    comtext = comtext + '.'
    
    if lang == "ru":
        for key in replacementlistRU.keys():
            comtext = comtext.replace(key,replacementlistRU[key])
    elif lang == "en":
        for key in replacementlistEN.keys():
            comtext = comtext.replace(key,replacementlistEN[key])
    
    # purify text from emoticons and non-text symbols
    comtext = myStringToPureText_RuEn(comtext)
    
    # method using regular expression
    if method ==  "naive":                 
        for index, sent in enumerate(naivepattern.findall(comtext)):
            res.append(sent[0])
    
    # method using algorithm by Philipp Koehn and Josh Schroeder
    elif method == "kohen-schroeder":
        splitter = SentenceSplitter(language = lang)
        res = splitter.split(text = comtext)
        
    return res 


# In[ ]:


# comtext = df["CommentText"][13]

# example - RU
comtext = "Храни вас Бог))) мирного + голубого неба всем! \n Слово слово слово... \n тртртрттр"

# example - EN
#comtext = "That's what you get \n haha why you threat? I'm typing/testing smth here! Day after day ... and here we go :)"
#comtext = "She is Mrs. Smith born in 1992. Here some fruits e.g. apples and oranges. Have to pay the bill i.e. 20 usd"

print(comtext)
print("----------------")
res = myTextToSentencesSplitter(comtext, method = "naive", lang = "ru")
for i in range(0,len(res)): 
    print(i, res[i])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df["CommentTextSplitNV"] = df["CommentText"].apply(lambda x: myTextToSentencesSplitter(x, method = "naive", lang = "ru"))\n#df["CommentTextSplitKS"] = df["CommentText"].apply(lambda x: myTextToSentencesSplitter(x, method = "kohen-schroeder", lang = "en"))')


# In[ ]:


df.head(3)


# In[ ]:


df.to_pickle('C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/immigrantsputin/immigrantsputin_dataset.pkl')


# In[ ]:




