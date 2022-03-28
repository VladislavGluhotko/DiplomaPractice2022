#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from datetime import datetime


# In[ ]:


spacelike_delimiters = [' ','\xa0']

def myLikesCountParser(stringg):
    for delimiter in spacelike_delimiters:
        stringg = stringg.replace(delimiter,' ').replace(',','.')
    num = float(stringg.partition(' ')[0])
    if "тыс" in stringg:
        num = num*10**3
    elif "млн" in stringg:
        num = num*10**6
    return int(num)


# In[3]:


df=pd.read_json('C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/putinspeech.json', lines=True, encoding="utf-8")
df.head(3)


# In[4]:


df = df.drop(['photo', 'channel', 'time'], axis=1)
df["time_parsed"] = df["time_parsed"].apply(lambda x: datetime.fromtimestamp(x))
df["votes"] = df["votes"].apply(lambda x: myLikesCountParser(x))

df['isBaseComment'] = False
df["RepliesCount"] = 0
df["RepliesID"] = [[] for i in range(0,df.count()[0])]


# In[5]:


commentsdict = dict()
n = df.count()[0]


# In[6]:


get_ipython().run_cell_magic('time', '', 'for index, row in df.iterrows():\n    temp = row["cid"].partition(\'.\')\n    \n    if temp[0] not in commentsdict:\n        commentsdict[temp[0]] = []\n        \n    if temp[2] == \'\': \n        df.loc[index,"isBaseComment"] = True\n    else:\n        commentsdict[temp[0]].append(index)')


# In[7]:


get_ipython().run_cell_magic('time', '', 'df.loc[df["isBaseComment"] == True, "RepliesID"] = df.loc[df["isBaseComment"] == True, "cid"].apply(lambda x: commentsdict[x])\ndf["RepliesCount"] = df["RepliesID"].apply(lambda x: len(x))')


# In[8]:


df = df.rename(columns={"author": "Nickname", "heart" : "LikedByAuthor", "text": "CommentText", "time_parsed": "Date", "votes" : "LikesCount"})
df = df.drop(["cid"], axis=1)


# In[9]:


df.head(10)


# In[10]:


df.to_csv(path_or_buf='C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/putinspeech.csv')


# In[ ]:




