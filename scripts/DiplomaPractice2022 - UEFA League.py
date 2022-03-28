#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timezone


# In[2]:


def myToDateTime(datestr,yearr):
    dayy,monthnamee,_,time24hh = datestr.rsplit(' ',4)
    dayy = int(dayy)
    hourss,minutess = [int(x) for x in time24hh.rsplit(':',2)]
    secondss,milliseconds = [0,0]
    monthh = {
         monthnamee == "янв" : 1,
         monthnamee == "фев" : 2,
         monthnamee == "мар" : 3,
         monthnamee == "апр" : 4,
         monthnamee == "мая" : 5,
         monthnamee == "июн" : 6,
         monthnamee == "июл" : 7,
         monthnamee == "авг" : 8,
         monthnamee == "сен" : 9,
         monthnamee == "окт" : 10,
         monthnamee == "ноя" : 11,
         monthnamee == "дек" : 12
    }[True]
    
    tLOC = datetime(yearr, monthh, dayy, hourss, minutess, secondss, milliseconds)
    return tLOC


# In[3]:


with open('C:/Users/VG/Saved Games/Desktop/dippractice/data/sports/ru/uefaleaguevk.html', 'rb') as html:
    soup = BeautifulSoup(html, "html.parser")


# In[4]:


count = 0
df_comments=[]


# In[5]:


commentboxes=soup.find("div","replies_list _replies_list",recursive=True)
children=commentboxes.findChildren(recursive=False)


# In[ ]:


for box in commentboxes:
    if 'replies_wrap_deep' in box["class"]:
        isbasecomment = False
    else:
        isbasecomment = True
        
    replies = box.find_all("div","reply_content")
    
    for reply in replies:
        comment = reply.find("div","wall_reply_text")
        if comment == None:
            commenttext = "-"
        else:
            commenttext = comment.text
        
        nickname = reply.find("div","reply_author").find("a","author").text
        date = myToDateTime(reply.find("div","reply_date").find("span","rel_date").text,2022)
        
        likesbox = reply.find("div","like_button_count")
        if likesbox.text == '':
            likescount = 0
        else:
            likescount = int(likesbox.text)
        
        df_comments.append([count,commenttext,nickname,date,likescount,isbasecomment])
        count = count + 1


# In[ ]:


uefaleaguevkData=pd.DataFrame(df_comments,columns=["countID","CommentText","Nickname","Date","LikesCount","IsBaseComment"])
uefaleaguevkData


# In[ ]:


# True = 1 , False = 0
b = True
for index, row in uefaleaguevkData.iterrows():
    b = b * (index == row["countID"])

if b == True : 
    uefaleaguevkData = uefaleaguevkData.drop(["countID"], axis = 1) 
    print("id and countID match")


# In[ ]:


uefaleaguevkData.to_csv(path_or_buf='C:/Users/VG/Saved Games/Desktop/dippractice/data/sports/ru/uefaleaguevk.csv')


# In[8]:


for box in commentboxes:
    replies = box.find_all("div","reply_content")

    for reply in replies:
        print(reply.find("div","wall_reply_text").text)


# In[11]:





# In[ ]:




