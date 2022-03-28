#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
from datetime import datetime, timezone
import string


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


# In[ ]:


def myStringToPureText_RuEn(stringg):
    res = ""
    for ch in stringg:
        if  mindigits_uni <= ord(ch) <= maxdigits_uni or minlettersRU_uni <= ord(ch) <= maxlettersRU_uni or minlettersEN_uni <= ord(ch) <= maxlettersEN_uni or mindelimiters_uni <= ord(ch) <= maxdelimiters_uni or minothersymbols_uni <= ord(ch) <= maxothersymbols_uni:
            res = res + ch
    return res


# In[ ]:


def removeBlankSpaces(stringg):
    i = 0;  j = 0;
    n = len(stringg)
    res = ""
    
    while i < n:
        if stringg[i] == " ":
            j = i + 1
            while j < n:
                if stringg[j] == " ":
                    j = j + 1
                else:
                    break
            i = j - 1
        res = res + stringg[i]
        i = i + 1
        
    return res


# In[ ]:


def myToDateTime(datestr,yearr):
    temp = datestr.rsplit(' ',4)
    temp.remove(''); temp.remove('') 
    dayy,monthnamee,time24hh = temp
    dayy = int(dayy)
    hourss,minutess = [int(x) for x in time24hh.rsplit(':',2)]
    secondss,milliseconds = [0,0]
    monthh = {
         monthnamee == "января" : 1,
         monthnamee == "февраля" : 2,
         monthnamee == "марта" : 3,
         monthnamee == "апреля" : 4,
         monthnamee == "мая" : 5,
         monthnamee == "июня" : 6,
         monthnamee == "июля" : 7,
         monthnamee == "августа" : 8,
         monthnamee == "сентября" : 9,
         monthnamee == "октября" : 10,
         monthnamee == "ноября" : 11,
         monthnamee == "декабря" : 12
    }[True]
    
    tLOC = datetime(yearr, monthh, dayy, hourss, minutess, secondss, milliseconds)
    return tLOC


# In[ ]:


with open('C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/immigrantsputin/immigrantsputin.html', 'rb') as html:
    soup = BeautifulSoup(html,"html.parser")


# In[ ]:


count = 0
df_comments = []


# In[ ]:


commentboxes = soup.find_all("div","chat__lenta-item popper-wrapper ")


# In[ ]:


for box in commentboxes:
    commenttext = box.find("div","chat__lenta-item-message-text").text
    commenttext = removeBlankSpaces(myStringToPureText_RuEn(commenttext))
    
    datetxt = box.find("div","chat__lenta-item-date").text
    datetxt = datetxt.replace(",","")
    datetxt = removeBlankSpaces(myStringToPureText_RuEn(datetxt))
    date = myToDateTime(datetxt,2022)
    
    nicktxt = box.find("a","chat__lenta-item-name-text").text
    nickname = removeBlankSpaces(myStringToPureText_RuEn(nicktxt))
    
    likestxt = box.find("div","chat__lenta-item-likes").text
    likescount = int(removeBlankSpaces(myStringToPureText_RuEn(likestxt)))
    
    isbasecomment = box.find("div","chat__lenta-quote") == None
    
    df_comments.append([count,commenttext,nickname,date,likescount,isbasecomment])
    count = count + 1


# In[ ]:


immigrantsPutinData=pd.DataFrame(df_comments,columns=["countID","CommentText","Nickname","Date","LikesCount","IsBaseComment"])
immigrantsPutinData


# In[ ]:


# True = 1 , False = 0
b = True
for index, row in immigrantsPutinData.iterrows():
    b = b * (index == row["countID"])

if b == True : 
    immigrantsPutinData = immigrantsPutinData.drop(["countID"], axis = 1) 
    print("id and countID match")


# In[ ]:


immigrantsPutinData.to_csv(path_or_buf='C:/Users/VG/Saved Games/Desktop/dippractice/data/politics/ru/immigrantsputin.csv')

