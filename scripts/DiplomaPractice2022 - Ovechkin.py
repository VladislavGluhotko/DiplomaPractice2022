#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime


# In[2]:


with open('C:/Users/VG/Saved Games/Desktop/dippractice/data/sports/en/ovechkin.html', 'rb') as html:
    soup = BeautifulSoup(html)


# In[3]:


count = 0
df_comments=[]


# In[4]:


def MyBS4commentScrapper(bs4commentbox):  
    global count
    
    commenttext = bs4commentbox.find("div","HTMLContent-root-5770ce4668399900d87c06ad10ba71a5 htmlContent-root-fe3506b39eaecd1a685ceda5e1292c4d coral coral-content coral-comment-content").text

    nickdate = bs4commentbox.find("div","Box-root-acb8bd0953d79380f99868ba1e9c06f2 Flex-root-e30c492230d1edb681fe55ee4db5c12e Flex-flex-d2ca733e3f89034ac350e74ab398ca2c Flex-wrap-6d1b6ac029ed4bb936d308545730f2ca Flex-alignCenter-26c1ac1572ede070f23436e4a05f81bd")
    nick = bs4commentbox.find("span","Username-root-c00565d265d60c03a8fd24001a836d2b").text
    date = bs4commentbox.find("time","RelativeTime-root-d30b4f3b231d2f240c2f63990321529f Timestamp-text-b943991e0f8e51bb4bdba60eabe40535 Comment-timestamp-b603d74501b527b492dba3e2561137cb coral coral-timestamp coral-comment-timestamp")["datetime"]

    reactbox = bs4commentbox.find("div","Box-root-acb8bd0953d79380f99868ba1e9c06f2 Flex-root-e30c492230d1edb681fe55ee4db5c12e CommentContainer-actionBar-8f945503291a6fbc016fbde66cee912e Flex-flex-d2ca733e3f89034ac350e74ab398ca2c Flex-halfItemGutter-a202fcab1dcbec6c729d74b586859a0d Flex-justifySpaceBetween-1e1ecf12d82a80d36b234d1261a4a513 Flex-alignCenter-26c1ac1572ede070f23436e4a05f81bd Flex-directionRow-2454f15d2085c2be40d132bad0acd66d gutter")
    likesbox = reactbox.find("span","ReactionButton-totalReactions-5c75d792070e447916aec4f5cebb1972")

    if likesbox == None:
        likescount = 0
    else:
        likescount = int(likesbox.text)

    repliesbox = bs4commentbox.find("div","Box-root-acb8bd0953d79380f99868ba1e9c06f2 HorizontalGutter-root-42028c0a7886c844bb9f01763cc43000")
    
    
    if repliesbox == None:     
        repliesindexes = []
        repliescount = 0
    else:
        replies = repliesbox.find_all("div", "Box-root-acb8bd0953d79380f99868ba1e9c06f2 HorizontalGutter-root-42028c0a7886c844bb9f01763cc43000",recursive=False)
        repliescount = len(replies)
        repliesindexes = []
        
        for reply in replies:
            
            MyBS4commentScrapper(reply)
            repliesindexes.append(count-1)
            
        
    df_comments.append([count,commenttext,nick,date,likescount,False,repliescount,repliesindexes])   
    count = count + 1


# In[5]:


commentboxes = soup.find_all("div","Box-root-acb8bd0953d79380f99868ba1e9c06f2 HorizontalGutter-root-42028c0a7886c844bb9f01763cc43000 AllCommentsTabCommentContainer-borderedCommentNotSeen-dc042ede476ae67068a90f1b6f6c567e")
# "Box-root-acb8bd0953d79380f99868ba1e9c06f2 HorizontalGutter-root-42028c0a7886c844bb9f01763cc43000 HorizontalGutter-full-680598106a6954360bcd94b9d3839ca7"

for comment in commentboxes:
    isbasecomment = True
    MyBS4commentScrapper(comment)
    df_comments[count-1][5] = True


# In[6]:


ovechkinData=pd.DataFrame(df_comments,columns=["countID","CommentText","Nickname","Date","LikesCount","IsBaseComment","RepliesCount","RepliesID"])
ovechkinData["Date"] = ovechkinData["Date"].apply(lambda x: datetime.strptime(x,"%Y-%m-%dT%H:%M:%S.%fZ"))

ovechkinData


# In[7]:


# True = 1 , False = 0
b = True
for index, row in ovechkinData.iterrows():
    b = b * (index == row["countID"])

if b == True : 
    ovechkinData = ovechkinData.drop(["countID"], axis = 1) 
    print("id and countID match")


# In[9]:


ovechkinData.to_csv(path_or_buf='C:/Users/VG/Saved Games/Desktop/dippractice/data/sports/en/ovechkin.csv')

