#!/usr/bin/env python
# coding: utf-8

# In[1]:



import os 
import pandas as pd
address1 = 'C://Users//KIRAN KONDISETTI//Desktop//CSCE Assignment - C//dataset//programsnames.txt' #Address of the programnames.txt
programsnames= pd.read_table(  address1 , delim_whitespace=True, names=('n'))
os.chdir('C://Users//KIRAN KONDISETTI//Desktop//CSCE Assignment - C//dataset')  #Change the directory to the directory of the datasets 
data = { } 
df = pd.DataFrame(data) 
for i in range(0,39):       
    z = programsnames.n[i]
    address1 =  z
    for filename in os.listdir(address1):
       
           
        if filename.endswith(".txt"):
            k = os.path.join(address1, filename)
            l = pd.read_table(  k , delim_whitespace=True, names=('name','value2','1','2','3','4','5'))
            df = pd.concat([df, l], axis=0)
file =  df.drop(['1','2', '3', '4', '5'] , axis = 1)
file = file.fillna(0.001)
address2 = 'C://Users//KIRAN KONDISETTI//Desktop//CSCE Assignment - C//dataset//features_65.txt' #input the address of the features_65.txt file
features = pd.read_table(  address2 , delim_whitespace=True, names=('n'))
features.drop_duplicates()
f1 = features.drop_duplicates()
data = { } 
df2 = pd.DataFrame(data)
df3 = pd.DataFrame(data)
for i in range(0,63):    #set the range from 0 to number of rows in f1  
    x = features.n[i]
    all_stat = file.loc[file['name']== x, :]
    t = all_stat.astype({'value2': 'int64'})
    maximum = t['value2'].max()
    
    normalized = t['value2'].div(maximum).round(4)
    
    
    df2 = pd.concat([df2, all_stat], axis=0)
    df3 =  pd.concat([df3, normalized], axis=0,)
    x = features.n[i]
    with open('C://Users//KIRAN KONDISETTI//Desktop//max.txt', 'a') as fo:
            fo.write(x+',')
            fo.write('%d' % maximum)
            fo.write('\n')
    all_stat = df2.loc[df2['name']== x, :]
    o = all_stat['value2'].tolist()
    with open('C://Users//KIRAN KONDISETTI//Desktop//allstats.txt', 'a') as fo:
        fo.write('\n')
        fo.write(x+',')
        for d in o:
            fo.write(str(d) + ',')
    s= normalized.tolist()
    with open('C://Users//KIRAN KONDISETTI//Desktop//allstats_norm.txt', 'a') as fo:
        fo.write('\n')
        fo.write(x+',')
        for d in s:
            fo.write(str(d) + ',')
    
df3.columns = ['value2']            
v = df3.reset_index()
for i in range(0,39):     #set the range from 0 to number of program files inputed 
    l = programsnames.n[i]
    with open('C://Users//KIRAN KONDISETTI//Desktop//data.txt', 'a') as fo:
        fo.write('\n')
        fo.write(l+','+'\n')
    for c in range(0,63): #set the range from 0 to number of rows in f1  
        g = (39*c) + i
        
        f = v.value2[g]
        with open('C://Users//KIRAN KONDISETTI//Desktop//data.txt', 'a') as fo:
            fo.write('\t'+ '%f' %f)
        
                    
            
                    


# In[14]:


f1


# In[ ]:




