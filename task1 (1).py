#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#Clustering Model Library
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

import os


# In[2]:


macd= pd.read_csv("C:\\Users\\SOURAV KUMAR\\Downloads\\McDonalds Case Study-20230624T095606Z-001\\McDonalds Case Study\\mcdonalds.csv")
macd.head()


# In[3]:


macd.info()


# In[4]:


macd.dtypes


# In[5]:


print(pd.isnull(macd).sum())


# In[6]:


macd.describe()


# In[7]:


macd['yummy'].value_counts()


# In[8]:


macd['VisitFrequency'].value_counts()


# In[9]:


macd['Like'].value_counts()


# In[10]:


macd['convenient'].value_counts()


# In[11]:


macd['Age'].value_counts()


# In[12]:


macd_data=macd.iloc[:,0:11].values
macd_data=(macd_data == "Yes").astype(int)
col_averages=np.round(np.mean(macd_data,axis=0),2)
print(col_averages)


# In[13]:


import matplotlib.pyplot as plt

labels=['Male','Female']
sizes=[macd.query('Gender=="Male"').Gender.count(),macd.query('Gender=="Female"').Gender.count()]
plt.figure(figsize=(4,4))
plt.pie(sizes,labels=labels)
plt.show()


# In[14]:


macd['Like']=macd['Like'].replace({'I hate it!-5':'-5','I Love it!+5':'+5'})
macd.head(10)


# In[15]:


from sklearn.preprocessing import LabelEncoder
def labelling(x):
    macd[x]=LabelEncoder().fit_transform(macd[x])
    return macd
labl=['yummy','convenient','spicy','fattening','greasy','fast','cheap','tasty','expensive','healthy','disgusting']
for i in labl:
    labelling(i)
                         
macd


# In[16]:


plt.rcParams['figure.figsize']=(12,14)
macd.hist()
plt.show


# In[17]:


macd1=macd.loc[:,labl]
macd1


# In[18]:


ar=macd.loc[:,labl].values
ar


# In[19]:


from sklearn.decomposition import PCA
from sklearn import preprocessing

pca_data=preprocessing.scale(ar)

pca=PCA(n_components=11)
pc=pca.fit_transform(ar)
names=['pc1','pc2','pc3','pc4','pc5','pc6','pc7','pc8','pc9','pc10','p11']
pf= pd.DataFrame(data =pc,columns=names)
pf


# In[20]:


pca.explained_variance_ratio_


# In[21]:


np.cumsum(pca.explained_variance_ratio_)


# In[22]:


pca=PCA()
pca.fit(macd1)
loadings=pca.components_
num_pc=pca.components_.shape[0]
pc_list = ["PC" + str(i) for i in range(1 , num_pc+1)]
loadings_df=pd.DataFrame(loadings.T,columns=pc_list)
loadings_df['variable']=macd1.columns.values
loadings_df = loadings_df.set_index('variable')
loadings_df


# In[23]:


plt.rcParams['figure.figsize']=(20,15)
ax=sns.heatmap(loadings_df,annot=True)
plt.show()


# In[24]:


get_ipython().system('pip install bioinfokit')
get_ipython().system('pip install yellowbrick')
from bioinfokit.visuz import cluster
cluster.screeplot(obj=[pc_list,pca.explained_variance_ratio_],show=True,dim=(10,5))


# In[25]:


pca_scores=PCA().fit_transform(ar)
cluster.biplot(cscore=pca_scores,loadings=loadings,labels=macd.columns.values,var1=round(pca.explained_variance_ratio_[0]*100,2),
               var2=round(pca.explained_variance_ratio_[1]*100,2),show=True,dim=(10,5))


# In[26]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model =KMeans()
visualizer=KElbowVisualizer(model,k=(1,12)).fit(macd1)
visualizer.show()


# In[27]:


kmeans=KMeans(n_clusters=4,init='k-means++',n_init=10,random_state=0).fit(macd1)
macd['cluster_num']=kmeans.labels_
print(kmeans.labels_)
print(kmeans.inertia_)
print(kmeans.n_iter_)
print(kmeans.cluster_centers_)


# In[28]:


from collections import Counter
Counter(kmeans.labels_)


# In[29]:


sns.scatterplot(data=pf,x="pc1",y="pc2",hue=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
           marker="X",c="r",s=80,label="centroids")
plt.legend()
plt.show()


# In[30]:


from statsmodels.graphics.mosaicplot import mosaic
from itertools import product
crosstab=pd.crosstab(macd['cluster_num'],macd["Like"])
crosstab=crosstab[['-5','-4','-3','-2','-1','0','+1','+2','+3','+4']]
crosstab


# In[31]:


crosstab_gender=pd.crosstab(macd['cluster_num'],macd['Gender'])
crosstab_gender


# In[32]:


plt.rcParams['figure.figsize']=(7,5)
mosaic(crosstab_gender.stack())
plt.show()


# In[33]:


sns.boxplot(x="cluster_num",y="Age",data=macd)


# In[34]:


macd['VisitFrequency']=LabelEncoder().fit_transform(macd['VisitFrequency'])
visit=macd.groupby('cluster_num')['VisitFrequency'].mean()
visit=visit.to_frame().reset_index()
visit


# In[35]:


macd['Like']=LabelEncoder().fit_transform(macd['Like'])
Like=macd.groupby('cluster_num')['Like'].mean()
Like=Like.to_frame().reset_index()
Like


# In[36]:


macd['Gender']=LabelEncoder().fit_transform(macd['Gender'])
Gender=macd.groupby('cluster_num')['Gender'].mean()
Gender=Gender.to_frame().reset_index()
Gender


# In[37]:


segment=Gender.merge(Like,on='cluster_num',how='left').merge(visit,on='cluster_num',how='left')
segment


# In[38]:


plt.figure(figsize=(9,4))
sns.scatterplot(x="VisitFrequency",y="Like",data=segment,s=400,color="r")
plt.title("Simple segement evaluation plot for the fast food dataset",
         fontsize=18)
plt.xlabel("Visit",fontsize=12)
plt.ylabel("Like",fontsize=12)
plt.show()


# In[ ]:




