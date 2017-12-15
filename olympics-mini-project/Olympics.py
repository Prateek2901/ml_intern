
# coding: utf-8

# In[1]:


import pandas as pd
import re


# In[2]:


df = pd.read_csv("./data/olympics.csv",header=1)


# In[3]:


df.head()


# In[4]:


df.columns


# In[5]:


df = df.rename(columns=lambda x: re.sub('^01 !','Gold',x))
df = df.rename(columns=lambda x: re.sub('^02 !','Silver',x))
df = df.rename(columns=lambda x: re.sub('^03 !','Bronze',x))


# In[6]:


df.head()


# In[7]:


country_Name,country_Code =[],[]
for i in df['Unnamed: 0']:
    start = i.find("(")
    end = i.find(")")
    #name = i[0:start]
    #print (name)
    country_Name.append(i[0:start])
    #code = i[start:end+1]
    #print (code)
    country_Code.append(i[start:end+1])


# In[8]:


for i,j in zip(country_Name,country_Code):
    print (i+"=>"+j)


# In[9]:


del df['Unnamed: 0']
df.insert(0,'country_Name',country_Name)
df.insert(1,'country_Code',country_Code)
df = df.set_index('country_Name')


# # DO:-
# * Split country name and country code and add country name as data frame index
# * Remove extra unnecessary characters from country name.

# In[10]:


df


# In[11]:


df.drop(df.tail(1).index,inplace=True)


# In[12]:


df # to be returned


# # Question 2:-

# In[13]:


df.iloc[0]


# # Question 3:-

# In[14]:


df.loc[df['Gold'].argmax()]


# # Question 4:-

# In[15]:


df.loc[(df['Gold'] - df['Gold.1']).abs().idxmax()]


# # Question 5:-

# In[16]:


points = []
for index, row in df.iterrows():
    pt = row['Gold.2']*3+row['Silver.2']*2+row['Bronze.2']
    points.append(pt)


# In[17]:


df['Points']=points


# In[18]:


df.head()


# # Question 6:-

# In[19]:


data = df[['# Games','Points']]


# In[20]:


data


# In[21]:


data.describe()


# In[22]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data,test_size=0.20, random_state=42)


# In[23]:


from sklearn.cluster import KMeans
import timeit
start = timeit.default_timer()
kmeans =  KMeans(n_clusters=3, random_state=0).fit(X_train)
print("Time take %.2f s"%((timeit.default_timer()-start)))


# In[24]:


print(kmeans.labels_)


# In[25]:


print(kmeans.predict(X_test))


# In[26]:


print(kmeans.cluster_centers_)


# In[33]:


from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
distortions = []

for i in range(1,10):
    start = timeit.default_timer()
    kmeans =  KMeans(n_clusters=i, random_state=0).fit(X_train)
    print("Time take for %d cluster is %.2f s"%(i,(timeit.default_timer()-start)))
    """
    print("Cluster Center\n")
    print(kmeans.cluster_centers_)
    print()
    """
    distortions.append(sum(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])


# In[34]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[35]:


plt.plot()
plt.scatter(data['# Games'],data['Points'])
plt.xlim([0, 53])
plt.title('Dataset')
plt.show()


# In[38]:


plt.plot([i for i in range(1,10)], distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('flay.png')
plt.grid()
plt.show()

