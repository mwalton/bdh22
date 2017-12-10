
# coding: utf-8

# ### TODO:
# Girish
# - ~~Dataset construction~~
#
# Mike
# - ~~implement build_sequences & build_skipgrams~~
# - ~~softmax word embeddings~~
# - ~~implement to_onehot_tensor~~
# - ~~seq2seq autoencoder~~
#
# Guowei
# - clustering & metrics
# - low dimensional PCA / MDN / TSNE visualizations

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score


# In[114]:

print("create chunks")
chunks = pd.read_table('../data/df_features_final.tsv', sep='|', chunksize=1000)
print("concat chunks")
data = pd.concat(chunk for chunk in chunks)
subj = 'df_features_final.subject_id'
hadm = 'df_features_final.hadm_id'
service = 'df_features_final.service'
exp = 'df_features_final.expire_flag'
seq_cnt = 'df_features_final.icd_seq_cnt'
seq = 'df_features_final.icd_seq_str'
data.reset_index(inplace=True)
data.set_index([subj, hadm, service], inplace=True)
sequences = data[seq].sample(1000)
print("loaded %d rows" % len(data))
