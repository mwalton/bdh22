import psycopg2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, LSTM, RepeatVector, Dense
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from scipy.stats import itemfreq
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import scipy
import pylab
import scipy.cluster.hierarchy as sch

plt.style.use('ggplot')

sqluser = 'mwalton'
dbname = 'mimic'
schema_name = 'mimiciii'

connection = psycopg2.connect(dbname=dbname, user=sqluser)
cursor = connection.cursor()
cursor.execute('SET search_path to {}'.format(schema_name))

n = 1000000

query = '''SELECT subject_id, hadm_id, itemid, charttime, value, flag
        FROM labevents
        WHERE hadm_id IS NOT NULL
        LIMIT {}'''.format(n)

data = pd.read_sql_query(query, connection)

print "Columns: {}".format(data.head().columns.tolist())
print "Unique Hospital stays: {}\nUnique Subjects: {}\nUnique Lab Events:{}".format(len(data['hadm_id'].unique()),
                                                                                    len(data['subject_id'].unique()),
                                                                                    len(data['itemid'].unique()))

print "Max / Min Chart time: {} / {}".format(data['charttime'].max(), data['charttime'].min())

data_grouped = data.sort(['charttime'])[['hadm_id','itemid']].groupby(['hadm_id'])
labevents = list(data_grouped['itemid'].apply(pd.Series.as_matrix))
#tokenizer = Tokenizer(num_words=len(np.unique(np.concatenate(labevents))))
#tokenizer.fit_on_sequences(labevents)
#X = tokenizer.sequences_to_matrix(labevents)

hist = itemfreq(np.concatenate(labevents))
plt.bar(hist[:,0],hist[:,1])
plt.show()

# TFIDF w/ MDS
tfidf = TfidfVectorizer().fit_transform([str(v)[1:-1] for v in labevents])
delta = 1. - cosine_similarity(tfidf)
plt.matshow(delta)
plt.show()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=666)
loc = mds.fit_transform(delta)
plt.scatter(loc[:, 0], loc[:, 1])
plt.show()


D = delta

def fix_verts(ax, orient=1):
    for coll in ax.collections:
        for pth in coll.get_paths():
            vert = pth.vertices
            vert[1:3,orient] = scipy.average(vert[1:3,orient])

fig = pylab.figure(figsize=(8,8))

# Compute and plot first dendrogram.
ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
Y = sch.linkage(D, method='centroid')
Z1 = sch.dendrogram(Y, orientation='right')
ax1.set_xticks([])
ax1.set_yticks([])

# Compute and plot second dendrogram.
ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
Y = sch.linkage(D, method='single')
Z2 = sch.dendrogram(Y)
ax2.set_xticks([])
ax2.set_yticks([])

# Plot distance matrix.
axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
idx1 = Z1['leaves']
idx2 = Z2['leaves']
D = D[idx1,:]
D = D[:,idx2]
im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
axmatrix.set_xticks([])
fix_verts(ax1,1)
fix_verts(ax2,0)
fig.savefig('hc_2.png')

# diy one-hot encoding to get around some bugs in keras' Tokenizer
nb_event_types = len(np.unique(np.concatenate(labevents)))
vocabulary = {}
v_base = 0
max_len = 0
for seq in labevents:
    if len(seq) > max_len:
        max_len = len(seq)

    for j in seq:
        if j not in vocabulary.keys():
            vocabulary[j] = v_base
            v_base += 1

X = np.zeros((len(labevents), max_len, nb_event_types))

for i, seq in enumerate(labevents):
    for j, v in enumerate(seq):
        X[i, j, vocabulary[v]] = 1

print "Maximum Nuber of events per seq: {}".format(max_len)

latent_dim = 2
decoder_dim = 32
nb_epoch = 25

encoder_inputs = Input(shape=(max_len, nb_event_types))
encoder = LSTM(latent_dim)(encoder_inputs)
z = RepeatVector(max_len)(encoder)
decoder = LSTM(decoder_dim, return_sequences=True)(z)
decoder = Dense(nb_event_types, activation='softmax')(decoder)

model = Model(encoder_inputs, decoder)
encoder_model = Model(encoder_inputs, encoder)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
hist = model.fit(X, X, nb_epoch=nb_epoch)

p_zX = encoder_model.predict(X)

plt.plot(hist.history['loss'])
plt.show()

plt.scatter(p_zX[:, 0], p_zX[:, 1])
plt.show()

