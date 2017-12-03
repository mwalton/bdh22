import collections
import numpy as np

def build(words, vocab_size=None):
    # builds from a continuous document
    counts = [['UNK', -1]]
    if vocab_size is None:
        vocab_size = np.unique(words).size

    counts.extend(collections.Counter(words.icd9_code).most_common(vocab_size))
    dictionary = dict()

    for word, _ in counts:
        dictionary[word] = len(dictionary)

    data = []
    unks = 0

    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unks += 1
        data.append(index)
    counts[0][1] = unks

    inverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts, dictionary, inverse_dict


def build_sequences(df, vocab_size, min_seq_len=1):
    # builds from a set of 'sentances'
    data_df = df[['hadm_id', 'seq_num', 'icd9_code']]
    data_df = data_df.set_index(['hadm_id', 'seq_num']).reset_index()

    counts = [['UNK', -1]]

    counts.extend(collections.Counter(data_df.icd9_code).most_common(vocab_size))
    dictionary = dict()

    for word, _ in counts:
        dictionary[word] = len(dictionary)

    data = []
    seq = []
    unks = 0
    last_hadm = df.ix[0].hadm_id

    for row in data_df.itertuples(index=False):
        if row.hadm_id != last_hadm:
            if len(seq) >= min_seq_len:
                data.append(seq)
            seq = []
            last_hadm = row.hadm_id

        index = dictionary.get(row.icd9_code, 0)
        if index == 0:
            unks += 1
        seq.append(index)

    counts[0][1] = unks

    inverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts, dictionary, inverse_dict


def to_one_hot(vocab_size):
    def f(idx):
        zeros = np.zeros(vocab_size)
        zeros[idx] = 1
        return zeros

    return f


def build_dataset(sequences, vocab_size, one_hot=False):
    ''' This constructs simple skip grams,
    assumes max 2 skips per word; skip window of one'''
    data = []
    for seq in sequences:
        for i in range(len(seq) - 1):
            data.append((seq[i], seq[i + 1]))
            data.append((seq[i + 1], seq[i]))

    if one_hot:
        vectorize = to_one_hot(vocab_size)
        x_train = []
        y_train = []

        for d in data:
            x_train.append(vectorize(d[0]))
            y_train.append(vectorize(d[1]))
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

    else:
        x_train = np.asarray(data)[:, 0]
        y_train = np.asarray(data)[:, 1]

    return x_train, y_train
