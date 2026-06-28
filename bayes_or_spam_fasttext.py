import fasttext  # Install from fasttext-community
import fasttext.util
import numpy as np
import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Perceptron
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Prepare dataset, tokenization and evaluation

texts = []
labels = []
indexes = []
base_dir = 'enron1'
ham = os.listdir(f'{base_dir}/ham')
spam = os.listdir(f'{base_dir}/spam')
for filename in ham + spam:
    metadata = filename.split('.')
    index = int(metadata[0]) - 1
    indexes.append(index)
    ham_or_spam = metadata[-2]
    label = 0 if ham_or_spam == 'ham' else 1
    labels.append(label)
    with open(f'{base_dir}/{ham_or_spam}/{filename}', encoding='latin_1') as f:
        text = f.read()[len('Subject: '):]
        texts.append(text)

df = pd.DataFrame({'text': texts, 'label': labels}, index=indexes).sort_index()                      
df_train, df_test = train_test_split(df, shuffle=False)
X_train, X_test = df_train['text'], \
                  df_test['text']
y_train, y_test = df_train['label'], \
                  df_test['label']

def tokenize(text):
    return re.split('[ \n]+', text)

X_t_train, X_t_test = [tokenize(text) for text in X_train], \
                      [tokenize(text) for text in X_test]

def evaluate(predict_fn, X_test, title):
    print(title)
    y_pred = predict_fn(X_test)
    print(confusion_matrix(y_test, y_pred, normalize='all'))


# Graham naive Bayes

# TODO: clean up
class GrahamNB(object):    
    def fit(self, X, y):
        good = {}
        bad = {}
        word_freqs = {}
        for tokens, label in zip(X, y):
            good_or_bad = bad if label == 1 else good
            for token in tokens:
                good_or_bad.setdefault(token, 0)
                good_or_bad[token] += 1
            for token in set(tokens):  # TODO: set for probas?
                word_freqs.setdefault(token, 0)
                word_freqs[token] += 1
        l = len(X)
        word_rfreqs = {w: word_freqs[w] / l for w in word_freqs}
        word_rfreqs = dict(sorted(word_rfreqs.items(), key=lambda item: item[1], reverse=True))
        self.word_rfreqs = word_rfreqs
        probas = {}
        for word in {**good, **bad}.keys():
            if word in probas:
                continue
            g = good.get(word, 0)
            b = bad.get(word, 0)
            probas[word] = b / (g + b)
        self.probas = probas

    def predict(self, X):
        def get_proba(word):
            return self.probas.get(word, 0.4)

        def combine(probas):
            if any(p == 0 for p in probas):
                return 0
            prod = np.prod(probas)
            neg_prod = np.prod([1 - p for p in probas])
            if prod + neg_prod == 0:  # Still possible due to floating point arithmetic
                return 0.5  # Assume that prod and neg_prod are equally small
            return prod / (prod + neg_prod)

        def get_interesting_probas(probas):
            return sorted(probas, key=lambda p: abs(p - 0.5), reverse=True)[:15]

        predictions = []
        for words in X:
            probas = [get_proba(w) for w in words]
            interesting_probas = get_interesting_probas(probas)
            proba = combine(interesting_probas)
            prediction = 1 if proba > 0.9 else 0
            predictions.append(prediction)
        return predictions


grnb = GrahamNB()
grnb.fit(X_t_train, y_train)

evaluate(grnb.predict, X_t_test, 'GrahamNB')

word_rfreqs = grnb.word_rfreqs
word_rfreqs_10 = dict(list(word_rfreqs.items())[:10])
print('P(spam | word) for top 10 frequent words')
print('\n'.join([f'"{w}" ({word_rfreqs[w]}): {grnb.probas[w]}' for w in word_rfreqs_10]))


# Perceptron

count_vectorizer = CountVectorizer()
X_c_train = count_vectorizer.fit_transform(X_train)
X_c_test = count_vectorizer.transform(X_test)
perc = Perceptron()  # TODO: clean up
perc.fit(X_c_train, y_train)
evaluate(perc.predict, X_c_test, 'Perceptron')
perc_feats = list(count_vectorizer.get_feature_names_out())
perc_weights = list(perc.coef_[0])
print(min(perc_weights), max(perc_weights))
print([perc_weights[perc_feats.index(w)] if w in perc_feats else None for w in word_rfreqs_10])


# fastText

print('Loading fastText...')
# fasttext.util.download_model('en', if_exists='ignore')  # FIXME: incompatible with fasttext-community
ft = fasttext.load_model('cc.en.300.bin')

def get_sentence_vector(words):
    word_vecs = np.array([ft.get_word_vector(word) for word in words])
    return np.average(word_vecs, axis=0)

print('Vectorizing...')
X_v_train, X_v_test = [get_sentence_vector(words) for words in X_t_train], \
                      [get_sentence_vector(words) for words in X_t_test]

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_v_train, y_train)

evaluate(neigh.predict, X_v_test, 'fastText + 5-NN')

def predict(n, X, t):
    probas = n.predict_proba(X)
    return [1 if p[1] > t else 0 for p in probas]

evaluate(lambda X: predict(neigh, X, 0.75), X_v_test, 'fastText + 5-NN with 0.75 threshold')
evaluate(lambda X: predict(neigh, X, 0.9), X_v_test, 'fastText + 5-NN with 0.9 threshold')

# FIXME: DRY
print('P(spam | word) for top 10 frequent words')
print('\n'.join([f'"{w}" ({word_rfreqs[w]}): {neigh.predict_proba([ft.get_word_vector(w)])[0][1]}' for w in word_rfreqs_10]))

print('Finetuning fastText...')
ft_transform = lambda text: text.replace('\n', ' ')
ft_enron1_filename = 'cc.en.300.enron1.bin'
try:
    ft_enron1 = fasttext.load_model(ft_enron1_filename)
    print('Loaded existing finetuned model')
except ValueError:
    labels_list = ['ham', 'spam']  # 0: ham, 1: spam
    for subset, df_subset in [('train', df_train), ('test', df_test)]:
        try:
            with open(f'enron1_{subset}.txt', 'x') as f:
                df_subset.apply(lambda row: f.write(f'__label__{labels_list[row["label"]]} {ft_transform(row["text"])}\n'), axis=1)
        except FileExistsError:
            print(f'{subset} file already exists')
    ft_enron1 = fasttext.train_supervised(input='enron1_train.txt', pretrainedVectors='cc.en.300.vec', dim=300)
    ft_enron1.save_model(ft_enron1_filename)
evaluate(lambda X: [1 if ft_enron1.predict(x)[0][0] == '__label__spam' else 0 for x in X],
         df_test['text'].apply(ft_transform),
         'fastText finetuned')
