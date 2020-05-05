import pandas
import collections
import math
import collections
from googletrans import Translator
import re
import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import time

import requests
import urllib.parse

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

class RefEnv(object):
    """docstring for RefEnv"""
    def __init__(self, path_PROJECT_DATA= '/Volumes/GoogleDrive/My Drive'):
        self.dst = 'Source'
        self.Export_folder = 'Export'
        self.path_PROJECT_DATA = path_PROJECT_DATA
        self.transformed = '{}/PROJET DATA/DATA/Transformed/{}'.format(self.path_PROJECT_DATA, '{}')

class Selector(object):
    def


class Transformation(object):
    """docstring for RefEnv"""
    def __init__(self):
        self.output = RefEnv().transformed.format('traduct.json')
        self.X = pandas.read_json(self.output).fillna('VIDE')
        #self.export = X.to_csv(RefEnv().src.format('Transformed/prep_df1_clean.csv'))
        self.header = ['cat_struc', 'cat_autre_struc']
        self.header_tr = ['prez_struc', 'prez_produit_struc', 'prez_marche_struc', 'prez_zone_struc', 'prez_objectif_struc', 'prez_innovante_struc', 'prez_duplicable_struc', 'prez_durable_struc'] 

    def lower_case(self, element):
        return element.lower()

    def number_to_text(self, element):
        return re.sub(r'\d+','', element)

    def remove_punctuation(self, element):
        return element.translate(str.maketrans('', '', string.punctuation))

    def remove_whitespaces(self, element):
        return element.strip()

    def preprocesssing(self, element):
        element = self.remove_punctuation(element)
        element = self.number_to_text(element)
        element = self.lower_case(element)    
        element = self.remove_whitespaces(element)
        return element
    
    def operation(self):
        self.X.columns = ['key_main', 'prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc']
        for head in self.header_tr:
            self.X['{}'.format(head)] = self.X[head].apply(self.preprocesssing)
        #self.X.to_csv(RefEnv().src.format('Transformed/prep_df1_clean_trad.csv'))
        return self.X
    
class Keyword_extraction(object):
    def __init__(self,list_stop_word):
        self.X = Transformation().operation()
        self.header_tr = [a for a in self.X.keys().values.tolist() if 'trad' in a]
        self.stopwords_file = '{}/stopwords.txt'.format(RefEnv().transformed)
        self.stop_words = get_set_stopwords()
        self.X = self.operation()

    def get_set_stopwords(self, ):
        fileName = self.stopword_file
        stopword_file = open(fileName, 'r')
        list_stop_word = [line.split(',') for line in stopword_file.readlines()][0]
        return set(stopwords.words('english')).union(set(list_stop_word))

    def tokenize_n_stemm(self, element):
        stemmer =  PorterStemmer()
        return ' '.join([stemmer.stem(i) for i in word_tokenize(element) if not i in self.stop_words])
    
    def tokenize_n_lemm(self, element):
        lemmer=  WordNetLemmatizer()
        return ' '.join([lemmer.lemmatize(i) for i in word_tokenize(element) if not i in self.stop_words])
    
    def operation(self):
        for head in self.header_tr:
            self.X['{}_stemm'.format(head)] = self.X[head].apply(self.tokenize_n_stemm)
            self.X['{}_lemm'.format(head)] = self.X[head].apply(self.tokenize_n_lemm)
        return self.X

class LDA(object):
    from time import time

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import NMF, LatentDirichletAllocation
    from sklearn.datasets import fetch_20newsgroups

    def __init__(self, n_samples = 2000, n_features = 1000, n_components = 20, n_top_words = 20):
        self.n_samples = n_samples 
        self.n_features = n_features
        self.n_components = n_components
        self.n_top_words = n_top_words

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = "Topic #%d: " % topic_idx
            message += " ".join([feature_names[i]
                                 for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)
        print()
        
    def perform_lda(self, data_samples):
        t0 = time()
        print("done in %0.3fs." % (time() - t0))

        # Use tf-idf features for NMF.
        print("Extracting tf-idf features for NMF...")
        tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                           max_features=self.n_features,
                                           stop_words='english')
        t0 = time()
        tfidf = tfidf_vectorizer.fit_transform(data_samples)
        print("done in %0.3fs." % (time() - t0))

        # Use tf (raw term count) features for LDA.
        print("Extracting tf features for LDA...")
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=self.n_features,
                                        stop_words='english')
        t0 = time()
        tf = tf_vectorizer.fit_transform(data_samples)
        print("done in %0.3fs." % (time() - t0))
        print()

        # Fit the NMF model
        print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
              "n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_components, random_state=1,
                  alpha=.1, l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model (Frobenius norm):")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

        # Fit the NMF model
        print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
              "tf-idf features, n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        t0 = time()
        nmf = NMF(n_components=self.n_components, random_state=1,
                  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
                  l1_ratio=.5).fit(tfidf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
        tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        self.print_top_words(nmf, tfidf_feature_names, self.n_top_words)

        print("Fitting LDA models with tf features, "
              "n_samples=%d and n_features=%d..."
              % (self.n_samples, self.n_features))
        lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        t0 = time()
        lda.fit(tf)
        print("done in %0.3fs." % (time() - t0))

        print("\nTopics in LDA model:")
        tf_feature_names = tf_vectorizer.get_feature_names()
        self.print_top_words(lda, tf_feature_names, self.n_top_words)

        return lda, tf_feature_names