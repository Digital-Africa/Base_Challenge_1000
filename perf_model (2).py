import model_lib
import chalenge1000
import numpy as np
import collections
import functools
import operator
import time
import random
import progressbar
import re
from operator import itemgetter
from sklearn.preprocessing import StandardScaler
from time import time
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display, HTML
from statistics import median
from statistics import mean 
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2)


### FIRST CLASS: PERF_MODEL => allows to perform the LDA & NMF models for 2 categories & get the precision of the models
class perf_model(object):
	"""docstring for perf_model"""
	def __init__(self, df, categories, test_size, silent=True): # silent allows to limit the information printed on the console 
		self.categories = categories # !!!! categories must be a list of EXACTLY 2 elements 
		self.test_size = test_size
		self.df = df # df must be the training/test set with the label column named 'cat_struc'
		self.silent = silent


	def get_df(self): # This function is not used in this class but allows to get the initial dataframe. The DF is get from the class perf_multicat => see there
		lemm = lambda x: model_lib.Lemm(x,file = 'stopwords_agri.txt').X
		lexic = lambda x: ' '.join([a for a,b in collections.Counter(x.split()).most_common(10000)])
		word_count = lambda x: {a:b for a,b in collections.Counter(x.split()).most_common(10000)}		

		challenge = chalenge1000.Native()
		df_cat_temp = challenge.temp
		
		df_trad = challenge.trad
		df_trad['corpus'] = df_trad[['prez_struc', 'prez_produit_struc']].agg(sum, axis = 1)
		df_trad['corpus_lemm'] = df_trad['corpus'].map(lemm)
		df_trad['lexic_lemm'] = df_trad['corpus_lemm'].map(lexic)
		df_trad['word_weight'] = df_trad['corpus_lemm'].map(word_count)

		df = df_trad.join(df_cat_temp)

		return df

	# lexic_extractor is the function that allows to get the vocabulary to perform models & weight for each word + word assigned to a category for 2 categories
	def lexic_extractor(self): 
		lex1 = collections.Counter(self.df[self.df['cat_struc']== self.categories[0]]['lexic_lemm'].agg(sum, axis=0).split()).most_common(5000) # gets the word and the count in each doc
		lex2 = collections.Counter(self.df[self.df['cat_struc']== self.categories[1]]['lexic_lemm'].agg(sum, axis=0).split()).most_common(5000)
		# CAREFUL !!! There's a limit in the collections.Counter above which is the most_common => it can be changed to 100000 for example, it limits the number of words in the vocabulary
		len_cat1 = len(self.df[self.df['cat_struc']==self.categories[0]])
		len_cat2 = len(self.df[self.df['cat_struc']==self.categories[1]])
    
		counter_cat1 = list({(a,b*100/len_cat1) for a,b in lex1}.union({(k,0) for k,l in lex2 if k not in [a for a,b in lex1]})) # gets a list of all words with a count for category 0 & 1 and with count 0 for words in category 1
		counter_cat2 = list({(a,b*100/len_cat2) for a,b in lex2}.union({(k,0) for k,l in lex1 if k not in [a for a,b in lex2]})) # Same but with 0 for words not in cat1

		vocab = [a for a,b in counter_cat1] # this is the full lexic of the 2 categories
    
		id = operator.itemgetter(0)
    
		weight_cat1 = self.df[self.df['cat_struc']== self.categories[0]]['word_weight']
		weight_cat2 = self.df[self.df['cat_struc']== self.categories[1]]['word_weight']
    
		vocab_weight_cat1 = [(key, value/len_cat1) for key, value in dict(functools.reduce(operator.add, map(collections.Counter, weight_cat1))).items()] 
		# gets a weight for each word which is the sum of the weight (count by cell for a word) of each word for the category 0
		vocab_weight_cat1  = list({i for i in vocab_weight_cat1 if id(i) in vocab}.union({(k, 0) for k in vocab if k not in [a for a,b in vocab_weight_cat1]}))
		# make sure to take only the words in the lexic defined above of the 2 categories and weight 0 for the words not in the category
		# the same operation is applied for category 1 
		vocab_weight_cat2 = [(key, value/len_cat2) for key, value in dict(functools.reduce(operator.add, map(collections.Counter, weight_cat2))).items()]
		vocab_weight_cat2  = list({i for i in vocab_weight_cat2 if id(i) in vocab}.union({(k, 0) for k in vocab if k not in [a for a,b in vocab_weight_cat2]}))
    	# we make 3 dictionaries with the word as key and the count in category 1, the weight of the word in cat 0 and in cat 1
		id_counter_cat2 = {id(rec): rec[1:] for rec in counter_cat2} # this dic is the unique count for category 1 with the word as key
		id_vocab_weight_cat1 = {id(rec): rec[1:] for rec in vocab_weight_cat1} # this dic is the word weight for category 0 with the word as key
		id_vocab_weight_cat2 = {id(rec): rec[1:] for rec in vocab_weight_cat2} # this dic is the word weight for category 1 with the word as key
    	# NB: the vocabulary (the keys for each dictionary) is the same in all dictionaries
		merged = [i + id_counter_cat2[id(i)] for i in counter_cat1 if id(i) in id_counter_cat2] 
		# we make a list of tuples with the word as first argument and unique count for category 0 as 2nd argument, we add the unique count for category 1 as 3rd argument  
		merged = [i + id_vocab_weight_cat1[id(i)] for i in merged if id(i) in id_vocab_weight_cat1]
		# we append the tuples  of this with the word weight for category 0
		merged = [i + id_vocab_weight_cat2[id(i)] for i in merged if id(i) in id_vocab_weight_cat2]
		# and we append the tuples with the word weight for category 1

		keyword_map = pandas.DataFrame(merged, columns = ['keyword','cat1', 'cat2', 'weight_cat1', 'weight_cat2']) #make a dataframe of the list
    
		def produit(x,y):
			return x*(y+1)

		def categorie_predite(x):
			if x>0:
				return self.categories[0]
			if x<0:
				return self.categories[1]
    
		keyword_map['score_cat1'] = keyword_map.apply(lambda x: produit(x['cat1'], x['weight_cat1']), axis = 1) # we make a score for each word for category 0 which is only the product of the word count and the word weight
		keyword_map['score_cat2'] = keyword_map.apply(lambda x: produit(x['cat2'], x['weight_cat2']), axis = 1) # same for category 1
		keyword_map['difference'] = abs(keyword_map['score_cat1'] - keyword_map['score_cat2']) # now take the absolute value of the difference of the 2 scores
		keyword_map['true_diff'] = keyword_map['score_cat1'] - keyword_map['score_cat2'] 
		keyword_map['predict_cat'] = keyword_map['true_diff'].apply(categorie_predite) # if the difference is positive predict category is category 0 if not it's category 1
		keyword_map.sort_values('difference', ascending = False)
    
		#percentile = pandas.DataFrame(keyword_map['difference'].quantile(np.linspace(.01, 1, 99, 00)))
		#percentile['lag_diff'] = percentile.diff(axis = 0)
    
		pct = []
		for i in range(1, 100):
			i = i/100
			j = i -0.01
			temp = keyword_map['difference'].quantile(i) # that's the value of the quantile for the difference in the 2 scores
			diff = temp - keyword_map['difference'].quantile(j) # the difference with the next quantile is a proxy for the growth rate
			if i ==0.01: 
				pct.append([i, temp, abs(temp)/2]) # NB for the first quantile there's no growth so we take the value of the first quantile divided by 2
				# this only allows to increase the mean value but THE DIVISION BY 2 IS COMPLETELY RANDOM 
			else: 
				pct.append([i, temp, diff])
		pct1 = pandas.DataFrame(pct, columns=['a', 'b', 'c']) # this is a dataframe of 100 rows with the value of the quantile as the 2nd column and the growth rate as 3rd column
		
		if self.silent == False:
			print(plt.scatter(pct1.index, pct1['c']))
    
		select = pct1[pct1['c']>pct1['c'].mean()]['a'].values.tolist() # we select the quantile for which the growth rate is above the mean 
		#this allows to select only the word which are different enough in terms of score so that the distinguish clearly the 2 categories
		
		if self.silent == False:
			print(select)
    
		def cible(liste):
			for i in liste:
				if i<=0.5:
					next
				else:
					return i
					break
    
		quant = cible(select) # !!! WARNING !!! Here again we decide completely randomnly to choose a quantile which is above the median
		# this means that we prefer to have words which have above the median in terms of difference i.e. words which distinguish the 2 categories above the median

		if self.silent == False:
			print('Quantile sélectionné :', quant)

		select_keyword = keyword_map[keyword_map['difference']>keyword_map['difference'].quantile(quant)] # this dataframe is limited to the word selected with a difference in score above the selected quantile
		share_per_cat = [round(len(select_keyword[select_keyword['predict_cat']==self.categories[0]])/len(select_keyword)*100, 2), round(len(select_keyword[select_keyword['predict_cat']==self.categories[1]])/len(select_keyword)*100,2)]
		print('Part du lexique correspondant à la categorie {}: {}%; part du lexique correspondant à la catégorie {}: {}%'.format(self.categories[0], share_per_cat[0], self.categories[1], share_per_cat[1]))
		# Here it could be possible to limit the distance between the share_per_cat in order to keep some balance between the lexic of the 2 categories
		lexic_model = [i for i in select_keyword['keyword']]    # this lexic will be the one used in the models later on
		if self.silent == False:
			print('Longueur du lexique :',len(lexic_model))
    
		dic_word_weight = keyword_map[['keyword', 'score_cat1', 'score_cat2', 'predict_cat']].set_index('keyword').T.to_dict('list') 
		# this dictionary have the words as keys and output with [0:the score for category 0, 1: the score for category 1, 2: the category assigned] 
    
		return lexic_model, dic_word_weight

	def train_test(self, seed): # this first functions allows to return 2 dataframes : one dataframe for test & one of train with the same ponderation for each category as the initial dataframe
		text_corpus_label = self.df[self.df['cat_struc'].isin(self.categories)][['corpus_lemm','cat_struc']]  
		test,train = train_test_split(text_corpus_label, test_size = self.test_size, random_state = seed)
		train = train['corpus_lemm']
		test = test['corpus_lemm']		
		return train, test

	def train_test2(self, seed): # this second functions returns 2 df one of train, one of test and a dictionary with the index for each dataframe allows to change the ponderation between the 2 categories of the initial dataframe
		random.seed(seed)
		corpus = self.df[self.df['cat_struc'].isin(self.categories)][['corpus_lemm', 'cat_struc']]
		if len(corpus[corpus['cat_struc']==self.categories[0]]) < len(corpus[corpus['cat_struc']==self.categories[1]]):
			max_train = round(len(corpus[corpus['cat_struc']==self.categories[0]]) * self.test_size)
			factor = 1 # if the first category is shorter, the train set will have as many observation as the length of the initial dataframe for that category
			# multiplied by a factor which is test_size

		else:
			max_train = round(len(corpus[corpus['cat_struc']==self.categories[1]]) * self.test_size) # if the first category is longer there are 2 possibility
			if len(corpus[corpus['cat_struc']==self.categories[0]])/len(corpus[corpus['cat_struc']==self.categories[1]]) <= 2: 
				# first if the length of the dataframe for the categories 0 is less than twice the size of the dataframe for category 1 we keep the same proportion
				factor = len(corpus[corpus['cat_struc']==self.categories[0]])/len(corpus[corpus['cat_struc']==self.categories[1]])
			else : 
				# if the length of the dataframe for the categories 0 is more than twice the size of the dataframe for category 1 we limit the proportion to twice the category 1
				factor = 2

		train_index = random.sample(list(corpus[corpus['cat_struc']==self.categories[0]].index.values), round(max_train*factor)) + random.sample(list(corpus[corpus['cat_struc']==self.categories[1]].index.values), max_train)
		# by this operation, the train dataframe is either 50/50 for the two categories or up to 66/33% proportion with 66% being the maximum for category 0 (50% is the maximum for category 1)
		test_index = [i for i in corpus.index.values if i not in train_index]

		train = corpus[corpus.index.isin(train_index)]['corpus_lemm']
		test = corpus[corpus.index.isin(test_index)]['corpus_lemm']
		train_test_index = {'train': train_index, 'test':test_index}

		return train, test, train_test_index

	def perf_lda(self, train, test, model_param): # simple function to perform the LDA model with the parameter defined and the train & test df
		model = model_lib.Models(**model_param)    
		models = model.run_model_LDA(train.values.tolist())
		predict = pandas.DataFrame(model.reverse_lda(test.values.tolist(), models) , index =test.index)
		#display(predict)
		return models, predict, models['LDA']['components'] # it returns the full model which is a dictionary with all the model output defined in model_lib
		# a dataframe with the prediction of the model and the components of the LDA model which are the words used by the model which we use for topic_to_category

	def perf_nmf(self, train, test, model_param): # exact same function as above for NMF model
		model = model_lib.Models(**model_param)    
		models = model.run_model_NMF(train.values.tolist())
		predict = pandas.DataFrame(model.reverse_nmf(test.values.tolist(), models) , index =test.index)
		#display(predict)
		return models, predict, models['NMF_Frobenius']['components']

	def topic_to_category(self, components, vocab_weight, reverse): # this function allows to define which topic corresponds to which category
		topic_category = {}
		for i in range(2):
			words = components['Topic {}'.format(i)]
			cat1 = []
			cat2 = []
			for j in words: # we take the words that are used in the components for each topic of the model
				if j in vocab_weight.keys():
					if reverse==2 : # just for the case when category 0 does not correspond to the vocabulary
						cat1.append(vocab_weight[j][1])
						cat2.append(vocab_weight[j][0])
					else:
						cat1.append(vocab_weight[j][0])
						cat2.append(vocab_weight[j][1])
			cat1 = sum(cat1)
			cat2 = sum(cat2)
			topic_category[i] = cat1 - cat2 # take the difference of the sum of the score of the words of the components for each topic in the model
    
		if topic_category[0]>topic_category[1]: # if the score is larger for the topic 0 than the topic 1 it is assigned to the category 0 and topic 1 to category 1
			topic_category[0]= self.categories[0]
			topic_category[1]= self.categories[1]
		else: # conversely if the score is small for the topic 0
			topic_category[0]= self.categories[1]
			topic_category[1]= self.categories[0]
        
		return topic_category

	def precision(self, predict, topic_category): # this functions allows to measure the precision of the model with respect to the true category
    
		def get_topic(e):
			if e >0:
				return topic_category[0]
			if e <0:
				return topic_category[1]
    
		def true_positve(a,b):
			if a == b:
				return 1
			else:
				return 0

		precision = predict.join(self.df['cat_struc'])
		precision[0] = precision[0]*100
		precision[1] = precision[1]*100
		precision['topic'] = precision[0]-precision[1]
		#display(predict)

		precision['predict_label'] = precision['topic'].map(get_topic)
		#display(predict)

		precision['classification_kpi'] = precision.apply(lambda x: true_positve(x['cat_struc'],x['predict_label']), axis= 1)
		#display(predict)
		# all the operations above create a dataframe with the prediction and the true categories and check the correspondance

		cat0 = precision[precision['cat_struc'] == self.categories[0]] # dataframe precision of the category 0
		cat1 = precision[precision['cat_struc'] == self.categories[1]] # dataframe precision of the category 1

		vrai_positif_cat0 = cat0['classification_kpi'].sum()/len(cat0)*100
		vrai_positif_cat1 = cat1['classification_kpi'].sum()/len(cat1)*100

		#print(model_parameters)
    
		if self.silent == False:
			print('\nPrécision vrai positif catégorie {}: {}'.format(self.categories[0], vrai_positif_cat0))
    
		if self.silent == False:
			print('\nPrécision vrai positif catégorie {}: {}'.format(self.categories[1], vrai_positif_cat1))
    
		#model_parameters.update({'precision':agri['classification_kpi'].sum()/len(agri)*100, 'vocabulaire': models['LDA']['feature_names']})
		#errors = agri[agri['classification_kpi'] == 0].sort_values('topic')
		#collections.Counter(' '.join(errors.join(text_corpus)['collection_counter'].values.tolist()).split()).most_common(10)
		#return model_parameters, agri, other, models['LDA']
    
		return precision, vrai_positif_cat0, vrai_positif_cat1

	def operation_lda(self, model_param, vocab, vocab_weight, reverse = 1): # this function perform the LDA model and gets the precision
		#vocab, vocab_weight = self.lexic_extractor()
		model_param.update({'vocabulary': vocab, 'n_features': len(vocab)})

		precision_param = []
		output = {}

		bar = progressbar.ProgressBar(maxval=99).start() # that's to display the advancement

		for i in range(100): # we iterate over different train/test output to get a median result of the model
			if self.silent == False:
				print('\n\nItération numéro:', i)
			#train, test = self.train_test(i)
			train,test,train_test_index = self.train_test2(i) # get the train/test df
			models, predict, components = self.perf_lda(train, test, model_param) # get the model, the prediction df & the components of the model
			topic_category = self.topic_to_category(components, vocab_weight, reverse) # find the right topic -category assignation
			precis, vrai_positif_cat0, vrai_positif_cat1 = self.precision(predict, topic_category) # here we get the kpis of the model with that train/test df
			precision_param.append([vrai_positif_cat0, vrai_positif_cat1]) # this is used to define the overall performance of the parameters
			output.update({i: {'seed':i, 'train_set':train, 'test_set': test, 'lda_model': models, 'topic_to_category':topic_category, 'precision_df': precis}}) 
			# this dic takes the result of all models over each iteration, the key corresponds to the seed used of the train/test repartition
			bar.update(i)

		med_precision_param = [median([a for a,b in precision_param]), median([b for a,b in precision_param])] # takes the median precision over all iteration
		print('\nPrécision médiane des paramètres',med_precision_param)

		df_precision_param = pandas.DataFrame(precision_param, columns = ['cat0', 'cat1'])
		df_precision_param['sum'] =  df_precision_param[['cat0', 'cat1']].agg(sum, axis = 1)
		best_seed = df_precision_param[df_precision_param['cat0']>=df_precision_param['cat0'].quantile(0.90)]['sum'].idxmax() 
		# the best seed is defined as the one that maximizes the sum of the precision for the 2 categories in the 10% that have the best precision to predict the category 0
		print('\nModèle le plus performant',best_seed)

		return df_precision_param, med_precision_param, best_seed, output  # recovers the dataframe of the precision for every iteration (which is df_precision_param)
		# the median precision of the parameters, the best_seed for the parameters and the full output with the model etc for each iteration

	def operation_nmf(self, model_param, vocab, vocab_weight, reverse = 1): # exact same than above with the nmf
		#vocab, vocab_weight = self.lexic_extractor()
		model_param.update({'vocabulary': vocab, 'n_features': len(vocab)})

		precision_param = []
		output = {}

		bar = progressbar.ProgressBar(maxval=99).start()

		for i in range(100):
			if self.silent == False:
				print('\n\nItération numéro:', i)
			#train, test = self.train_test(i)
			train,test,train_test_index = self.train_test2(i)
			models, predict, components = self.perf_nmf(train, test, model_param)
			topic_category = self.topic_to_category(components, vocab_weight, reverse)
			precis, vrai_positif_cat0, vrai_positif_cat1 = self.precision(predict, topic_category)
			precision_param.append([vrai_positif_cat0, vrai_positif_cat1])
			output.update({i: {'seed':i, 'train_set':train, 'test_set': test, 'nmf_model': models, 'topic_to_category':topic_category, 'precision_df': precis}})
			bar.update(i)

		med_precision_param = [median([a for a,b in precision_param]), median([b for a,b in precision_param])]
		print('\nPrécision médiane des paramètres',med_precision_param)

		df_precision_param = pandas.DataFrame(precision_param, columns = ['cat0', 'cat1'])
		df_precision_param['sum'] =  df_precision_param[['cat0', 'cat1']].agg(sum, axis = 1)
		best_seed = df_precision_param[df_precision_param['cat0']>=df_precision_param['cat0'].quantile(0.90)]['sum'].idxmax()
		print('\nModèle le plus performant',best_seed)

		return df_precision_param, med_precision_param, best_seed, output 


class kmeans(object):
	def __init__(self, dic_predict, df_predict):
		self.dic_predict = dic_predict # This dictionary is supposed to be the output of the class multicat with the predicted value of the nmf/lda/word score
		self.df_predict = df_predict # this is a dataframe with the keys & the true labels


	def train_test_index(self, df, cat, test_size, seed = 0): # this function allows to get a balanced train sample + a test sample
		random.seed(seed)
		df = df[df['cat_struc'].isin(cat)]
		if len(df[df['cat_struc']==cat[0]])<len(df[df['cat_struc']==cat[1]]): # there should be no more than 2 categories as an input
			max_train = round(len(df[df['cat_struc']==cat[0]]) * test_size)
			flag = 1 
		else: 
			max_train = round(len(df[df['cat_struc']==cat[1]]) * test_size)
			flag = 2
		train_index = random.sample(list(df[df['cat_struc']==cat[0]].index.values), max_train) + random.sample(list(df[df['cat_struc']==cat[1]].index.values), max_train)
		if test_size > 0.9: # this chunk is only for the case when test_size = 1, in that case, for one category, there is no observation in the test df 
			if flag == 1: # we just take randomly 10% of the category which is saturated in the train set for the test set
				test_index = [i for i in df.index.values if i not in train_index] + random.sample(list(df[df['cat_struc']==cat[0]].index.values), round(len(df[df['cat_struc']==cat[0]]) * 0.1))
			if flag == 2:
				test_index = [i for i in df.index.values if i not in train_index] + random.sample(list(df[df['cat_struc']==cat[0]].index.values), round(len(df[df['cat_struc']==cat[1]]) * 0.1))
		else:
			test_index = [i for i in df.index.values if i not in train_index]
		return train_index, test_index

	def apply_model(self, model, X, df_cat, max_dist0, max_dist1, dist_cat):
		transform = model.transform(X) 
		# the model here is the kmeans model trained (fitted) on the train sample, we get an object with tuples of 2 values for each observation, the distance for cluster 0 and 1
		df = X.copy()
		df['distance0'] = [a for a, b in transform]
		df['distance1'] = [b for a, b in transform]

		def in_range(value, max_value):
			if value < max_value:
				return 1
			else:
				return 0
		
		df[dist_cat['distance0']] = df.apply(lambda x: in_range(x['distance0'], max_dist0), axis=1) # the maximum distance is defined by the train set, dist_cat assigns a category to a cluster
		df[dist_cat['distance1']] = df.apply(lambda x: in_range(x['distance1'], max_dist1), axis=1)
		df = df.join(df_cat['cat_struc'])

		all_cat = list(set(df['cat_struc']))
		all_cat = [x for x in all_cat if str(x) not in ['nan', 'autre']]
		vrai_positif = []
		faux_positif = []

		for cat in all_cat: # this loop is what allows to get the performance of the model, we compute only true positive and false positive
			true_dist0 = sum(df[df['cat_struc']==cat][dist_cat['distance0']])/len(df[df['cat_struc']==cat])*100
			true_dist1 = sum(df[df['cat_struc']==cat][dist_cat['distance1']])/len(df[df['cat_struc']==cat])*100
			if cat == dist_cat['distance0']:
				vrai_positif.append(true_dist0)
			else:
				faux_positif.append(true_dist0)

			if cat == dist_cat['distance1']:
				vrai_positif.append(true_dist1)
			else:
				faux_positif.append(true_dist1)

		vrai_positif = mean(vrai_positif) # then take the mean of the list of the true positive values and false positive
		faux_positif = mean(faux_positif)

		result = vrai_positif - faux_positif # the result (objective function) is the difference between the 2 scores

		return df, vrai_positif, faux_positif, result

	def precision_train(self, model, X, df_cat, quant = 0.95):
		df = X.copy()
		transform = model.transform(df)
		df['distance0'] = [a for a, b in transform]
		df['distance1'] = [b for a, b in transform] # we repeat the same operation as above to get the predicted distance with the kmeans model fitted on the train sample
		
		predict =  df.join(df_cat['cat_struc'])
		cat = list(set(predict['cat_struc']))
		dist_cat = {} # this dictionary will enable to assign one category for each cluster in the kmeans model
		
		if predict[predict['cat_struc']==cat[0]]['distance0'].mean() < predict[predict['cat_struc']==cat[1]]['distance0'].mean():
			dist_cat.update({'distance0': cat[0], 'distance1': cat[1]}) # if the mean distance is smaller for the category 0 than the category 1, the distance 0 (i.e. cluster 0) is assigned the category 0 
		else:
			dist_cat.update({'distance0': cat[1], 'distance1': cat[0]}) # and conversely
		max_dist0 = predict[predict['cat_struc']==dist_cat['distance0']]['distance0'].quantile(quant) # the maximum distance from the kmeans is defined by a quantile from which quant% will be true positives in the train sample
		max_dist1 = predict[predict['cat_struc']==dist_cat['distance1']]['distance1'].quantile(quant) # and the same with the other category for the cluster 1
		# the maximum distance will be the cutoff value for which it is considered that the observation belongs or not to the assigned category in the test sample
		return max_dist0, max_dist1, dist_cat

	def train_parameter(self, df, df_cat, categories): # this function is the deprecated function for train_parameter2
		all_results = {}
		full_results = {}
		for i in range(5,11): # this tests with only one draw of train/test sets and therefore is subject to random bias, see train_parameter2 for the actual function used
			test_size = i/10
			train_index, test_index = self.train_test_index(df_cat, categories, test_size)
			cols = list(df.columns)
			col_test = []
			j = 0
			best_cols = {}
			result_cols = []
			full_result_cols = {}
			while len(col_test)<len(cols):
				col_test.append(cols[j])
				train = df[df.index.isin(train_index)][col_test]
				test = df[df.index.isin(test_index)][col_test]
				model = km.fit(train)
				max_dist0, max_dist1, dist_cat = self.precision_train(model, train, df_cat)
				_, vrai_positif, faux_positif, result = self.apply_model(model, test, df_cat, max_dist0, max_dist1, dist_cat)
				#result = vrai_positif_cat0 + vrai_positif_cat1 - faux_positif_cat0 - faux_positif_cat1
				best_cols.update({j: col_test.copy()})
				result_cols.append(result)
				j += 1
			while len(col_test) > 1:
				col_test.pop(0)
				train = df[df.index.isin(train_index)][col_test]
				test = df[df.index.isin(test_index)][col_test]
				model = km.fit(train)
				max_dist0, max_dist1, dist_cat = self.precision_train(model, train, df_cat)
				_, vrai_positif, faux_positif, result = self.apply_model(model, test, df_cat, max_dist0, max_dist1, dist_cat)
				#result = vrai_positif_cat0 + vrai_positif_cat1 - faux_positif_cat0 - faux_positif_cat1
				best_cols.update({j: col_test.copy()})
				result_cols.append(result)
				full_result_cols.update({j:[vrai_positif, faux_positif, result]})
				j += 1
			all_results.update({i: result_cols})
			full_results.update({i: full_result_cols})

		right_size = pandas.Series(pandas.DataFrame.from_dict(all_results, orient='index').max(axis = 1)).idxmax()
		right_cols = best_cols[pandas.Series(all_results[right_size]).idxmax()]

		perf = full_results[right_size][pandas.Series(all_results[right_size]).idxmax()]
		print('Vrai positif: {}, faux positif : {}, résultat: {}'.format(perf[0], perf[1], perf[2]))

		return right_size, right_cols


	def train_parameter2(self, df, df_cat, categories): 
	# this functions allows to define the right size of the train set and the right columns to maximize the precision of the kmeans model
		all_results = {}
		full_results = {}
		n_iter = 20*23*6
		bar = progressbar.ProgressBar(maxval=n_iter).start()
		n = 0
		for i in range(5,11): # with this loop we define the test_size i.e. the share of the observation of the smallest category that will be used for the train set 
			# !!! ATTENTION we start from 5 for the loops for i but to be consistent we should start at 1
			test_size = i/10
			cols = list(df.columns)
			col_test = []
			j = 0
			best_cols = {}
			result_cols = []
			full_result_cols = {}
			
			while len(col_test)<len(cols): # here this while allows to gradually test the model with an addition, one by one, of all the columns from the initial dataframe
				col_test.append(cols[j])
				med_result = []
				med_vrai_positif = []
				med_faux_positif = []
				for k in range(20): # this loop allows to get 20 draw of train/test sets in order to test the performance of the parameters (test_size + columns)
					n +=1
					train_index, test_index = self.train_test_index(df_cat, categories, test_size, seed = k) # here define the train/test sets
					train = df[df.index.isin(train_index)][col_test]
					test = df[df.index.isin(test_index)][col_test]
					model = km.fit(train) # fit the model
					max_dist0, max_dist1, dist_cat = self.precision_train(model, train, df_cat) # get the max values for which one distance is small enough to be considered in the category + the cluster-category assignation
					_, vrai_positif, faux_positif, result = self.apply_model(model, test, df_cat, max_dist0, max_dist1, dist_cat) # we apply the model on the test sets and get the performance results
					med_result.append(result) # this lists registers the result for each draw of train/test
					med_vrai_positif.append(vrai_positif)
					med_faux_positif.append(faux_positif)
					bar.update(n)
				true_result = median(med_result) # to get the true value for the combination test_size/columns we take the median of the results of the 20 draws
				true_vrai_positif = median(med_vrai_positif)
				true_faux_positif = median(med_faux_positif)
				best_cols.update({j: col_test.copy()}) # this dictionary registers the combinations of columns used
				result_cols.append(true_result)
				full_result_cols.update({j:[true_vrai_positif, true_faux_positif, true_result]})
				j += 1
			
			while len(col_test) > 1: # we repeat the exact same process than above but removing the first column
				col_test.pop(0)
				med_result = []
				med_vrai_positif = []
				med_faux_positif = []
				for k in range(20):
					n+=1
					train_index, test_index = self.train_test_index(df_cat, categories, test_size, seed = k)
					train = df[df.index.isin(train_index)][col_test]
					test = df[df.index.isin(test_index)][col_test]
					model = km.fit(train)
					max_dist0, max_dist1, dist_cat = self.precision_train(model, train, df_cat)
					_, vrai_positif, faux_positif, result = self.apply_model(model, test, df_cat, max_dist0, max_dist1, dist_cat)
					med_result.append(result)
					med_vrai_positif.append(vrai_positif)
					med_faux_positif.append(faux_positif)
					bar.update(n)
				true_result = median(med_result)
				true_vrai_positif = median(med_vrai_positif)
				true_faux_positif = median(med_faux_positif)
				best_cols.update({j: col_test.copy()})
				result_cols.append(true_result)
				full_result_cols.update({j:[true_vrai_positif, true_faux_positif, true_result]})
				j += 1
			all_results.update({i: result_cols}) # this dictionary registers the performance of a parameter test_size = i/10 for all the different combination of columns (23)
			full_results.update({i: full_result_cols})
		# Here we have a dictionary (all_results) with the results for all columns and for all values from 0.5 to 1 of the test_size
		right_size = pandas.Series(pandas.DataFrame.from_dict(all_results, orient='index').max(axis = 1)).idxmax() # this selects the best test_size (the one which has the maximal value of true_result)
		right_cols = best_cols[pandas.Series(all_results[right_size]).idxmax()] # this selects the best set of columns (the one with the maximal value of results in the line i)

		perf = full_results[right_size][pandas.Series(all_results[right_size]).idxmax()]
		print('Vrai positif: {}, faux positif : {}, résultat: {}'.format(perf[0], perf[1], perf[2]))

		return right_size, right_cols, all_results, full_results

	def perform_km(self, weight = 0.5): # the weight is assigned by default but can be changed, it's the balance between true positive & false positive
	# the larger the weight, the more true positives are weighted in the optimization function
		keys = self.dic_predict.keys() # takes the dictionary with all the duos of categories from the output of the class multicat
		new_dic = {}
		new_output = {}
		X_cat = self.df_predict

		def select_text(text, list_text):
			output = []
			for i in list_text:
				if type(re.search(text, i))== re.Match:
					output.append(i)
			return output
		
		for k in keys: # this loop with be used for every duo of categories
			X = self.dic_predict[k].copy() # it takes the dataframe of the predicted values for the 2 categories (12 columns)
			X_col = list(X.columns)
			cat = k.split('_') # defines the 2 categories at stake
			scale_col = select_text('word', X_col)
			not_scale = [col for col in X_col if col not in scale_col] 
			X[scale_col] = MinMaxScaler().fit_transform(X[scale_col]) 
			# !!! ATTENTION in the line just above the 4 first columns (count of the word weight for each category by observation and unique count for each category)
			# is scaled with a min-max scaler (the value will be combined between 0 and 1)

			right_size, right_cols, _,_ = self.train_parameter2(X, X_cat, cat) # gets the right size & right columns which will be used for the model
			right_size = float(right_size/10)
			train_index, test_index = self.train_test_index(X_cat, cat, right_size) # defines the train & test sets (randomly, doesn't take the most efficient)
			train = X[X.index.isin(train_index)][right_cols]
			model = km.fit(train) # here we define the kmeans model that will be used all along for the duo of categories 
			full_test = X[right_cols]
			print(right_size, right_cols)
			best_quant = []
			for i in range(75, 100): # here we define the best quantile for which the maximum values for the 2 clusters will be defined starting in centile 75
				# NB: this loop should be improved by iterating over multiple draws of train/test sets
				quant = i/100
				max_dist0, max_dist1, dist_cat = self.precision_train(model, train, X_cat, quant)
				_, vrai_positif, faux_positif, _ = self.apply_model(model, full_test, X_cat, max_dist0, max_dist1, dist_cat)
				performance = weight * vrai_positif - (1-weight) * faux_positif # the objective function by default assigns the same weight to true positives and false positives
				best_quant.append(performance)
			final_quant = (pandas.Series(best_quant).idxmax() + 75)/100 # we only take the quantile that maximizes the objective function performance
			print(final_quant)
			max_dist0, max_dist1, dist_cat = self.precision_train(model, train, X_cat, final_quant) # we use that quantile in the final model
			full_test, vrai_positif, faux_positif, result = self.apply_model(model, full_test, X_cat, max_dist0, max_dist1, dist_cat)
			print('Performance globale: vrai positif :{}, faux positif: {}, résultat: {}'.format(vrai_positif, faux_positif, result))
			new_dic.update({k: full_test})

			# !!!!! ATTENTION: THIS IS THE MOST IMPORTANT PART OF THE CLASS !!!
			# With that loop, for all the observations that has not been assigned in the category (i.e. those for which the distance was above the threshold)
			# the value of the prediction by the NMF/LDA and word count is assigned to 0 => this category will not appear in the final decision to assign the category
			for c in cat: 
				col_cat = select_text(c, X_col)
				list_index = full_test[full_test[c]==0].index
				X.loc[list_index, col_cat] = 0

			X[not_scale] = MinMaxScaler().fit_transform(X[not_scale]) # !!! ATTENTION last transformation, all the columns not scaled are transformed with a min-max scaler
			new_output.update({k: X})
			

		return new_dic, new_output



class perf_multicat(object): # this class is the main one to be used in order to activate the 2 above classes with multiple categories to define the models

	def __init__(self, categories, test_size = 0.8): 
		self.categories = categories
		self.test_size = test_size
		self.df = self.get_df()

	def get_df(self): # this function gets a ready dataframe with the text ready for processing in the model and the true label 
		lemm = lambda x: model_lib.Lemm(x,file = 'stopwords_agri.txt').X # this process get the lematisation of the text
		lexic = lambda x: ' '.join([a for a,b in collections.Counter(x.split()).most_common(10000)]) # this takes single occurence of words in a text
		word_count = lambda x: {a:b for a,b in collections.Counter(x.split()).most_common(10000)} # this gets a dictionary with single occurence of words and the number of times the word appeared in the original text		

		challenge = chalenge1000.Native()
		df_cat_temp = challenge.temp
		
		df_trad = challenge.trad # text of initial dataframe traduced in english
		df_trad['corpus'] = df_trad[['prez_struc', 'prez_produit_struc']].agg(sum, axis = 1) # we take only 2 columns, presentation of the structure & the product
		df_trad['corpus_lemm'] = df_trad['corpus'].map(lemm) # apply the lemmatization
		df_trad['lexic_lemm'] = df_trad['corpus_lemm'].map(lexic) # gets single occurence of words from the text lemmatized
		df_trad['word_weight'] = df_trad['corpus_lemm'].map(word_count) # gets the count of each word in the text lemmatized

		df = df_trad.join(df_cat_temp) # allows to get the true labels (categories with "cat_struc")

		return df
	
	def score_word(self, df, vocab_weight, categories): # this function gets the score by word for each observation
		keys = list(df.index)
		dic_out = {}
		for index in keys:
			sample = df[index]
			sample = sample.split() # takes the text lemmatized for the observation as a list with each word as element 

			cat0 = []
			cat1 = []
			count_cat0 = []
			count_cat1 = []

			for word in sample:
				if word in vocab_weight.keys(): # from the dictionary vocab_weight we can get the score for category 0, category 1 and the category assigned
					cat0.append(vocab_weight[word][0]) # gives the score for the category 0
					cat1.append(vocab_weight[word][1]) # give the score for the category 1

					if vocab_weight[word][2] == categories[0]: # adds one if the category assigned is category 0
						count_cat0.append(1)
					if vocab_weight[word][2] == categories[1]: # the same if category assigned is category 1
						count_cat1.append(1)

			cat0 = sum(cat0)
			cat1 = sum(cat1)
			count_cat0 = sum(count_cat0)
			count_cat1 = sum(count_cat1)
			dic_out[index] = [cat0, cat1, count_cat0, count_cat1] # for each observations returns the weight for category 0 & 1 and the count of word corresponding to category 0 and 1
		
		df_out = pandas.DataFrame.from_dict(dic_out, orient='index', columns = ['{}_word_weight'.format(categories[0]), '{}_word_weight'.format(categories[1]), '{}_word_count'.format(categories[0]), '{}_word_count'.format(categories[1])])
		return df_out

	def cv_lda(self, perf, vocab, vocab_weight, max_range, reverse = 1): # this function allows the cross validation to define the parameters of the LDA
		perf_param_lda = {}
		best_param_lda = []

		print('\nDéfinition du meilleur modèle & prédiction')

		if type(max_range)== int: # 2 possibilities, either give a list of max range or a single number and it will iterate over that number
			for k in range(1,max_range): # the problem here is to define the right alpha for the doc_topic_prior
				t0 = time()
				print('Progression: {}/{}'.format(k,len(range(1,max_range))))
				alpha = k/10
				model_parameters = {'n_components' : 2, 'n_top_words' : 500, 'doc_topic_prior':alpha} # we define the parameters to enter into the class perf
				df_precision_param, med_precision_param, best_seed, output = perf.operation_lda(model_parameters, vocab, vocab_weight, reverse) # perform the LDA with the right parameters
				best_param_lda.append(med_precision_param[0] + med_precision_param[1])  # gets the precision of the parameters as the sum of the median precision of the 100 iteration for the 2 categories
				perf_param_lda.update({k:{'precision_all_seed':df_precision_param, 'precision_param': med_precision_param, 'best_seed': best_seed, 'dic_output': output}})
				print("Réalisé en: %0.3fs" % (time() - t0))

		if type(max_range)==list: # same process with a list of alpha
			n = 0
			for k in max_range:
				t0 = time()
				n = n+1
				print('Progression: {}/{}'.format(n,len(max_range)))
				alpha = k
				model_parameters = {'n_components' : 2, 'n_top_words' : 500, 'doc_topic_prior':alpha}
				df_precision_param, med_precision_param, best_seed, output = perf.operation_lda(model_parameters, vocab, vocab_weight, reverse)
				best_param_lda.append(med_precision_param[0] + med_precision_param[1])
				perf_param_lda.update({n:{'precision_all_seed':df_precision_param, 'precision_param': med_precision_param, 'best_seed': best_seed, 'dic_output': output}})
				print("Réalisé en: %0.3fs" % (time() - t0))
		
		best_alpha = pandas.Series(best_param_lda).idxmax() + 1 # defines the best alpha
		best_model_lda = perf_param_lda[best_alpha]['dic_output'][perf_param_lda[best_alpha]['best_seed']]['lda_model'] # recovers the best model LDA
		best_topic_category_lda = perf_param_lda[best_alpha]['dic_output'][perf_param_lda[best_alpha]['best_seed']]['topic_to_category'] # and the corresponding topic - category dictionary
		return perf_param_lda, best_alpha, best_model_lda, best_topic_category_lda



	def operation_multicat(self, max_range):
		categorie = self.categories
		predict_all = {}
		predict_all_sc = {}
		multi_predict = {}
		result_all = {}
		for i in range(len(categorie)): 

			for j in range(i+1, len(categorie), 1):
				print('Les deux catégories en cours sont: {} et {}'.format(categorie[i], categorie[j])) # we process the operation for 2 categories in 2 sides cat0 - cat1 & cat1 - cat0
				duo = [[categorie[i], categorie[j]], [categorie[j], categorie[i]]]
				
				print('\n   Extraction du vocabulaire')
				
				vocab, vocab_weight = perf_model(self.df, duo[0], self.test_size).lexic_extractor() # get the lexic which is the same for the 2 sides

				full_test = self.df[['corpus_lemm','cat_struc']]
				full_test = full_test['corpus_lemm'] # defines the full sample

				full_predict = self.score_word(full_test, vocab_weight, duo[0]) # gets the score for words with the first side cat0 - cat1
				#full_predict.columns = [str(col) + '{}_{}'.format(i,j) for col in full_predict.columns]
				
				v = 0

				for cat in duo:
					print('\n   Définition des modèles dans le sens: {} - {}'.format(cat[0], cat[1]))
					v = v +1 

					perf = perf_model(self.df, cat, self.test_size) # defines the class to perform the LDA/NMF

					output_lda, best_alpha, best_model_lda, best_topic_category_lda = self.cv_lda(perf, vocab, vocab_weight, max_range, reverse = v) 
					# defines the best LDA model with the best alpha

					model_parameters = {'n_components' : 2, 'n_top_words' : 500, 'doc_topic_prior':1}
					_, _, best_seed_nmf, output_nmf = perf.operation_nmf(model_parameters, vocab, vocab_weight, reverse = v)
					# applies the NMF with no cross validation

					best_model_nmf = output_nmf[best_seed_nmf]['nmf_model']
					best_topic_category_nmf = output_nmf[best_seed_nmf]['topic_to_category']

					param = {'n_components' : 2, 'n_top_words' : 500, 'n_features': len(vocab), 'doc_topic_prior':best_alpha, 'vocabulary': vocab}
					model = model_lib.Models(**param)    
					
					full_predict_lda = pandas.DataFrame(model.reverse_lda(full_test.values.tolist(), best_model_lda), index =full_test.index, columns = ['{}_{}_lda'.format(best_topic_category_lda[0], v), '{}_{}_lda'.format(best_topic_category_lda[1], v)])
					# recovers the entire sample reversing the best LDA model (best alpha)	
					full_predict_nmf = pandas.DataFrame(model.reverse_nmf(full_test.values.tolist(), best_model_nmf), index =full_test.index, columns = ['{}_{}_nmf'.format(best_topic_category_nmf[0], v), '{}_{}_nmf'.format(best_topic_category_nmf[1], v)])
					# recovers the entire sample reversing the NMF model
					full_predict_models = full_predict_lda.join(full_predict_nmf)
					
					full_predict = full_predict.join(full_predict_models)
					# join the prediction of the LDA/NMF with the word score
					perf_param = {'LDA': {'best_alpha': best_alpha, 'dic_output': output_lda}, 'NMF': {'best_seed': best_seed_nmf, 'dic_output': output_nmf}}
					
					result_all.update({'{}_{}'.format(cat[0], cat[1]): perf_param})
					# the dictionary that registers all the result for every side of the duo of categories => we find here the best model
				predict_all.update({'{}_{}'.format(duo[0][0], duo[0][1]): full_predict})
				# this dictionary is used for the final prediction of the categories
				full_predict_sc = full_predict.copy()
				cols = list(full_predict.columns)
				full_predict_sc[cols] = StandardScaler().fit_transform(full_predict_sc[cols])
				# just get the same thing with a standardization of all the columns
				predict_all_sc.update({'{}_{}'.format(duo[0][0], duo[0][1]): full_predict_sc})

		print('\nPrecision des modèles toutes catégories')
		keys = list(predict_all.keys())
		_, predict_all_km = kmeans(predict_all, self.df).perform_km() # here we perform the kmeans to remove the score for the observation which are above the categories' max distance
		# !!! ATTENTION the kmeans model is not saved => it must be added as an output of the kmeans.perform_km() function
		# this last section only constructs dataframes to compute the performance
		df_predict_all = pandas.DataFrame(predict_all[keys[0]])
		df_predict_all_sc = pandas.DataFrame(predict_all_sc[keys[0]])
		df_predict_all_km = pandas.DataFrame(predict_all_km[keys[0]])
		# it takes all the dataframes registered in predict_all and its derivation (standardized & kmeans)
		keys.pop(0)
		k = 0
		for i in keys: # creates a dataframe with the prediction for all duos of categories side by side in every column
			k = k+1
			merge = pandas.DataFrame(predict_all[i])
			merge.columns = [str(col) + '_{}'.format(k) for col in merge.columns]
			df_predict_all = df_predict_all.join(merge)

			merge_sc = pandas.DataFrame(predict_all_sc[i])
			merge_sc.columns = [str(col) + '_{}'.format(k) for col in merge_sc.columns]
			df_predict_all_sc = df_predict_all_sc.join(merge_sc)

			merge_km = pandas.DataFrame(predict_all_km[i])
			merge_km.columns = [str(col) + '_{}'.format(k) for col in merge_km.columns]
			df_predict_all_km = df_predict_all_km.join(merge_km)
		# defines the columns which has the highest result which will be assigned as the predicted category (all column names have the corresponding categorie as title)
		df_predict_all['col_max'] = df_predict_all.idxmax(axis=1)
		df_predict_all_sc['col_max'] = df_predict_all_sc.idxmax(axis=1)
		df_predict_all_km['col_max'] = df_predict_all_km.idxmax(axis=1)

		def clean_col(text):
			text = re.sub(r'_[\d+]', '', text)
			text = re.sub(r'_(lda)', '', text)
			text = re.sub(r'_(nmf)', '', text)
			text = re.sub(r'_(sc)', '', text)
			return text
			# removes all the text around the name of the category
		df_predict_all['predict_cat'] = df_predict_all['col_max'].apply(clean_col)
		df_predict_all_sc['predict_cat'] = df_predict_all_sc['col_max'].apply(clean_col)
		df_predict_all_km['predict_cat'] = df_predict_all_km['col_max'].apply(clean_col)


		def true_positive(a,b):
			if a == b:
				return 1
			else:
				return 0
				# verifies if the predicted category corresponds to the true category
		df_predict_all = df_predict_all.join(self.df['cat_struc'])
		df_predict_all_sc = df_predict_all_sc.join(self.df['cat_struc'])
		df_predict_all_km = df_predict_all_km.join(self.df['cat_struc'])

		df_predict_all['precision'] = df_predict_all.apply(lambda x: true_positive(x['predict_cat'],x['cat_struc']), axis= 1)
		df_predict_all_sc['precision'] = df_predict_all_sc.apply(lambda x: true_positive(x['predict_cat'],x['cat_struc']), axis= 1)
		df_predict_all_km['precision'] = df_predict_all_km.apply(lambda x: true_positive(x['predict_cat'],x['cat_struc']), axis= 1)
		# compute the precision by counting the true positive for all the dataframe sample which have as true category the category which were input in the class
		precis_all_cat = df_predict_all[df_predict_all['cat_struc'].isin(categorie)]
		precis_all_cat_sc = df_predict_all_sc[df_predict_all_sc['cat_struc'].isin(categorie)]
		precis_all_cat_km = df_predict_all_km[df_predict_all_km['cat_struc'].isin(categorie)]


		final_precision = [precis_all_cat['precision'].sum()/len(precis_all_cat)*100, precis_all_cat_sc['precision'].sum()/len(precis_all_cat_sc)*100, precis_all_cat_km['precision'].sum()/len(precis_all_cat_km)*100]
		print('Résultats non standardisés {}; résultats standardisés {}; résultats avec K-Means {}'.format(final_precision[0], final_precision[1], final_precision[2]))

		return result_all, df_predict_all, df_predict_all_sc, df_predict_all_km, predict_all, predict_all_sc, predict_all_km

