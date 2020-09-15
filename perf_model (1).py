class perf_model(object):
	import model_lib
	import chalenge1000
	import numpy as np
	import collections
	import functools
	import operator
	from operator import itemgetter
	import pandas
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from IPython.display import display, HTML
	from statistics import median
	import time
	import progressbar
	"""docstring for perf_model"""
	def __init__(self, categories, test_size, silent=True):
		self.categories = categories
		self.test_size = test_size
		self.df = self.get_df()
		self.silent = silent


	def get_df(self):
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

	def lexic_extractor(self):
		lex1 = collections.Counter(self.df[self.df['cat_struc']== self.categories[0]]['lexic_lemm'].agg(sum, axis=0).split()).most_common(5000)
		lex2 = collections.Counter(self.df[self.df['cat_struc']== self.categories[1]]['lexic_lemm'].agg(sum, axis=0).split()).most_common(5000)

		len_cat1 = len(self.df[self.df['cat_struc']==self.categories[0]])
		len_cat2 = len(self.df[self.df['cat_struc']==self.categories[1]])
    
		counter_cat1 = list({(a,b*100/len_cat1) for a,b in lex1}.union({(k,0) for k,l in lex2 if k not in [a for a,b in lex1]}))
		counter_cat2 = list({(a,b*100/len_cat2) for a,b in lex2}.union({(k,0) for k,l in lex1 if k not in [a for a,b in lex2]}))

		vocab = [a for a,b in counter_cat1]
    
		id = operator.itemgetter(0)
    
		weight_cat1 = self.df[self.df['cat_struc']== self.categories[0]]['word_weight']
		weight_cat2 = self.df[self.df['cat_struc']== self.categories[1]]['word_weight']
    
		vocab_weight_cat1 = [(key, value/len_cat1) for key, value in dict(functools.reduce(operator.add, map(collections.Counter, weight_cat1))).items()]
		vocab_weight_cat1  = list({i for i in vocab_weight_cat1 if id(i) in vocab}.union({(k, 0) for k in vocab if k not in [a for a,b in vocab_weight_cat1]}))

		vocab_weight_cat2 = [(key, value/len_cat2) for key, value in dict(functools.reduce(operator.add, map(collections.Counter, weight_cat2))).items()]
		vocab_weight_cat2  = list({i for i in vocab_weight_cat2 if id(i) in vocab}.union({(k, 0) for k in vocab if k not in [a for a,b in vocab_weight_cat2]}))
    
		id_counter_cat2 = {id(rec): rec[1:] for rec in counter_cat2}
		id_vocab_weight_cat1 = {id(rec): rec[1:] for rec in vocab_weight_cat1}
		id_vocab_weight_cat2 = {id(rec): rec[1:] for rec in vocab_weight_cat2}
    
		merged = [i + id_counter_cat2[id(i)] for i in counter_cat1 if id(i) in id_counter_cat2]
		merged = [i + id_vocab_weight_cat1[id(i)] for i in merged if id(i) in id_vocab_weight_cat1]
		merged = [i + id_vocab_weight_cat2[id(i)] for i in merged if id(i) in id_vocab_weight_cat2]

		keyword_map = pandas.DataFrame(merged, columns = ['keyword','cat1', 'cat2', 'weight_cat1', 'weight_cat2'])
    
		def produit(x,y):
			return x*(y+1)
    
		keyword_map['score_cat1'] = keyword_map.apply(lambda x: produit(x['cat1'], x['weight_cat1']), axis = 1)
		keyword_map['score_cat2'] = keyword_map.apply(lambda x: produit(x['cat2'], x['weight_cat2']), axis = 1)
		keyword_map['difference'] = abs(keyword_map['score_cat1'] - keyword_map['score_cat2'])
		keyword_map.sort_values('difference', ascending = False)
    
		#percentile = pandas.DataFrame(keyword_map['difference'].quantile(np.linspace(.01, 1, 99, 00)))
		#percentile['lag_diff'] = percentile.diff(axis = 0)
    
		pct = []
		for i in range(1, 100):
			i = i/100
			j = i -0.01
			temp = keyword_map['difference'].quantile(i)
			diff = temp - keyword_map['difference'].quantile(j)
			if i ==0.01: 
				pct.append([i, temp, abs(temp)/2])
			else: 
				pct.append([i, temp, diff])
		pct1 = pandas.DataFrame(pct, columns=['a', 'b', 'c'])
		
		if self.silent == False:
			print(plt.scatter(pct1.index, pct1['c']))
    
		select = pct1[pct1['c']>pct1['c'].mean()]['a'].values.tolist()
		
		if self.silent == False:
			print(select)
    
		def cible(liste):
			for i in liste:
				if i<=0.5:
					next
				else:
					return i
					break
    
		quant = cible(select)

		if self.silent == False:
			print('Quantile sélectionné :', quant)
    
		lexic_model = [i for i in keyword_map[keyword_map['difference']>keyword_map['difference'].quantile(quant)]['keyword']]    
		if self.silent == False:
			print('Longueur du lexique :',len(lexic_model))
    
		dic_word_weight = keyword_map[['keyword', 'score_cat1', 'score_cat2']].set_index('keyword').T.to_dict('list')
    
		return lexic_model, dic_word_weight

	def train_test(self, seed):
		text_corpus_label = self.df[self.df['cat_struc'].isin(self.categories)][['corpus_lemm','cat_struc']]  
		test,train = train_test_split(text_corpus_label, test_size = self.test_size, random_state = seed)
		train = train['corpus_lemm']
		test = test['corpus_lemm']		
		return train, test

	def perf_lda(self, train, test, model_param):
		model = model_lib.Models(**model_param)    
		models = model.run_model_LDA(train.values.tolist())
		predict = pandas.DataFrame(model.reverse_lda(test.values.tolist(), models) , index =test.index)
		#display(predict)
		return models['LDA'], predict, models['LDA']['components']

	def topic_to_category(self, components, vocab_weight):
		topic_category = {}
		for i in range(2):
			words = components['Topic {}'.format(i)]
			cat1 = []
			cat2 = []
			for j in words: 
				if j in vocab_weight.keys():
					cat1.append(vocab_weight[j][0])
					cat2.append(vocab_weight[j][1])
			cat1 = sum(cat1)
			cat2 = sum(cat2)
			topic_category[i] = cat1 - cat2
    
		if topic_category[0]>topic_category[1]:
			topic_category[0]= self.categories[0]
			topic_category[1]= self.categories[1]
		else:
			topic_category[0]= self.categories[1]
			topic_category[1]= self.categories[0]
        
		return topic_category

	def precision(self, predict, topic_category):
    
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

		cat0 = precision[precision['cat_struc'] == self.categories[0]]
		cat1 = precision[precision['cat_struc'] == self.categories[1]]

		vrai_positif_cat0 = cat0['classification_kpi'].sum()/len(cat0)*100
		vrai_positif_cat1 = cat1['classification_kpi'].sum()/len(cat1)*100

		#print(model_parameters)
    
		if self.silent == False:
			print('\nPrécision vrai positif catégorie {}: {}'.format(categories[0], vrai_positif_cat0))
    
		if self.silent == False:
			print('\nPrécision vrai positif catégorie {}: {}'.format(categories[1], vrai_positif_cat1))
    
		#model_parameters.update({'precision':agri['classification_kpi'].sum()/len(agri)*100, 'vocabulaire': models['LDA']['feature_names']})
		#errors = agri[agri['classification_kpi'] == 0].sort_values('topic')
		#collections.Counter(' '.join(errors.join(text_corpus)['collection_counter'].values.tolist()).split()).most_common(10)
		#return model_parameters, agri, other, models['LDA']
    
		return precision, vrai_positif_cat0, vrai_positif_cat1



	def operation(self, model_param, vocab, vocab_weight):
		#vocab, vocab_weight = self.lexic_extractor()
		model_param.update({'vocabulary': vocab, 'n_features': len(vocab)})
		#model_param['vocabulary'] = vocab
		#model_param['n_features'] = len(vocab)

		precision_param = []
		output = {}

		bar = progressbar.ProgressBar(maxval=100).start()

		for i in range(100):
			if self.silent == False:
				print('\n\nItération numéro:', i)
			train, test = self.train_test(i)
			models, predict, components = self.perf_lda(train, test, model_param)
			topic_category = self.topic_to_category(components, vocab_weight)
			precis, vrai_positif_cat0, vrai_positif_cat1 = self.precision(predict, topic_category)
			precision_param.append([vrai_positif_cat0, vrai_positif_cat1])
			output.update({i: [train, test, models, precis]})
			bar.update(i)

		med_precision_param = [median([a for a,b in precision_param]), median([b for a,b in precision_param])]
		print('\nPrécision médiane des paramètres',med_precision_param)

		df_precision_param = pandas.DataFrame(precision_param, columns = ['cat0', 'cat1'])
		df_precision_param['sum'] =  df_precision_param[['cat0', 'cat1']].agg(sum, axis = 1)
		best_seed = df_precision_param[df_precision_param['cat0']>df_precision_param['cat0'].quantile(0.90)]['sum'].idxmax()
		print('\nModèle le plus performant',best_seed)

		return df_precision_param, med_precision_param, best_seed, output 
		