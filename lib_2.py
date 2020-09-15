import pandas
import math
import collections
import re
import nltk
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import time
from nltk.corpus import wordnet

import requests
import urllib.parse

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np

class RefEnv(object):
	"""edit docs"""
	def __init__(self, path_PROJECT_DATA= '/Volumes/GoogleDrive/My Drive'):
		self.dst = 'Source'
		self.Export_folder = 'Export'
		self.path_PROJECT_DATA = path_PROJECT_DATA
		self.transformed = '{}/Open Data Platform/DATA/Transformed/{}'.format(self.path_PROJECT_DATA, '{}') ### how exactly is working .format(, '{}') ?

class Filter(object): ###allows to get df1_clean with 6 columns : 'categorie','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019' and key_main as index => this is X dataframe. The dataframe text gets only the presentation from df1_clean and the index
	"""docstring for RefEnv"""
	def __init__(self):
		self.X = pandas.read_csv(RefEnv().transformed.format('df1_clean.csv')) 
		self.text = self.operation_text()
		self.X = self.operation()

	
	def categorie_structure(self,cat_struc,cat_autre_struc):
		if cat_struc == "Santé, bien être, et protection sociale" or cat_struc == "Health, well-being and social protection":
			return "sante" 
		if cat_struc == "Fintech":
			return "finance"
		if cat_struc == "Technologies émergentes" or cat_struc == "Emerging technologies":
			return "technologie"
		if cat_struc == "Médias, édition et communication" or cat_struc =="Media, publishing and communication":
			return "media"
		if cat_struc == "Agriculture, territoires ruraux et ressources naturelles" or cat_struc == "Agriculture, rural areas and natural resources":
			return  "agriculture"
		if cat_struc == "Education, formation professionnelle et emploi" or cat_struc =="Education, vocational training and employment":
			return  "education" 
		if cat_struc == "E-commerce, marketing et distribution" or cat_struc == "E-commerce, marketing and distribution":
			return  "ecommerce"
		if cat_struc == "Gouvernement ouvert et transformation sociale" or cat_struc == "Open government and social transformation":
			return  "gouvernement"
		if cat_struc == "Energie" or cat_struc == "Energy":
			return  "energie" 
		if cat_struc == "Territoires intelligents et mobilité" or cat_struc == "Intelligent territories and mobility":
			return  "mobilite"
		if cat_struc == "Industries culturelles et créatives" or cat_struc == "Cultural and creative industries":
			return  "culture"
		if (cat_struc == "Autre (précisez)" or cat_struc == "Other - please specify" or cat_struc == "autre (précisez)"):
			return  'autres'
		if cat_struc != cat_struc: #what is it for ? 
			return 'non renseigné'
		else:
			return cat_struc
	
	def template(self, nom, prenom, email, struc, origine, activite, site, linkedin, prez, produit, marche, zone, objectif, innovant, duplicable, durable):
		description = 'Nom du répondant : {nom} \nPrénom du répondant : {prenom} \nemail : {email} \n \nNom de la structure : {struc} \nPays d\'origine : {origine} \nPays d\'activité : {activite} \nSite internet : {site} \nLinkedin : {linkedin} \n \nPrésentation de la structure \n{prez} \n \nPrésentation du produit, de la solution phare porté par la structure \n{produit} \n \nSur quel marché et vers quelle cible \n{marche} \n \nZone géographique de l’activité \n{zone} \n \nObjectifs de croissance et vision à moyen et long terme du marché \n{objectif} \n \nEn quoi votre solution est elle innovante \n{innovant} \n \nEn quoi est elle duplicable à moindre coût \n{duplicable} \n \nEn quoi votre solution rend les villes françaises et africaines plus durables du point de vue environnemental et social \n{durable} \n \n'.format(nom = nom, prenom = prenom, email = email, struc = struc, origine = origine, activite = activite, site = site, linkedin = linkedin, prez = prez, produit = produit, marche = marche, zone = zone, objectif = objectif, innovant = innovant, duplicable = duplicable, durable = durable)
		return description

	def operation_text(self):
		self.X['temp'] = self.X.apply(lambda x: self.template(x['nom_pers'], x['prenom_pers'], x['email_pers'],x['nom_struc'], x['pays_struc1'], x['pays_struc2'], x['site_struc'],x['linkedin_struc'], x['prez_struc'],x['prez_produit_struc'], x['prez_marche_struc'], x['prez_zone_struc'],x['prez_objectif_struc'], x['prez_innovante_struc'], x['prez_duplicable_struc'],x['prez_durable_struc']), axis=1)
		return self.X.set_index('key_main')[['cat_struc','prez_struc', 'prez_produit_struc','prez_marche_struc', 'prez_zone_struc', 'prez_objectif_struc','prez_innovante_struc', 'prez_duplicable_struc', 'prez_durable_struc', 'temp']]

	def operation(self):
		self.X['nbr_salarie'] = self.X[['empl_h_struc', 'empl_f_struc']].agg(sum, axis=1)
		self.X['ca_2019'] = self.X[['ca_2019_trim1', 'ca_2019_trim2', 'ca_2019_trim3']].agg(sum, axis=1)
		self.X['categorie'] = self.X.apply(lambda x: self.categorie_structure(x['cat_struc'], x['cat_autre_struc']), axis=1)
		return self.X.set_index('key_main')[['nom_struc','categorie','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019','pays_struc1','date_struc','prix_struc','linkedin_struc','email_pers']]

class Transformation(object): ### Takes the X dataframe from traduct.json => description translated in english. Then do all the preprocessing (lower case, removes numbers, punctuation and whitespaces)
	"""docstring for RefEnv"""
	def __init__(self):
		self.output = RefEnv().transformed.format('traduct.json')
		self.X = pandas.read_json(self.output).fillna('VIDE')#.set_index('key_main')
		self.header_tr = ['prez_struc', 'prez_produit_struc', 'prez_marche_struc', 'prez_zone_struc', 'prez_objectif_struc', 'prez_innovante_struc', 'prez_duplicable_struc', 'prez_durable_struc'] 
		self.X = self.operation()

	def lower_case(self, element):
		return element.lower()

	def number_to_text(self, element):
		return re.sub(r'\d+','', element) # why do we remove number and not convert it into text ? 

	def remove_punctuation(self, element):
		return element.translate(str.maketrans('', '', string.punctuation)) # we should change that to replace punctuation by a space. How str.maketrans is working exactly and why to we use translate ? 

	def remove_special_char(self, element):
		return element.replace('\r\n', ' ').replace('\t',' ').replace('/', ' ') #why don't we use the same synthax everywhere ? 

	def remove_whitespaces(self, element):
		return element.strip()

	def preprocesssing(self, element):
		element = self.remove_punctuation(element)
		element = self.remove_special_char(element)
		element = self.number_to_text(element)
		element = self.lower_case(element)    
		element = self.remove_whitespaces(element)
		return element
	
	def operation(self):
		self.X.columns = ['key_main', 'prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc'] #changes the name of the columns ? why do we have to do that ? 
		for head in self.header_tr:
			self.X['{}'.format(head)] = self.X[head].apply(self.preprocesssing) # I don't get what is done here and why do we do .apply(self.preprocesssing) and not solely .apply(preprocesssing) ?
		return self.X.set_index('key_main')
	
class Keyword_extraction(object): # returns a dataframe with new columns that are lemmatized with a suffix lemm but the data frame must contain the headers that are defined in header_tr 
	def __init__(self, X, file ='stopwords.txt'):
		self.X = X
		self.header_tr = ['prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc'] 
		self.stopwords_file = RefEnv().transformed.format(file)
		self.stop_words = self.get_set_stopwords()
		self.lemmatizer = WordNetLemmatizer()
		self.X = self.operation() # I don't really understand how it works? we define the dataframe first in the first line and then apply operation on it but why do we have to define it at the first line ? 

	def get_set_stopwords(self):
		fileName = self.stopwords_file
		stopword_file = open(fileName, 'r')
		list_stop_word = [line.split(',') for line in stopword_file.readlines()][0] # if I understand correctly this enables to get a list of all the stopwords that are in the stopword file spliting with , => why do we use readlines() and what the [0] stands for ? 
		return set(stopwords.words('english')).union(set(list_stop_word)) #here we have 2 sets => at the end can we use the set as a list ? I don't really get how we can use set. Is it only to make sure that we have a unique example for each word ? 
	
	def nltk2wn_tag(self, nltk_tag): 
		if nltk_tag.startswith('J'): #I don't get what is nltk_tag here, how do we get it ? it only checks if the value starts with J right ? 
			return wordnet.ADJ # Then what is the return ? only saying this is adjective ? 
		elif nltk_tag.startswith('V'):
			return wordnet.VERB
		elif nltk_tag.startswith('N'):
			return wordnet.NOUN
		elif nltk_tag.startswith('R'):
			return wordnet.ADV
		else:                    
			return None

	def is_stopword(self, element): #this function is not used ? 
		if element in self.stop_words:
			return True
		else:
			return False

	def lemmatize_sentence(self, sentence):

		nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))   # So here we tokenize the sentence (make a list of word in the text) and then for each word in the list we use the function pos_tag => for each word, returns a least with the word and its type
		wn_tagged = map(lambda x: (x[0], self.nltk2wn_tag(x[1])), nltk_tagged) # So here this is only a way to transform the tag after the word with a tag from wordnet ? what's the interest of doing that ? and why do why do self.nltk2wn_tag and not only nltk2wn_tag?

		res_words = []
		for word, tag in wn_tagged:
			if tag is None:
				res_words.append(word) #takes the word as it if there's no tag
			else:
				res_words.append(self.lemmatizer.lemmatize(word, tag)) # ok so we have to change the name of the tag because the lemmatizer doesn't understand the tags of pos_tag ? 
		sentence = " ".join([wrd for wrd in res_words if wrd not in self.stop_words]) # then we return the sentence lemmatized by joining all the words that are contained in res_word
		assert  'africa' not in sentence.split(' ') # is this just a check ? 
		return sentence#[wrd for wrd in ' '.join(sentence.split()) if self.is_stopword(wrd) is False]
	
	def operation(self):
		print(self.stopwords_file)
		for head in self.header_tr:
			self.X['{}_lemm'.format(head)] = self.X[head].apply(self.lemmatize_sentence) #what is the point of using all these self ? 
		return self.X

class Models(object):
	from time import time

	from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
	from sklearn.decomposition import NMF, LatentDirichletAllocation
	from sklearn.datasets import fetch_20newsgroups

	def __init__(self,to_exclude, file ='stopwords.txt',distance_NMF = 'Frobenius', n_samples = 2000, n_features = 1000, n_components = 20, n_top_words = 20):
		self.n_samples = n_samples
		self.n_features = n_features
		self.n_components = n_components
		self.n_top_words = n_top_words
		self.stopwords_file = RefEnv().transformed.format(file) 
		self.stop_words = self.get_set_stopwords()
		self.distance_NMF = distance_NMF
		self.header_left = ['categorie', 'age_pers', 'nbr_salarie', 'ca_2017', 'ca_2018', 'ca_2019','max_topic_frobenius']
		self.header_topic = ['Frobenius_topic_{}'.format(i) for i in range(0,self.n_components)]
		self.header = self.header_left + self.header_topic
		self.to_exclude = to_exclude


	def get_set_stopwords(self):
		fileName = self.stopwords_file
		stopword_file = open(fileName, 'r')
		list_stop_word = [line.split(',') for line in stopword_file.readlines()][0]
		return set(stopwords.words('english')).union(set(list_stop_word))

	def print_top_words(self, model, feature_names, n_top_words): # I don't get what this functions is for ? 
		for topic_idx, topic in enumerate(model.components_): #what is model.components_ where does it come from ? 
			message = "Topic #%d: " % topic_idx
			message += " ".join([feature_names[i]
								 for i in topic.argsort()[:-n_top_words - 1:-1]])
			print(message)
		print()

	def get_components(self, model, feature_names, n_top_words): # what's the difference between this function and the previous one? 
		components = {}
		for topic_idx, topic in enumerate(model.components_):
			message = {"Topic {}".format(topic_idx) :[feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]}
			components.update(message)
		return components

	def reverse_lda(self, X, models):
		model = models['LDA']['model']
		features  =  model['feature_names']
		vectorizer = model['vectorizer']
		model['tf_vectorizer'].set_params(vocabulary =  model['feature_names'])
		tf_x = vectorizer.fit_transform(X)
		predict = model.fit(LDA['tf']).transform(tf_x)
		max_topic = np.where(predict == max(predict[0]))[1][0]
		pdf = predict
		return max_topic

	def reverse_nmf(self, X , models):
		model = models['NMF_{}'.format(self.distance_NMF)]['model']
		data = models['NMF_{}'.format(self.distance_NMF)]['tf']
		model.fit(X = data)
		models['NMF_{}'.format(self.distance_NMF)]['vectorizer'].set_params(vocabulary = models['NMF_{}'.format(self.distance_NMF)]['feature_names']) 
		new_data = models['NMF_{}'.format(self.distance_NMF)]['vectorizer'].fit_transform(X)
		predict = model.transform(new_data)
		max_topic = np.where(predict == max(predict[0]))[1][0]
		return predict


	def prepare(self, to_exclude): # define the 2 dataframes of origine : X from Filter() () and X from Transformation()
		df_filter = Filter().X 
		df_filter = df_filter[~df_filter.index.isin(to_exclude)]

		text = Transformation().X
		text = text[~text.index.isin(to_exclude)]

		display(df_filter.describe(include = 'all'))
		return df_filter, text

	def get_NMF_results(self, cat, stopword_list):
		df_filter, text = self.prepare(self.to_exclude)
		results = dict()
		df_filter_category = df_filter[df_filter['categorie'] == cat]
		print(cat.upper())    
		subset = df_filter_category.join(text)
		subset_glob = df_filter.join(text)
		kwE = Keyword_extraction(X = subset.fillna('africa'), file = stopword_list)
		kwE_glob = Keyword_extraction(X = subset_glob.fillna('africa'), file = stopword_list)
		kw = kwE.X
		kw_glob = kwE_glob.X
		data_samples = kw[['prez_struc_lemm', 'prez_produit_struc_lemm', 'prez_marche_struc_lemm', 'prez_zone_struc_lemm', 'prez_objectif_struc_lemm', 'prez_innovante_struc_lemm', 'prez_duplicable_struc_lemm', 'prez_durable_struc_lemm']].agg(' '.join, axis=1)
		data_samples_glob = kw_glob[['prez_struc_lemm', 'prez_produit_struc_lemm', 'prez_marche_struc_lemm', 'prez_zone_struc_lemm', 'prez_objectif_struc_lemm', 'prez_innovante_struc_lemm', 'prez_duplicable_struc_lemm', 'prez_durable_struc_lemm']].agg(' '.join, axis=1)
		display(df_filter_category.describe(include = 'all'))
		#Models = Models(n_components = n_components,distance_NMF = distance_NMF, n_top_words = n_top_words, n_features = n_features)
		models = self.run_models(data_samples)
		components = models['NMF_Frobenius']['components']
		predict = self.reverse_nmf(data_samples, models)
		predict_glob = self.reverse_nmf(data_samples_glob, models)
		r = pandas.DataFrame(predict)
		r.columns = self.header_topic
		r = r.set_index(data_samples.index)

		r_glob = pandas.DataFrame(predict_glob)
		r_glob.columns = self.header_topic
		r_glob = r_glob.set_index(data_samples_glob.index)

		max_topic = pandas.Series([np.where(predict == max(record))[1][0] for record in predict])
		max_topic = max_topic.to_frame().set_index(data_samples.index)
		max_topic.columns = ['max_topic_frobenius']

		max_topic_glob = pandas.Series([np.where(predict_glob == max(record))[1][0] for record in predict_glob])
		max_topic_glob = max_topic_glob.to_frame().set_index(data_samples_glob.index)
		max_topic_glob.columns = ['max_topic_frobenius']

		kw_stats = pandas.concat([kw, r, max_topic], axis=1)[self.header]
		display(kw_stats.groupby('max_topic_frobenius').count()[['categorie','age_pers']])
		kw_stats_glob = pandas.concat([kw_glob, r_glob, max_topic_glob], axis=1)[self.header]
		
		results['kw'] = kw
		results['data_samples'] = data_samples
		results['model'] = models['NMF_Frobenius']
		results['kw_stats'] = kw_stats
		results['df'] = kw_stats.join(Filter().text)
		results['all'] = kw_stats_glob.join(Filter().text)
		return results   
	
	def run_models(self, data_samples):
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
			  % (self.n_samples, self.n_features)) # how does % works to fill missing text ? 
		t0 = time()
		nmf_frobenius = NMF(n_components=self.n_components, random_state=1,
				  alpha=.1, l1_ratio=.5).fit(tfidf)
		print("done in %0.3fs." % (time() - t0)) # explain how this is print too

		#print("\nTopics in NMF model (Frobenius norm):")
		tfidf_feature_names = tfidf_vectorizer.get_feature_names() # here i don't get how it works
		self.print_top_words(nmf_frobenius, tfidf_feature_names, self.n_top_words) # here neither

		# Fit the NMF model
		#print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
		#     "tf-idf features, n_samples=%d and n_features=%d..."
		#     % (self.n_samples, self.n_features))
		#t0 = time()
		#nmf_kullback = NMF(n_components=self.n_components, random_state=1,
		#		  beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
		#		  l1_ratio=.5).fit(tfidf)
		#print("done in %0.3fs." % (time() - t0))

		#print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
		#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
		#self.print_top_words(nmf_kullback, tfidf_feature_names, self.n_top_words)

		#print("Fitting LDA models with tf features, "
		#     "n_samples=%d and n_features=%d..."
		#     % (self.n_samples, self.n_features))
		#lda = LatentDirichletAllocation(n_components=self.n_components, max_iter=5,
		#								learning_method='online',
		#								learning_offset=50.,
		#								random_state=0)
		#t0 = time()
		#lda = lda.fit(tf)
		#print("done in %0.3fs." % (time() - t0))

		#print("\nTopics in LDA model:")
		#tf_feature_names = tf_vectorizer.get_feature_names()
		#self.print_top_words(lda, tf_feature_names, self.n_top_words)

		models = {}
		#models['NMF_Kullback'] = {'model':nmf_frobenius, 'feature_names':tfidf_feature_names, 'tf':tfidf, 'vectorizer': tfidf_vectorizer, 'components':self.get_components(nmf_frobenius,tfidf_feature_names,self.n_top_words)}
		models['NMF_Frobenius'] = {'model':nmf_frobenius, 'feature_names':tfidf_feature_names, 'tf':tfidf, 'vectorizer': tfidf_vectorizer, 'components':self.get_components(nmf_frobenius,tfidf_feature_names,self.n_top_words)}
		#models['LDA'] = {'model':lda, 'feature_names':tf_feature_names, 'tf':tf, 'vectorizer':tf_vectorizer, 'components':self.get_components(lda,tfidf_feature_names,self.n_top_words)}

		return models