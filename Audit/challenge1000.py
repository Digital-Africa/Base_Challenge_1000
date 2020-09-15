# coding: utf-8
import pandas
import collections
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

#import unidecode
import http.client

import requests
import urllib.parse

from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import json


class RefEnv(object):
	"""edit docs"""
	def __init__(self, path_PROJECT_DATA= '/Volumes/GoogleDrive/My Drive/Open Data Platform/', uri = 'star_schema.json'):
		self.dst = 'Source'
		self.Export_folder = 'Export'
		self.path_PROJECT_DATA = path_PROJECT_DATA
		self.transformed = '{}DATA/Transformed/{}'.format(self.path_PROJECT_DATA, '{}') ### how exactly is working .format(, '{}') ?
		self.uri = '{}DATA/SCHEMA/{}'.format(self.path_PROJECT_DATA,uri)
		self.schema = self.get_schema()

	def get_schema(self):
		with open(self.uri, 'r') as s:
			schema = json.load(s)
		return schema

class Native_1(object): 
	"""docstring for RefEnv"""
	def __init__(self):
		self.X = pandas.read_csv(RefEnv().transformed.format('df_new.csv')).set_index('key_main')
		self.raw = self.X
		self.X = self.operation()
		self.descriptions = self.X[['nom_struc', 'prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc']]
		self.descriptions_trad = self.get_trans()
		self.template = self.X[['cat_struc', 'temp']]
		self.df = self.X[['nom_pers', 'prenom_pers', 'email_pers', 'nom_struc', 'site_struc', 'cat_struc', 'status', 'date_struc', 'age','pays_struc1','empl_struc','ca_2017','ca_2018','ca_2019','prix_struc','linkedin_struc']]
		#self.X = self.X[['nom_struc','categorie','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019','pays_struc1','date_struc','prix_struc','linkedin_struc','email_pers']]

	def get_trans(self):
		frame = pandas.read_json(RefEnv().transformed.format('traduct.json')).set_index('key_main')
		frame.columns = ['prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc']
		return frame
	
	def template(self, nom, prenom, email, struc, origine, activite, site, linkedin, prez, produit, marche, zone, objectif, innovant, duplicable, durable):
		description = 'Nom du répondant : {nom} \nPrénom du répondant : {prenom} \nemail : {email} \n \nNom de la structure : {struc} \nPays d\'origine : {origine} \nPays d\'activité : {activite} \nSite internet : {site} \nLinkedin : {linkedin} \n \nPrésentation de la structure \n{prez} \n \nPrésentation du produit, de la solution phare porté par la structure \n{produit} \n \nSur quel marché et vers quelle cible \n{marche} \n \nZone géographique de l’activité \n{zone} \n \nObjectifs de croissance et vision à moyen et long terme du marché \n{objectif} \n \nEn quoi votre solution est elle innovante \n{innovant} \n \nEn quoi est elle duplicable à moindre coût \n{duplicable} \n \nEn quoi votre solution rend les villes françaises et africaines plus durables du point de vue environnemental et social \n{durable} \n \n'.format(nom = nom, prenom = prenom, email = email, struc = struc, origine = origine, activite = activite, site = site, linkedin = linkedin, prez = prez, produit = produit, marche = marche, zone = zone, objectif = objectif, innovant = innovant, duplicable = duplicable, durable = durable)
		return description

	def check_site(self, site):
		try:
			if site.startswith('https:'):
				text = re.sub(r'https://', '', site)
				text = re.sub(r'www.', '', text)
				text = re.sub(r'/$', '', text)
				conn = http.client.HTTPSConnection(text)

			elif site.startswith('http:'):
				text = re.sub(r'http://', '', site)
				text = re.sub(r'www.', '', text)
				text = re.sub(r'/$', '', text)
				conn = http.client.HTTPConnection(text)
			
			elif site.startswith('www.'):
				text=re.sub(r'www.', '', site)
				text = re.sub(r'/$', '', text)
				conn = http.client.HTTPConnection(text)
				conn.request("GET", "/")
				rep = conn.getresponse().status
				if rep == 301:
					conn = http.client.HTTPSConnection(text)
			
			conn.request("GET", "/")
			rep = conn.getresponse().status
			return rep
			
		except:
			return 'error'

	def check_neg(self, text):
		if text!= text:
			return 0
		else:
			text = str(text)
			text = unidecode.unidecode(re.sub('[^\w ]','',text.lower()))
			if re.search(r'\b(non)\b|\b(no)\b|\b(ne)\b|\b(pas)\b|\b(not)\b|\b(none)\b', text):
				return 2
			elif len(text)<10:
				return 3
			else:
				return 1

	def grow(self, x, y):
		if y != 0:
			return (x - y)/y*100
		else:
			return float('NaN')

	def filtre(self, narg, col, cond, value, select='null'):
		df=self.X
		if narg==1:
			if cond=='eq':
				df = df[df[col].isin(value)]
			elif cond=='leq':
				df = df[df[col] <= value]
			elif cond=='geq':
				df = df[df[col] >= value]
			elif cond=='inf':
				df = df[df[col] < value]
			elif cond=='sup':
				df = df[df[col] > value]
		elif narg>1:
			for i in range(narg):
				if cond[i]=='eq':
					df = df[df[col[i]].isin(value[i])]
				elif cond[i]=='leq':
					df = df[df[col[i]] <= value[i]]
				elif cond[i]=='geq':
					df = df[df[col[i]] >= value[i]]
				elif cond[i]=='inf':
					df = df[df[col[i]] < value[i]]
				elif cond[i]=='sup':
					df = df[df[col[i]] > value[i]]
		if select=='null':
			return df
		else:
			return df[select]

	def operation(self):
		self.X['temp'] = self.X.apply(lambda x: self.template(x['nom_pers'], x['prenom_pers'], x['email_pers'],x['nom_struc'], x['pays_struc1'], x['pays_struc2'], x['site_struc'],x['linkedin_struc'], x['prez_struc'],x['prez_produit_struc'], x['prez_marche_struc'], x['prez_zone_struc'],x['prez_objectif_struc'], x['prez_innovante_struc'], x['prez_duplicable_struc'],x['prez_durable_struc']), axis=1)
		self.X['site_status'] = self.X['site_struc'].apply(self.check_site)
		self.X['check_invest'] = self.X['investisseur_struc'].apply(self.check_neg)
		self.X['check_prix'] = self.X['prix_struc'].apply(self.check_neg)
		self.X['check_incub'] = self.X['incubateur_struc'].apply(self.check_neg)
		self.X['ca_grow_1819'] = self.X.apply(lambda x: self.grow(x['ca_2019'],x['ca_2018']), axis=1)
		self.X['ca_grow_1718'] = self.X.apply(lambda x: self.grow(x['ca_2018'],x['ca_2017']), axis=1)
		return self.X
		
class Native(object): ###allows to get df1_clean with 6 columns : 'categorie','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019' and key_main as index => this is X dataframe. The dataframe text gets only the presentation from df1_clean and the index
	"""docstring for RefEnv"""
	def __init__(self):
		self.X = pandas.read_csv(RefEnv().transformed.format('df1_clean.csv'))
		self.raw = self.X
		self.ALL = self.operation().set_index('key_main') 
		self.X = self.operation().set_index('key_main')#[['nom_struc','categorie','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019','pays_struc1','date_struc','prix_struc','linkedin_struc','email_pers']] 
		self.descriptions = self.operation()[['nom_struc','prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc','categorie']]
		self.descriptions_trad = self.get_trans()
		self.text = self.operation_text()
		self.X = self.X[['nom_struc','categorie', 'cat_struc', 'cat_autre_struc','age_pers','nbr_salarie','ca_2017','ca_2018','ca_2019','pays_struc1','pays_struc2','date_struc','prix_struc','linkedin_struc','email_pers']]

	def get_trans(self):
		frame = pandas.read_json(RefEnv().transformed.format('traduct.json')).set_index('key_main')
		frame.columns = ['prez_struc', 'prez_produit_struc','prez_marche_struc','prez_zone_struc','prez_objectif_struc', 'prez_innovante_struc','prez_duplicable_struc', 'prez_durable_struc']
		return frame
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
		return self.X[['cat_struc','prez_struc', 'prez_produit_struc','prez_marche_struc', 'prez_zone_struc', 'prez_objectif_struc','prez_innovante_struc', 'prez_duplicable_struc', 'prez_durable_struc', 'temp']]

	def operation(self):
		self.X['nbr_salarie'] = self.X[['empl_h_struc', 'empl_f_struc']].agg(sum, axis=1)
		self.X['ca_2019'] = self.X[['ca_2019_trim1', 'ca_2019_trim2', 'ca_2019_trim3']].agg(sum, axis=1)
		self.X['categorie'] = self.X.apply(lambda x: self.categorie_structure(x['cat_struc'], x['cat_autre_struc']), axis=1)
		return self.X