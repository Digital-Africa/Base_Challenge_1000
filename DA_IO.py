# coding: utf-8
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas
import json

class Gspreadsheet(object):
	"""docstring for Gspreadsheet"""
	def __init__(self, name, service_account = "/Users/Shared/digital-africa/web-services-d4-digital-africa-2eb794c55b04.json"):
		super(Gspreadsheet, self).__init__()
		self.name = name
		self.service_account = service_account
		self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
		self.credentials = ServiceAccountCredentials.from_json_keyfile_name(self.service_account, self.scope)
		self.print_creds = self.print_service_account()
		self.gc = gspread.authorize(self.credentials)
		try:
			self.sheet = self.gc.open(self.name)
		except:
			pass

	def sheet_to_df(self,sheet_name = None):
		try:
			wks = self.sheet.worksheet(sheet_name)
		except:
			wks = self.sheet.get_worksheet(0)
		data = wks.get_all_values()
		headers = data.pop(0)
		df = pandas.DataFrame(data, columns=headers)#.set_index('key_main')
		return df

	def get_schema(self, uri):
	    with open(uri, 'r') as s:
	        schema = json.load(s)
	    return schema

	def df_to_sheet(self, df, sheet_name = None):
		try:
			wks = self.sheet.worksheet(sheet_name)
		except:
			wks = self.sheet.get_worksheet(0)
		headers = [df.columns.values.tolist()]
		data = df.values.tolist()
		response = wks.update(headers + data)
		print('https://docs.google.com/spreadsheets/d/{spreadsheetId}/edit#gid=0'.format(spreadsheetId = response['spreadsheetId']))
	
	def print_service_account(self):
		print('Project ID: ', self.get_schema(self.service_account)['project_id'])
		print('Service Account Email: ', self.get_schema(self.service_account)['client_email'])

	def list_sheets(self):
		return [e for e in self.sheet.worksheets()]

	def create(self):
		self.gc.create(self.name)
		try:
			self.sheet.share(self.get_schema(self.service_account)['client_email'], perm_type ='user', role = 'writer')
			self.sheet.share('mohamed.diabakhate@digital-africa.co', perm_type ='user', role = 'writer')
		except:
			sheet = self.gc.open(self.name)
			sheet.share(self.get_schema(self.service_account)['client_email'], perm_type ='user', role = 'writer')
			sheet.share('mohamed.diabakhate@digital-africa.co', perm_type ='user', role = 'writer')

	def add_sheet(self,sheet_name):
		self.sheet.add_worksheet(title = sheet_name, rows = 100, cols=20)
