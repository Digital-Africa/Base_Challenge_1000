from oauth2client.service_account import ServiceAccountCredentials
import gspread
import pandas
import json

class Gspreadsheet(object):
	"""docstring for Gspreadsheet"""
	def __init__(self, service_account = "/Users/Shared/digital-africa/web-services-d4-digital-africa-2eb794c55b04.json"):
		super(Gspreadsheet, self).__init__()
		self.service_account = service_account
		self.scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
		self.credentials = ServiceAccountCredentials.from_json_keyfile_name(self.service_account, self.scope)
		self.print_creds = self.print_service_account()

	def sheet_to_df(self, name):
		gc = gspread.authorize(self.credentials)
		wks = gc.open(name).sheet1
		data = wks.get_all_values()
		headers = data.pop(0)
		df = pandas.DataFrame(data, columns=headers)#.set_index('key_main')
		return df

	def get_schema(self, uri):
	    with open(uri, 'r') as s:
	        schema = json.load(s)
	    return schema

	def print_service_account(self):
		print('Project ID: ', self.get_schema(self.service_account)['project_id'])
		print('Service Account Email: ', self.get_schema(self.service_account)['client_email'])