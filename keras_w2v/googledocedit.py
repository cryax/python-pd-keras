# -*- coding: utf-8 -*-
import gspread
from oauth2client.service_account import ServiceAccountCredentials
 
 
# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds']
creds = ServiceAccountCredentials.from_json_keyfile_name('sentiment incremental-f71d34189dbc.json', scope)
client = gspread.authorize(creds)
 
# Find a workbook by name and open the first sheet
# Make sure you use the right name here.
wks = client.open("sentiment").sheet1
mylist = ['A','B','C','D']
for index,value in enumerate(mylist):
    cellname = 'A'+str(index+1)
    wks.update_acell(cellname, value)
# Extract and print all of the values
#list_of_hashes = sheet.get_all_records()
#print(list_of_hashes)

#data =['a','b','c']
#vale =[1,2,3]
##sheet.cell(1, 1).value
##for index,value in enumerate(data):
#sheet.insert_row('a',1)