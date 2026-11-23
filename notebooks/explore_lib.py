import sys; sys.path.insert(0,'src')
import pandas as pd, numpy as np
from celeris.clean.normalize import *
def load_norm():
    tr=pd.read_csv('student_data/train.csv',dtype=str)
    tr['date']=map_unique(tr.Date,parse_month)
    tr['sku']=map_unique(tr.SKU_ID,normalize_sku)
    tr['region']=map_unique(tr.Region,normalize_region)
    tr['channel']=map_unique(tr.Sales_Channel,normalize_channel)
    tr['customer']=map_unique(tr.Customer_Name,lambda v:normalize_customer(v)[0])
    tr['units']=to_number(tr.Units_Sold); tr['price']=to_number(tr.Unit_Price_USD); tr['rev']=to_number(tr.Revenue_USD)
    tr['ts']=pd.to_datetime(tr.Load_Timestamp)
    tr=tr.sort_values('ts').drop_duplicates('Record_ID',keep='last')
    return tr
