from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

filefullpath = r"123.xlsx"
df = pd.read_excel(filefullpath)
sentences = df['中医疾病']
addrs = []
ak = []
for i in sentences:
    a = i.split(",")
    for k in a:
        #print (k)
        ak.append(k)
        if not k in addrs :
            addrs.append(k)
print (len(addrs))
print (len(ak))
# #p = addrs.sort()
# for i in sorted(k):
#     print (i)