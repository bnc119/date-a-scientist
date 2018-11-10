import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter


#Create your df here:
df  = pd.read_csv("profiles.csv")


body_mapping = {
"thin" : 0,
"skinny" : 0,
"average": 0,
"fit" : 0,
"athletic" :0,
"jacked" :1 ,
"curvy" : 1,
"a little extra" : 1,
"full figured" : 1,
"overweight" : 1,
"used up" : 1,
"rather not say" : 1 }

# add the body code, assume "unhealthy" body type if not filled in
df["body_code"] = df.body_type.map(body_mapping)
df["body_code"].fillna(1, inplace=True)


plt.hist(df.age, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()


plt.figure()


plt.hist(df.body_type, bins=5)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.xlim(0, 10)
plt.show()



df.height.describe()

# 68.3 is mean
df.height.fillna(68.3, inplace=True)


n, bins, patches = plt.hist(df.height)
plt.xlabel("Height")
plt.ylabel("Frequency")
plt.show()


plt.scatter(df.age, df.height, alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Height")
plt.show()


df.body_type.fillna("average", inplace=True)
df.body_type.isnull().any()

body_counts = Counter(df.body_type)
df = pd.DataFrame.from_dict(body_counts, orient='index')
df.plot(kind='bar', title="Body type bar chart", legend=False)



#drop rows that unanswered information for drinking, smoking, or drug use
df.dropna(subset = ['drinks', 'smokes', 'drugs'], inplace=True)

drink_counts = Counter(df.drinks)
d = pd.DataFrame.from_dict(drink_counts, orient='index')
d.plot(kind='bar', title="Alcohol consumption bar chart", legend=False)


smoke_counts = Counter(df.smokes)
e = pd.DataFrame.from_dict(smoke_counts, orient='index')
e.plot(kind='bar', title="Tobacco usage bar chart", legend=False)


drug_counts = Counter(df.drugs)
f = pd.DataFrame.from_dict(drug_counts, orient='index')
f.plot(kind='bar', title="Drug usage bar chart", legend=False)
