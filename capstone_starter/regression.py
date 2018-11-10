import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC


#Create your data frame here:
df  = pd.read_csv("profiles.csv")

# Setup a binary classification labels for body type
# 0: healthy (positive body image), 1: unhealthy, (negative body image)
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


#drop rows that unanswered information for drinking, smoking, or drug use
df.dropna(subset = ['drinks', 'smokes', 'drugs'], inplace=True)

# Perform drink mapping
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drink_codes"] = df.drinks.map(drink_mapping)

# Perform smoke mapping
smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smoke_codes"] = df.smokes.map(smoke_mapping)

# Perform drug mapping
drug_mapping = {"never": 0, "sometimes": 1, "often": 2 }
df["drug_codes"] = df.drugs.map(drug_mapping)

# Perform sex mapping
sex_mapping = {"m": 0, "f": 1}
df["sex_codes"] = df.sex.map(sex_mapping)


# Predict a continuous value - height, based on lifestyle factors
feature_data = df[['age','sex_codes', 'smoke_codes', 'drink_codes', 'drug_codes']]
#x = feature_data.values
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
x_train, x_test, y_train, y_test = train_test_split(feature_data, df.height, train_size = 0.8, test_size = 0.2, random_state=6)


# Lets use linear regression first
mlr = LinearRegression()
mlr.fit(x_train, y_train)

print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))

y_predicted = mlr.predict(x_test)
mean_squared_error(y_test, y_predicted)
r2_score(y_test, y_predicted)


# Now, let's use the KNN regressor
knr = KNeighborsRegressor()
knr.fit(x_train, y_train)
print(knr.score(x_train, y_train))
print(knr.score(x_test, y_test))

y_predk = knr.predict(x_test)
mean_squared_error(y_test, y_predk)
r2_score(y_test, y_predk)
