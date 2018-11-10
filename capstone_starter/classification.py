import itertools
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

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

# Perform the smoke mapping
smoke_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smoke_codes"] = df.smokes.map(smoke_mapping)

# Perform the drug mapping
drug_mapping = {"never": 0, "sometimes": 1, "often": 2 }
df["drug_codes"] = df.drugs.map(drug_mapping)

# Perform the sex mapping
sex_mapping = {"m": 0, "f": 1}
df["sex_codes"] = df.sex.map(sex_mapping)

# Define a function that prints a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Classify body type, based on lifestyle factors
feature_data = df[['age','smoke_codes', 'drink_codes', 'drug_codes']]
#x = feature_data.values
#min_max_scaler = preprocessing.MinMaxScaler()
#x_scaled = min_max_scaler.fit_transform(x)
#feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)
x_train, x_test, y_train, y_test = train_test_split(feature_data, df.body_code, train_size = 0.8, test_size = 0.2, random_state=6)


# Train the KKN classifier on k=1 thru 20, and plot the results
neighbors = np.arange(1,20)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    #Fit the model
    knn.fit(x_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)

    #Compute accuracy on the test set
    test_accuracy[i] = knn.score(x_test, y_test)



plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy', marker='.')
plt.plot(neighbors, train_accuracy, label='Training accuracy', marker='.')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()



# Run the classifier for k=10
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
knn.score(x_train, y_train)
knn.score(x_test,y_test)
y_predicted = knn.predict(x_test)

target_names = ["0: healthy", "1: unhealthy" ]

print(classification_report(y_test,y_predicted,target_names=target_names))

cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

plt.show()


#
# OK, now let's try a second classification technique - Naive Bayes
#

bayes = MultinomialNB()
bayes.fit(x_train, y_train)

y_predb = bayes.predict(x_test)
print(classification_report(y_test,y_predb,target_names=target_names))



# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predb)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

plt.show()



# try the random forest classifier, which is a tree-based classifer that performs well on unbalanced classes
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
y_predtree = rfc.predict(x_test)
print(classification_report(y_test,y_predtree,target_names=target_names))

# Separate majority and minority classes
df_majority = df[df.body_code==0]
df_minority = df[df.body_code==1]

len(df_majority)
len(df_minority)

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=32683,    # to match majority class
                                 random_state=123) # reproducible results

len(df_minority_upsampled)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
len(df_upsampled)

df_upsampled.body_code.value_counts()

feature_data = df_upsampled[['age','smoke_codes', 'drink_codes', 'drug_codes']]
x_train, x_test, y_train, y_test = train_test_split(feature_data, df_upsampled.body_code, train_size = 0.8, test_size = 0.2, random_state=6)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
knn.score(x_train, y_train)
knn.score(x_test,y_test)
y_predicted = knn.predict(x_test)

target_names = ["0: healthy", "1: unhealthy" ]

print(classification_report(y_test,y_predicted,target_names=target_names))


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

plt.show()
