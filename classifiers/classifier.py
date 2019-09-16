#Created on Tue Aug 27 22:13:17 2019

import re
import glob
import os
import pandas as pd
#import csv


path = r'C:\\Users\\jeffr\\Desktop\\projects\\nkb classifier\\neverknowsbest.chat'
all_files = glob.glob(os.path.join(path, "*.html"))

read = []

for i in range(len(all_files)):
    f = open(all_files[i], encoding="utf8")
    read.append(f.read())

data = ''.join(read)

#cleanup
htmlcleanup = re.compile(r'<[^>]+>')

def remove_html(text):
    return htmlcleanup.sub('', text)

data = remove_html(data).split('\n')

for i in range(len(data)):
    data[i] = re.split('\) |\: ', data[i])
    if data[i][0].find(r'(') == 0:
        del data[i][0]

data = [x for x in data if len(x) > 1 and x[0].find(' ') == -1]

#users
users = list(set([x[0] for x in data]))

#with open("C:\\Users\\jeffr\\Desktop\\projects\\nkb classifier\\users.csv",'w') as resultFile:
#    wr = csv.writer(resultFile, dialect='excel')
#    wr.writerow(users)

usersmapped = []

f_in = open('C:\\Users\\jeffr\\Desktop\\projects\\nkb classifier\\users mapped.csv','r')
usersmapped = f_in.readlines()
f_in.close()

for i in range(len(usersmapped)):
    usersmapped[i] = usersmapped[i].strip().split(',')

userdict = {usersmapped[0]:usersmapped[1] for usersmapped in usersmapped}

for i in range(len(data)):
    data[i][0] = userdict[data[i][0]]
    
df = pd.DataFrame(data)

df = df.drop(list(range(2,15)), axis = 1)
df.columns = ['user', 'message']

#taking a look
df['user'].value_counts()

#tf-idf, split by user?
#from sklearn.feature_extraction.text import TfidfVectorizer
# 
#tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1,2), lowercase=True, stop_words='english')
# 
#df2 = df[0:100000]
# 
#features = tfidf.fit_transform(df2.message).toarray()
#labels = df2.user
#features.shape

#training naive bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['user'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)


clf = MultinomialNB().fit(X_train_tfidf, y_train)

#training mlp
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(5), batch_size = 'auto', max_iter=500, verbose=True)

mlp.fit(X_train_tfidf, y_train)

#evaluation
from sklearn.metrics import classification_report, confusion_matrix
predictions = clf.predict(count_vect.transform(X_test))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


predictions = mlp.predict(count_vect.transform(X_test))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

#predictions
print(clf.predict(count_vect.transform([""])))

#save models
import pickle
filename = 'C:\\Users\\jeffr\\Desktop\\projects\\nkb classifier\\nn_nkb.sav'
pickle.dump(mlp, open(filename, 'wb'))

# load models
#loaded_model = pickle.load(open(filename, 'rb'))
#result = loaded_model.score(X_test, Y_test)
#print(result)