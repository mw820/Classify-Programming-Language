import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

#Importing Data
df = pd.read_csv('data.csv')

#Dropping not usefull columns
df.drop(labels=['proj_id', 'file_id'], axis="columns", inplace=True)

# dropping duplicated and null values
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


# After exploration of text body I decided to remove single characters like "a", "b" etc. and strings started with numerical 
# valuee for example "43bp", as they are usually names of sort variables or other non informative strings which are not
# specific for any particular language.

def cleaning(x):
    clean_row = pd.Series(x).replace(r'\b([A-Za-z])\1+\b', '', regex=True)
    clean_row = pd.Series(x).replace(r'\b[A-Za-z]\b', '', regex=True)
    return clean_row

transformer = FunctionTransformer(cleaning)
transformer.transform(df['file_body'])


# dividing dataset into train and test split
X = df['file_body']
y = df['language']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_test,y_test,test_size = 0.5)


# In order to turn strings to numerical values I used tfidf vectorizer. There are other popular method for obtaining word 
# embeddings like BERT ord Word2Vec (a bit oldschool), but as we are not dealing with classic text, we could not use 
# pre-trained embeddings of BERT/Word2Vec thus TFIDF is more appropriate

def vectorize(data,tfidf_vect_fit):
    X_tfidf = tfidf_vect_fit.transform(data)
    words = tfidf_vect_fit.get_feature_names()
    X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
    X_tfidf_df.columns = words
    return(X_tfidf_df)


# extracting module names and variebles | operator characters | brackets, tabes ets
token_pattern = r"""(\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""


# creating word embeddings
tfidf_vect = TfidfVectorizer(token_pattern=token_pattern, max_features=3000)
tfidf_vect_fit=tfidf_vect.fit(X_train)
X_train=vectorize(X_train,tfidf_vect_fit)
X_val=vectorize(X_val,tfidf_vect_fit)
X_test=vectorize(X_test,tfidf_vect_fit)

# checking RF performance
rf = RandomForestClassifier()
scores = cross_val_score(rf,X_train,y_train.values,cv=10)

#printing results
def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, results.cv_results_['params']):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

#choosing best params
parameters = {
    'n_estimators': [50,100,300],
    'max_depth': [2,10,20,None]
}
cv = GridSearchCV(rf,parameters)
cv.fit(X_val,y_val.values)
print_results(cv)


#fitting rf with best params
rf = RandomForestClassifier(n_estimators=300,max_depth=None)
rf.fit(X_train, y_train.values)

#results
y_pred = rf.predict(X_test)
accuracy = round(accuracy_score(y_test,y_pred), 3)
print('MAX DEPTH: {} / NUMBER OF EST: {} / Accuracy: {} '.format(rf.max_depth,
                                                                     rf.n_estimators,
                                                                     accuracy,
                                                                     ))



# This was the basic idea I wanted to explore. Of course we could experiment more with 1) different parameters for grid search
# like gini/entropy, max depth or min sample slit. We could also test performance of different classification models 
# like RRN, MLPClassifier, basic Logistic Regression, just decision tree or many more.



# Let's just plot a heatmap to see if any language is often classify as other specific language.
cf_matrix = confusion_matrix(y_test,y_pred)
df_cm = pd.DataFrame(cf_matrix, index = [i for i in sorted(list(set(y_pred)))],
                  columns = [i for i in sorted(list(set(y_pred)))])
plt.figure(figsize = (12,12))
sn.heatmap(df_cm, annot=True)
