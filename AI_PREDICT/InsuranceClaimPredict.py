import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#importing and displaying head of dataset
instabl = pd.read_csv('C:\\Users\\aronk\\Desktop\\AI_PREDICT\\insurance2.csv')
instabl.head()

#checking for null values
instabl.isnull().sum()

instabl.columns

X = instabl.iloc[:,:-1]
y = instabl['insuranceclaim']

X.head()

y.head()

instabl.head()

#plot visulization
plot1 = plt.figure(figsize=(20,10))
plot1.add_subplot(2,2,1)
sns.scatterplot(instabl['age'], instabl['bmi'])
plot1.add_subplot(2,2,2)
sns.scatterplot(instabl['sex'],instabl['bmi'])
plot1.add_subplot(2,2,3)
sns.scatterplot(instabl['smoker'],instabl['bmi'])
plot1.add_subplot(2,2,4)
sns.scatterplot(instabl['children'],instabl['bmi'])



#Training
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())



#Feature selection using Extra Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
mlmodel = ExtraTreesClassifier()
mlmodel.fit(X,y)

print(mlmodel.feature_importances_)
#ouputting bar graph for important features
ranked_factors = pd.Series(mlmodel.feature_importances_, index=X.columns)
ranked_factors.nlargest(len(X.columns)).plot(kind='bar')
plt.show()

#Model Building
#Random Forest
from sklearn.ensemble import RandomForestClassifier
rndm_classifier = RandomForestClassifier(n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features='sqrt', max_depth=10, criterion='entropy')
rndm_classifier.fit(X_train,y_train)

rndm_classifier_predict = rndm_classifier.predict(X_test)

rndm_classifier_predict

# #Performance Checking
from sklearn.metrics import confusion_matrix
rndm_classifier_cm = confusion_matrix(y_test,rndm_classifier_predict)
rndm_classifier_cm


# #Classification Report
from sklearn.metrics import classification_report
rndm_classifier_report = classification_report(y_test,rndm_classifier_predict)
print(rndm_classifier_report)

#Randomized Search CV
from sklearn.model_selection import RandomizedSearchCV


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
criterion = ['entropy','gini']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]


# Create the random grid
rndGrid = {'n_estimators': n_estimators,
               'criterion':criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(rndGrid)


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
random_rf = RandomizedSearchCV(estimator = rndm_classifier, param_distributions = rndGrid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
random_rf.fit(X_train,y_train)
random_rf.best_params_
random_rf.best_score_


predictions=random_rf.predict(X_test)
predictions

print(classification_report(y_test, predictions))