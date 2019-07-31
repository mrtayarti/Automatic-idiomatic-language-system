import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC

data = pd.read_csv('Dataset.csv')
X = data.iloc[0:, 0:11].values
Y = data.iloc[0:, 11].values
# Y = Y.reshape(-1, 1)

ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(X)  # It returns an numpy array
v = DictVectorizer(sparse=False)
# turn word of training set into vector for training
# X = v.fit_transform(X.to_dict('records'))

# for i in X:
#     print(X_ohe)
#
# print(Y)

x_train, x_test, y_train, y_test = train_test_split(X_ohe, Y, test_size=0.50, random_state=0)

parameter_grid = {'kernel': ('linear', 'rbf'), 'C': (600, 800, 1000, 1200),
                  'gamma': (0.05, 0.08, 0.1, 0.15, 'scale'), 'decision_function_shape': ('ovo', 'ovr'),
                  'shrinking': (True, False)}

SVM = GridSearchCV(SVC(), parameter_grid, cv=5)

print("====================Grid Search================\n", SVM.fit(x_train, y_train))
print("\n====================Best Parameter====================\n", SVM.best_params_)
print("\n====================Kappa Statistic====================\n", cohen_kappa_score(y_test, SVM.predict(x_test)))
print("\n====================Confusion Matrix====================\n", pd.crosstab(y_test, SVM.predict(x_test),
                                                                                  rownames=['True'],
                                                                                  colnames=['Predicted'], margins=True))
print("\n====================Precision table====================\n", classification_report(y_test, SVM.predict(x_test)))
print("\n====================Accuracy====================\n ", accuracy_score(y_test, SVM.predict(x_test)))
