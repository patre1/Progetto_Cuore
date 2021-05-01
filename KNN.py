
import numpy as np, pandas as pd, seaborn as sn
from matplotlib import pyplot
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from inspect import signature
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

feature = ["age", "sex", "cp", "trtbps", "chol", "restecg", "thalachh", "slp", "caa", "output"]
feature_dummied = ["age", "sex", "cp", "trtbps", "chol", "restecg", "thalachh", "slp", "caa"]
dataset = pd.read_csv("heart.csv", sep=";", names=feature,
                      dtype={'age': object, 'sex': object, 'cp': object, 'trtbps': object, 'chol': object,
                             'restecg': object, 'thalachh': object,'slp': object, 'caa': object,'output': object})
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
data_dummies = data_dummies.drop(["output"], axis=1)

X = data_dummies
y = pd.get_dummies(dataset["output"], columns=["output"])
y = y["1"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=13)

error = []

for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# Grafico che mostra l'errore medio nelle predizioni a seguito di una variazione del valore K(numero vicini)
plt.plot(range(1, 20), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
# prima
plt.show()

neigh = KNeighborsClassifier(n_neighbors=7)
knn = neigh.fit(X_train, y_train)


prediction = knn.predict(X_test)
accuracy = accuracy_score(prediction, y_test)

print('\nClasification report:\n', classification_report(y_test, prediction))



probs = knn.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
# seconda
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
# show the plot
pyplot.show()
confusion_matrix = confusion_matrix(y_test, prediction)

df_cm = pd.DataFrame(confusion_matrix, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()
#pyplot.show()

average_precision = average_precision_score(y_test, prediction)
precision, recall, _ = precision_recall_curve(y_test, prediction)
f1 = f1_score(y_test, prediction)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

""" DA USARE SE SI VUOLE UTILIZZARE L'ALGORITMO SMOTE PER BILANCIARE IL DATASET """
'''
sm = SMOTE(random_state=0)
X1, y1 = sm.fit_resample(X, y.ravel())
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=13)
knn1 = neigh.fit(X1_train, y1_train)
prediction_res = knn1.predict(X1_test)
accuracy_res = accuracy_score(prediction_res, y1_test)

print('\n')

print('\nClasification report:\n', classification_report(y1_test, prediction_res))

# cancella la prima matrice di confusione per stamparne la seconda
confusion_matrix2 = confusion_matrix(y1_test, prediction_res)
df_cm = pd.DataFrame(confusion_matrix2, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
plt.show()

# train model with cv of 5


# train model with cv of 5

probs = knn.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]

auc = roc_auc_score(y_test, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
# plot no skill
pyplot.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
pyplot.plot(fpr, tpr, marker='.')
pyplot.xlabel('FP RATE')
pyplot.ylabel('TP RATE')
pyplot.show()


average_precision2 = average_precision_score(y1_test, prediction_res)
precision2, recall2, _ = precision_recall_curve(y1_test, prediction_res)

# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall2, precision2, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall2, precision2, alpha=0.2, color='b', **step_kwargs)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision2))
f1_smote = f1_score(y1_test, prediction_res)
plt.show()
'''
