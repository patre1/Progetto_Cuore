from inspect import signature
import numpy as np
import pandas as pd
import seaborn as sn
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------------------------------------

# create model
from tensorflow.python.keras.applications.densenet import layers
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

#creazione struttura rete neurale
def create_model():
    network = Sequential(np.shape(9, ))
    network.add(layers.Dense(25, activation='sigmoid'))
    network.add(layers.Dense(1, activation='sigmoid'))
    # Compile model
    network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return network


# ----------------------------------------------------------------------------------------------------------

np.random.seed(7)
dataset = pd.read_csv("heart.csv", sep=";",
                      names=["age","sex","cp","trtbps","chol","restecg","thalachh","slp","caa","output"],
                      dtype={ 'age': object, 'sex': object, 'cp': object,
                             'trtbps': object, 'chol': object,'restecg': object, 'thalachh':object,
                             'slp': object, 'caa': object, 'output':object})
network_data = pd.get_dummies(dataset, columns=["age","sex","cp","trtbps","chol","restecg","thalachh","slp","caa"])

network_data = network_data.drop(["output"], axis=1)

X = network_data
y = pd.get_dummies(dataset["output"], columns=["output"])
y = y["1"]



sm = SMOTE(random_state=0)

X, y = sm.fit_resample(X, y.ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=13)

model = create_model()
# Fit the model
model.fit(X_train, y_train, epochs=100, batch_size=10)

scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# calcolo della predizione
predictions = model.predict(X_test)
rounded = [round(x[0]) for x in predictions]

accuracy = accuracy_score(y_test, rounded)

print('\nClasification report:\n', classification_report(y_test, rounded))
print('\nConfussion matrix:\n', confusion_matrix(y_test, rounded))


model1 = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
print(model1)


model1 = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)


probs = model.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 0]

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
# show the plot
confusion_matrix = confusion_matrix(y_test, rounded)
df_cm = pd.DataFrame(confusion_matrix, index=[i for i in "01"], columns=[i for i in "01"])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.xlabel('Valori Predetti')
plt.ylabel('Valori Effettivi')
plt.show()

average_precision = average_precision_score(y_test, rounded)
precision, recall, _ = precision_recall_curve(y_test, rounded)


step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
plt.show()

f1 = f1_score(y_test, rounded)

