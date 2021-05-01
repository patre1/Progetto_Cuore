import pandas as pd, seaborn as sn
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature



feature = ["age", "sex", "cp", "trtbps", "chol", "restecg", "thalachh", "slp", "caa", "output"]
feature_dummied = ["age", "sex", "cp", "trtbps", "chol", "restecg", "thalachh", "slp", "caa"]
dataset = pd.read_csv("heart.csv", sep=";", names=feature,
                      dtype={'age':object, 'sex':object, 'cp':object, 'trtbps':object, 'chol':object,'restecg':object, 'thalachh':object,'slp':object,'caa':object,'output':object})
data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
X = data_dummies.drop(["output"], axis=1)
y = pd.get_dummies(data_dummies['output'], columns=['output'])
y = y.drop(["1"], axis=1)
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=2, n_init=9, random_state=0)
y_kmeans = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
print("\nLabels:")
print(kmeans.labels_)
average_precision = average_precision_score(y, y_kmeans)
precision, recall, _ = precision_recall_curve(y, y_kmeans)


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


accuracy = accuracy_score(y_kmeans, y)
print ('\nClasification report:\n',classification_report(y, y_kmeans))

matrix = confusion_matrix(y, y_kmeans)
df_cm = pd.DataFrame(matrix, index = [i for i in "01"], columns = [i for i in "01"])
plt.figure(figsize = (10,7))

sn.heatmap(df_cm, annot=True)
plt.show()

