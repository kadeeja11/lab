#maam's clustering
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,davies_bouldin_score,calinski_harabasz_score

dataframe=pd.read_csv("drug200.csv")
print(dataframe)
dataframe.info()
dataframe.isnull().sum()
dataframe.describe()

dataframe_numeric = dataframe.select_dtypes(include=['number']).copy()
imputer=SimpleImputer(strategy="mean")
dataframe_numeric = pd.DataFrame(imputer.fit_transform(dataframe_numeric), columns=dataframe_numeric.columns, index=dataframe.index)
dataframe[dataframe_numeric.columns] = dataframe_numeric
dataframe = pd.get_dummies(dataframe, columns=['Sex', 'BP', 'Cholesterol', 'Drug'])
print(dataframe)
plt.plot(dataframe_numeric)
plt.title("Data Before Normalization")
plt.show()

scaler=StandardScaler()
dataframe_numeric = pd.DataFrame(scaler.fit_transform(dataframe_numeric), columns=dataframe_numeric.columns, index=dataframe.index)
dataframe[dataframe_numeric.columns] = dataframe_numeric
print(dataframe_numeric) 
plt.plot(dataframe_numeric)
plt.title("Data After Normalization")
plt.show()


kmeans=KMeans(n_clusters=2)
kmeans.fit(dataframe) 
label=kmeans.labels_
plt.scatter(dataframe.iloc[:,0],dataframe.iloc[:,1],c=label,s=20,cmap="plasma")
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,color="red")
plt.title("K-Means Clustering")
plt.show()
sse=[]
ss=[]

for i in range(2,11):
    kmeans=KMeans(n_clusters=i,random_state=42)
    kmeans.fit(dataframe)
    label=kmeans.labels_
    sse.append(kmeans.inertia_)
    print("Number of Clusters:",i)
    plt.scatter(dataframe.iloc[:,0],dataframe.iloc[:,1],c=label,s=20,cmap="plasma") 
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,color="red")
    plt.show()
plt.plot(range(2,11),sse,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Squared Error")
plt.title("Elbow Method")
plt.show()


hierarchical = AgglomerativeClustering(n_clusters=2, linkage='ward')
hierarchical.fit(dataframe)
linkage=linkage(dataframe,method="ward")
dendrogram(linkage,truncate_mode="level",p=3)
plt.title("Hierarchical Custering")
plt.show()

dbscan=DBSCAN(eps=0.3,min_samples=10)
dbscan.fit(dataframe)
label=dbscan.labels_
sns.scatterplot(x=dataframe.iloc[:,0],y=dataframe.iloc[:,1],hue=label,palette="autumn")
plt.title("DBSCAN Clustering")
plt.show()



silhouette_score={
    "K-Means Clustering":silhouette_score(dataframe,kmeans.labels_),
    "Hierarchial Clustering":silhouette_score(dataframe,hierarchical.labels_), # Assuming 'hierarchical' is defined and fitted elsewhere

}
davies_bouldin_score={
    "K-Means Clustering":davies_bouldin_score(dataframe,kmeans.labels_),
    "Hierarchial Clustering":davies_bouldin_score(dataframe,hierarchical.labels_) # Assuming 'hierarchical' is defined and fitted elsewhere
}
calinski_harabasz_score={
    "K-Means Clustering":calinski_harabasz_score(dataframe,kmeans.labels_),
    "Hierarchial Clustering":calinski_harabasz_score(dataframe,hierarchical.labels_) # Assuming 'hierarchical' is defined and fitted elsewhere
}

plt.bar(silhouette_score.keys(),silhouette_score.values(),color="red")
plt.xlabel("Clustering Algorithm")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.show()
plt.bar(davies_bouldin_score.keys(),davies_bouldin_score.values())
plt.xlabel("Clustering Algorithm")
plt.ylabel("Davies Bouldin Score")
plt.title("Davies Bouldin Score")
plt.show()
plt.bar(calinski_harabasz_score.keys(),calinski_harabasz_score.values(),color="green")
plt.xlabel("Clustering Algorithm")
plt.ylabel("Calinski Harabasz Score")
plt.title("Calinski Harabasz Score")
plt.show()
plt.scatter(calinski_harabasz_score.values(),davies_bouldin_score.values(),s=30,color="violet")
plt.xlabel("Calinski Harabasz Score")
plt.ylabel("Davies Bouldin Score")
plt.title("Evaluation Metrics")
plt.show()
plt.scatter(calinski_harabasz_score.values(),silhouette_score.values(),color="magenta")
plt.xlabel("Calinski Harabasz Score")
plt.ylabel("Silhouette Score")
plt.title("Evaluation Metrics")
plt.show()
plt.scatter(davies_bouldin_score.values(),silhouette_score.values(),color="purple")
plt.xlabel("Davies Bouldin Score")
plt.ylabel("Silhouette Score")
plt.title("Evaluation Metrics")
plt.show()
df=pd.DataFrame({
    "Silhouette Score":silhouette_score.values(),
    "Davies Bouldin Score":davies_bouldin_score.values(),
    "Calinski Harabasz Score":calinski_harabasz_score.values()
},index=silhouette_score.keys())

sns.heatmap(df.corr())







#classification
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = 'drug200.csv'
df = pd.read_csv(path)
df.drop_duplicates(inplace=True)
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = le.fit_transform(df[col])

df[df.columns] = df[df.columns].fillna(df[df.columns].median())

threshold = 0.8
corr_matrix = df.corr().abs()
sns.heatmap(corr_matrix)
upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1)
highly_correlated_mask = (corr_matrix > threshold) & upper_triangle
dropped_list = df.columns[highly_correlated_mask.any()].tolist()
df = df.drop(df.columns[highly_correlated_mask.any()], axis=1)
print("Dropped Features:", dropped_list)

X = df.drop(columns='Drug')
y = df['Drug']

scaler = StandardScaler()
X_rescaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_rescaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
ada = AdaBoostClassifier()

rf_param_grid = {'n_estimators': [2, 3, 5], 'max_depth': [None, 3, 5]}
gb_param_grid = {'n_estimators': [2, 3, 5], 'learning_rate': [0.05, 0.1, 0.5]}
ada_param_grid = {'n_estimators': [2, 3, 5], 'learning_rate': [0.05, 0.1, 0.5]}

rf_grid = GridSearchCV(rf, param_grid=rf_param_grid, scoring='accuracy', cv=3)
rf_grid.fit(X_train, y_train)
gb_grid = GridSearchCV(gb, param_grid=gb_param_grid, scoring='accuracy', cv=3)
gb_grid.fit(X_train, y_train)
ada_grid = GridSearchCV(ada, param_grid=ada_param_grid, scoring='accuracy', cv=3)
ada_grid.fit(X_train, y_train)

best_classifiers = [(rf_grid.best_estimator_, rf_grid.best_score_),
                    (gb_grid.best_estimator_, gb_grid.best_score_),
                    (ada_grid.best_estimator_, ada_grid.best_score_)]

best_classifiers.sort(key=lambda x: x[1], reverse=True)
top_classifiers = [classifier[0] for classifier in best_classifiers[:3]]

ensemble = VotingClassifier(estimators=[('rf', top_classifiers[0]),
                                        ('gb', top_classifiers[1]),
                                        ('ada', top_classifiers[2])], voting='hard')

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("Ensemble Model Performance:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)





#decision tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("drug200.csv")

label_encoder = LabelEncoder()
categorical_features = ['Sex', 'BP', 'Cholesterol', 'Drug']
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])


x = df.drop(columns=["Drug"])
y = df["Drug"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#check for missing values
print("missing values in train data:\n",x_train.isnull().sum())
print("missing values in test data:\n",x_test.isnull().sum())

#decision tree
clf = DecisionTreeClassifier(max_depth=3,random_state=42)
clf.fit(x_train,y_train)

y_test_pred=clf.predict(x_test)

train_accuracy=accuracy_score(y_test,y_test_pred)
print("Test accuracy:",train_accuracy)

train_cm=confusion_matrix(y_test,y_test_pred)
print("Test confusion matrix:",train_cm)

plt.figure(figsize=(20,10))

plot_tree(clf, feature_names=x_train.columns,
          class_names=['drugX','drugY','drugA','drugB','drugC'],
          filled=True,
          rounded=True,
          fontsize=12
          )
plt.show()
