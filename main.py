import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


st.title("Interactive WebApp")

st.write("""
# Explore Different Classifiers Here
which one is the best for your data? 
""") #H1 tag

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine dataset"))
classifier_name = st.sidebar.selectbox("Choose Classifier", ("KNN", "SVM", "Random Forest"))

#Load Dataset
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

#calling the get_dataset function
x, y = get_dataset(dataset_name)
st.write("Shape of the Dataset: ", x.shape)
st.write("No. of Classes: ", len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 15.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

#classifier
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1224)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {classifier_name}")
st.write(f"Accuracy = {acc}")

#Plot
pca = PCA(3)
x_projected = pca.fit_transform(x)

x1 = x_projected[:,0]
x2 = x_projected[:,1]

fig =plt.figure()
plt.scatter(x1,x2,c=y, alpha = 0.8, cmap= "viridis")
plt.xlabel("Principle component 1")
plt.ylabel("Principle component 2")
plt.colorbar()

st.pyplot(fig)


