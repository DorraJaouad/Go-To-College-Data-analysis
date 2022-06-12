# 'Go To College Dataset'
# https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset?datasetId=2195547
# when the decision tree is made you can download it from the output folder (./DT.png)

# import all modules needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree

# open/read the .csv dataset with panda
coll = pd.read_csv("C://Users//Dorra//Desktop//Job related//Portfolio//data.csv")


# make categorical variables numeric
coll['type_school']= coll['type_school'].replace('Academic', 0)
coll["type_school"]= coll["type_school"].replace("Vocational", 1)
coll["school_accreditation"]= coll["school_accreditation"].replace("A", 0)
coll["school_accreditation"]= coll["school_accreditation"].replace("B", 1)
coll["gender"]= coll["gender"].replace("Male", 1)
coll["gender"]= coll["gender"].replace("Female", 0)
coll["interest"]= coll["interest"].replace("Not Interested", -2)
coll["interest"]= coll["interest"].replace("Less Interested", -1)
coll["interest"]= coll["interest"].replace("Uncertain", 0)
coll["interest"]= coll["interest"].replace("Quiet Interested", 1)
coll["interest"]= coll["interest"].replace("Very Interested", 2)
coll["residence"]= coll["residence"].replace("Urban", 1)
coll["residence"]= coll["residence"].replace("Rural", 0)
coll["parent_was_in_college"]= coll["parent_was_in_college"].replace("TRUE", 1)
coll["parent_was_in_college"]= coll["parent_was_in_college"].replace("FALSE", 0)

# make it a numpy array so numpy can read it
# first 8 columns of matrix are considered parameters
# the last one is considered to be the class
X = np.array(coll)[:, :8]

#second method
# X = pd.get_dummies(df.drop("animal", axis=1), drop_first=True)

#take the column to be predicted
y=coll["in_college"]

#split the data to 2 parts respectively for the training and for the testing )
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

#build the decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

#create the predictions for X_test
predictions = model.predict(X_test)
# print(predictions)

#calculate the efficiency of the model
h=np.array(predictions==y_test)
print('the efficiency is ', round((sum(h[h==True])/len(h))*100,2), '%')

#check the importance of each metric
print(list((zip(coll.columns.tolist()[:-1],model.feature_importances_))))

#visualizing the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10,8), dpi=150)
plot_tree(model, feature_names=coll.columns.tolist()[:-1], filled=True);
# print("Save the decision tree by right-clicking the tree below.")
# plt.savefig('./collegeDT.png')
# plt.show()