# 'Go To College Dataset'
# https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset?datasetId=2195547
# when the decision tree is made you can download it from the output folder (./DT.png)

# import all modules needed
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtWidgets import QDialog, QStackedWidget
from PyQt5.uic import loadUi
from sklearn import tree
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

class welcomeScreen(QDialog):
    def __init__(self):
        super(welcomeScreen,self).__init__()
        loadUi("UI go to college.ui",self)
        self.single.clicked.connect(self.gotoSinglePerson)
    def gotoSinglePerson(self):
        singleP=singlePersonScreen()
        widget.addWidget(singleP)
        widget.setCurrentIndex(widget.currentIndex()+1)
class singlePersonScreen(QDialog):
    def __init__(self):
        super(singlePersonScreen,self).__init__()
        loadUi("singlePerson.ui",self)
        self.checkD.clicked.connect(self.FDecision)
        self.Back.clicked.connect(self.BackToMain)

    def BackToMain(self):
        wlc=welcomeScreen()
        widget.setCurrentIndex(widget.currentIndex() - 1)
    def FDecision(self):
        global h1
        msg=''
        try :
            house_area= float(self.houseArea.toPlainText())
        except Exception as e:
            print(e)
            msg +='The house Area should be a float .'
        try :
            parent_age=float(self.parentAge.toPlainText())
        except ValueError:
            msg += 'The parent age should be a float .'
        try :
            average_grade=float(self.averageGrade.toPlainText())
            if not(average_grade in range(0,101))  :
                raise ValueError
        except ValueError :
            msg+= ' the average grade should be a float between 0 and 100 .'
        try:
            parent_salary= int(self.parentSalary.toPlainText())
            parent_salary=parent_salary*1000
        except ValueError:
            msg+='the salary should be an integer .'
        if msg != '' :
            self.message_3.setText(msg)
            return
        else:

            self.message_3.setText('data type entered is valid')

        #transforming the single data input ( for one person)
        try:
            school_type=self.schoolType.currentText()
            school_type= school_type.replace('Academic','0')
            school_type= int(school_type.replace('Vocational','1'))
            school_accreditation =self.schoolAccreditation.currentText()
            school_accreditation = school_accreditation.replace('A','0')
            school_accreditation =int( school_accreditation.replace('B', '1'))
            gender=self.gender.currentText()
            gender = gender.replace("Male", '1')
            gender = int(gender.replace("Female", '0'))
            residence= self.Residence.currentText()
            residence= residence.replace('Urban', '1')
            residence= int(residence.replace('Rural', '0'))
            interest=self.Interest.currentText()
            interest= interest.replace("Not Interested", '-2')
            interest= interest.replace("Less Interested", '-1')
            interest= interest.replace("Uncertain", '0')
            interest= interest.replace("Quiet Interested", '1')
            interest= int(interest.replace("Very Interested", '2'))
            parent_was_in_college= self.parentCollege.currentText()
            parent_was_in_college= parent_was_in_college.replace("True", '1')
            parent_was_in_college= int(parent_was_in_college.replace("False", '0'))
        except Exception as e:
            print(e)

        #creating a numpy array for the model implementation
        try:
            arr=np.array([school_type,school_accreditation, gender, interest, residence,parent_age , parent_salary,
                      house_area,average_grade, parent_was_in_college])
        except Exception as e:
            print(e, 'transformation error')
        #calculate the predictions
        try:
            predictions = model.predict(arr.reshape(1, -1))
            # print(predictions)
            a=''
            if not(predictions) :
                a='NOT '
            msg=  str(h1)+'% chance that the student is '+a+'going to college'
            self.message_2.setText(msg)

        except Exception as e:
            print(e, ' prediction error')


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
X = np.array(coll)[:, :-1]

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
h1=round((sum(h[h==True])/len(h))*100,2)
print('the efficiency is ', h1, '%')

#check the importance of each metric
print("the importance of each column in the model: ")
print(list((zip(coll.columns.tolist()[:-1],model.feature_importances_))))

#visualizing the tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10,8), dpi=150)
plot_tree(model, feature_names=coll.columns.tolist()[:-1], filled=True);
# print("Save the decision tree by right-clicking the tree below.")
# plt.savefig('./collegeDT.png')
# plt.show()



#main
app = QtWidgets.QApplication(sys.argv)
wlc=welcomeScreen()
widget=QStackedWidget()
widget.addWidget(wlc)
widget.setFixedWidth(909)
widget.setFixedHeight(686)
widget.show()
try:
    sys.exit(app.exec_())
except:
    print("exiting..")