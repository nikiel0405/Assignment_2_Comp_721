import time
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

'''
    Begining of PreProcessing
'''

col_names = ['ham','pineapple','mushroom', 'pepperoni','chicken','extra_cheese','BBQ_sauce']
label_name = ['good_pizza']

trainingData = pd.read_csv("Training_Data.csv", usecols=col_names,sep=',')
trainingLable = pd.read_csv('Training_Data.csv', usecols=label_name, sep=',')

testingData = pd.read_csv('Testing_Data.csv', usecols=col_names, sep=',')
testingLable = pd.read_csv('Testing_Data.csv', usecols=label_name, sep=',')

trainingham = trainingData.ham.tolist()
trainingpine = trainingData.pineapple.tolist()
trainingmus = trainingData.mushroom.tolist()
trainingpep = trainingData.pepperoni.tolist()
trainingchick = trainingData.chicken.tolist()
trainingcheese = trainingData.extra_cheese.tolist()
trainingBBQ = trainingData.BBQ_sauce.tolist()
traininglabel_list = trainingLable.good_pizza.tolist()


testingham = testingData.ham.tolist()
testingpine = testingData.pineapple.tolist()
testingmus = testingData.mushroom.tolist()
testingpep = testingData.pepperoni.tolist()
testingchick = testingData.chicken.tolist()
testingcheese = testingData.extra_cheese.tolist()
testingBBQ = testingData.BBQ_sauce.tolist()
testinglabel_list = testingLable.good_pizza.tolist()

FeatureVectorTrain=[]
FeatureVectorTest =[]

for i in range(len(traininglabel_list)):
    fv = []
    fv.append(trainingham[i])
    fv.append(trainingpine[i])
    fv.append(trainingmus[i])
    fv.append(trainingpep[i])
    fv.append(trainingchick[i])
    fv.append(trainingcheese[i])
    fv.append(trainingBBQ[i])
    FeatureVectorTrain.append(fv)
    if i < 20 :
        fvtest = []
        fvtest.append(testingham[i])
        fvtest.append(testingpine[i])
        fvtest.append(testingmus[i])
        fvtest.append(testingpep[i])
        fvtest.append(testingchick[i])
        fvtest.append(testingcheese[i])
        fvtest.append(testingBBQ[i])
        FeatureVectorTest.append(fvtest)


'''print("-----------Classifier 1 - Adaboost-----------")

classifier1 = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=3, learning_rate=0.6)

start_Time = time.time()
classifier1.fit(FeatureVectorTrain, traininglabel_list)
print("Time taken to train Adaboost Classifier ->" , time.time() - start_Time)

predictions = classifier1.predict(FeatureVectorTest)
print(predictions)

cm = confusion_matrix(testinglabel_list, predictions)
tn, fp, fn, tp = cm.ravel()
print("For Adaboost Classifier:")
print("True Negatives (Complaints) -> ",tn)
print("False Positives -> ", fp)
print("False Negatives -> ", fn)
print("True Positives (Compliments)-> ", tp)


ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix For Adaboost Classifier')
ax.xaxis.set_ticklabels(['0 - Not Good Pizza', '1 - Good Pizza']); ax.yaxis.set_ticklabels(['0 - Not Good Pizza', '1 - Good Pizza'])
plt.show()
print("Adaboost Classifier Predictions ->", predictions)
print(classification_report(testinglabel_list, predictions, target_names=["Not Good Pizza","Good Pizza"]))'''








'''print("-----------Classifier 2 - Support Vector Machine-----------")


classifier2 = SVC(C=3)
start_Time = time.time()
classifier2.fit(FeatureVectorTrain, traininglabel_list)
print("Time taken to train Support Vector Machine Classifier->", time.time() - start_Time)

predictions = classifier2.predict(FeatureVectorTest)
cm = confusion_matrix(testinglabel_list, predictions)
tn, fp, fn, tp = cm.ravel()
print("For Support Vector Machine Classifier:")
print("True Negatives (Complaints) -> ",tn)
print("False Positives -> ", fp)
print("False Negatives -> ", fn)
print("True Positives (Compliments) -> ", tp)


ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix For Support Vector Machine Classifier')
ax.xaxis.set_ticklabels(['0 - Not Good Pizza', '1 - Good Pizza']); ax.yaxis.set_ticklabels(['0 - Not Good Pizza', '1 - Good Pizza'])
#plt.show()

print(classification_report(testinglabel_list, predictions, target_names=["Not Good Pizza","Good Pizza"]))
print("Support Vector Classifier Predictions -> ", predictions)'''




classifier3 = DecisionTreeClassifier(max_depth=3)
start_time = time.time()
classifier3.fit(FeatureVectorTrain, traininglabel_list)
print("Time taken to train Decision Tree Classifier ->", time.time() - start_time)
predictions = classifier3.predict(FeatureVectorTest)
cm = confusion_matrix(testinglabel_list, predictions)
print(classification_report(testinglabel_list, predictions,target_names=["Not Good Pizza","Good Pizza"]))

