import urllib.request
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
response = urllib.request.urlopen(url)
data = response.read()
text = data.decode('utf-8') # use the utf-8 string format to create a string 'str' object
iris_df=pd.read_csv(url, names=("sepal length","sepal width","petal length","petal width","class")) # Panda object

pd.set_option('display.max_columns', None)
plot = plt.subplot(111, projection='3d')

x = iris_df["sepal length"]
y = iris_df["sepal width"]
z = iris_df["petal length"]
iris_df["class"]
colors = {'Iris-setosa':'red', 'Iris-virginica':'blue', 'Iris-versicolor':'green'}

plot.scatter(x, y, z, c  = iris_df["class"].apply(lambda x: colors[x]))

#labels
plot.set_xlabel('sepal length')
plot.set_ylabel('sepal width')
plot.set_zlabel('petal length')

#plt.show()

######### decision tree to classify the iris dataset

from sklearn import tree

attributes = iris_df[["sepal length","sepal width","petal length","petal width"]]
target = iris_df[["class"]]

clf = tree.DecisionTreeClassifier(criterion='entropy')

clf = clf.fit(attributes,target)

clf.predict([[1,1,1,1]])

prediction = clf.predict(attributes)
prediction_df = pd.DataFrame({"prediction": prediction})
prediction_df.head()


# create a result that contains the training data classes and the prediction result
# use the pandas function concat to join the data frames - note the axis parameter means to join columns
training_result = pd.concat([prediction_df, target], axis=1)
print (training_result.head())

# write the code to calculate the misclassifications here...
# calculate the proportion of records where the predicted class is not equal to the actual class

diff = 0
for index,row in training_result.iterrows():
    if row["prediction"] != row ["class"]:
        diff += 1

proportion = diff/training_result.shape[0]

print (proportion)

#################

attributes_training = attributes[attributes.index % 2 != 0]  # Use very 2rd row, exclude every second element starting from 0
target_training = target[target.index % 2 != 0] # every second row

# learn the decision tree
clf2 = tree.DecisionTreeClassifier(criterion='entropy')
clf2 = clf.fit(attributes_training,target_training)

attributes_test = attributes[attributes.index % 2 != 1]  # Use very 2rd row, exclude every second element starting from 0
prediction = clf.predict(attributes_test)
prediction_df_1 = pd.DataFrame({"prediction": prediction})
prediction_df_1.head()

actual_class_test = target[target.index % 2 != 1]
actual_class_test.index=range(75)
training_result = pd.concat([prediction_df_1, actual_class_test], axis=1)

diff = 0
for index,row in training_result.iterrows():
    if row["prediction"] != row ["class"]:
        diff += 1

proportion = diff/training_result.shape[0]

print ("new one:",proportion)

#print(training_result)