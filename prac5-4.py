import urllib.request
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn import tree

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'
response = urllib.request.urlopen(url)
data = response.read()
text = data.decode('utf-8')

breastCancer_df = pd.read_csv (url, sep="\s+", names=("Sequence Name","mcg","gvh","lip","chg","aac","alm1","alm2","Class Distribution"))

pd.set_option('display.max_columns', None)

print (breastCancer_df[:].head())

plot = plt.subplot(111, projection='3d')

x= breastCancer_df["mcg"]
y= breastCancer_df["gvh"]
z= breastCancer_df["lip"]
breastCancer_df["Class Distribution"]
colors={"cp":"red","pp": "blue","im":"green","imS":"green","imU":"green","imL":"green","om":"yellow","omL":"yellow"}

plot.scatter(x, y, z, c= breastCancer_df["Class Distribution"].apply(lambda x: colors[x]))

#labels
plot.set_xlabel('mcg')
plot.set_ylabel('gvh')
plot.set_zlabel('lip')

#plt.show()

######### decision tree to classify the iris dataset
attributes = breastCancer_df[["mcg","gvh","lip","lip","chg","aac","alm1","alm2"]]
target = breastCancer_df[["Class Distribution"]]

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(attributes,target)

clf.predict([[1,1,1,1,1,1,1,1]])
prediction = clf.predict(attributes)
prediction_df = pd.DataFrame({"prediction": prediction})

# create a result that contains the training data classes and the prediction result
# use the pandas function concat to join the data frames - note the axis parameter means to join columns
training_result = pd.concat([prediction_df, target], axis=1)

diff = 0
for index,row in training_result.iterrows():
    if row["prediction"] != row ["Class Distribution"]:
        diff += 1
proportion = diff/training_result.shape[0]
print (proportion)

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
actual_class_test.index=range(168)
training_result = pd.concat([prediction_df_1, actual_class_test], axis=1)

# calculate the misclassification percentage
diff = 0
for index,row in training_result.iterrows():
    if row["prediction"] != row ["Class Distribution"]:
        diff += 1

proportion = diff/training_result.shape[0]

print ("misclassification percentage = ",proportion)