import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


iris = pd.read_csv("iris.csv")


y = iris["Species"]

x = iris[[
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]]

print(y)
print(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=50, test_size=0.2
)


dtree1 = tree.DecisionTreeClassifier(max_depth=3, random_state=1)


dtree1.fit(x_train, y_train)


y_predict = dtree1.predict(x_test)


print("Accuracy of Decision Tree-Test:",
      accuracy_score(y_test, y_predict))


print("\nConfusion Matrix - Test:\n",
      confusion_matrix(y_test, y_predict))


print(classification_report(y_test, y_predict))
