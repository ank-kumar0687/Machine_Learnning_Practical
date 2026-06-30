from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import dtreeviz  # Corrected modern import

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a DecisionTreeClassifier using entropy (Information Gain)
model = DecisionTreeClassifier(criterion='entropy', max_depth=3)  
model.fit(X_train, y_train)

print("Model trained successfully!")

# Visualize the decision tree using the updated dtreeviz API
try:
    viz_model = dtreeviz.model(
        model,
        X_train=X,
        y_train=y,
        feature_names=iris.feature_names,
        class_names=list(iris.target_names)
    )
    v = viz_model.view()
    v.show()  # Opens the gorgeous visual diagram window
except Exception as e:
    print("\n[Note] Beautiful dtreeviz rendering failed because Graphviz isn't fully set up on your PC.")
    print("Falling back to standard text tree layout:\n")
    from sklearn import tree
    print(tree.export_text(model, feature_names=list(iris.feature_names)))
