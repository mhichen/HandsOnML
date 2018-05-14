#!/usr/bin/python3

from sklearn.datasets import fetch_mldata
import scipy
import scipy.io as sio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import sklearn.metrics as skm
from sklearn.ensemble import RandomForestClassifier

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
    plt.xlabel("Threshold")
    plt.legend(loc = "upper left")
    plt.ylim([0, 1])


def plot_roc_curve(fpr, tpr, label = None):
    plt.plot(fpr, tpr, lw = 2, label = label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    

#mnist = fetch_mldata('MINST original')
mnist = sio.loadmat('/home/ivy/scikit_learn_data/mldata/mnist-original', squeeze_me = True)

X, y = mnist["data"].T, mnist["label"].T

# (70000, 784) - 70000 images with 784 features each
# np.sqrt(784) = 28 -- images are 28 x 28 pixels
print(X.shape)

# (70000, )
print(y.shape)

# Check out an example in the dataset
i_ind = 36000

some_digit = X[i_ind]
some_digit_image = some_digit.reshape(28, 28)

print("Showing image of digit", y[i_ind])
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()

# Separate out train vs test
b_ind = 60000
X_train, X_test, y_train, y_test = X[:b_ind], X[b_ind:], y[:b_ind], y[b_ind:]

# Shuffle dataset
shuffle_ind = np.random.permutation(b_ind)
X_train, y_train = X_train[shuffle_ind], y_train[shuffle_ind]

##***********************************************************
## Training a simple binary classifier
##***********************************************************
# Start by training classifier that identifies digit 5 or not

# Create target vectors of True/False
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# Pick a classifier and train it
# Stochastic Gradient Descent (SGD)
# Give a seed for reproducible result
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)

# predict images of number 5
print(sgd_clf.predict(X_train[1:20]))
print(y_train[1:20])

## Performance
## Measuring Accuracy using CV
# 3-fold and score using accuracy
result = cross_val_score(sgd_clf, X_train, y_train_5, cv = 3, scoring = "accuracy")
print(result)

## Measuring performance using confusion matrix
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)

# each row is an actual class and each column is a predicted class
print("The confusion matrix is")
print(skm.confusion_matrix(y_train_5, y_train_pred))

print("The precision score is") # 0.50988
print(skm.precision_score(y_train_5, y_train_pred))

print("The recall score is") # 0.84246
print(skm.recall_score(y_train_5, y_train_pred))

print("The F1-score is") # 0.63528
print(skm.f1_score(y_train_5, y_train_pred))

# cannot set threshold for decisions, but can see the score for each instance
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

# can return scores instead of predictions
y_scores = sgd_clf.decision_function(X_train)
#cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)
print("Printing y_scores")
print(y_scores)

precisions, recalls, thresholds = skm.precision_recall_curve(y_train_5, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

y_train_pred_90 = (y_scores > 70000)

print(skm.precision_score(y_train_5, y_train_pred_90))
print(skm.recall_score(y_train_5, y_train_pred_90))


##***********************************************************
## ROC Curve
##***********************************************************
fpr, tpr, thresholds = skm.roc_curve(y_train_5, y_scores)

plot_roc_curve(fpr, tpr)
plt.show()

print('AUC is')
# 0.96879
print(skm.roc_auc_score(y_train_5, y_scores))


##***********************************************************
## Training a Random Forest Classifier
##***********************************************************
print()
print("Training Random Forest Classifier")

forest_clf = RandomForestClassifier(random_state = 42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = "predict_proba")
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = skm.roc_curve(y_train_5, y_scores_forest)


plt.plot(fpr, tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc = "lower right")
plt.show()

# AUC is 0.99813
print("AUC is")
print(skm.roc_auc_score(y_train_5, y_scores_forest))


y_scores_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = "predict")


print("The precision score is") # 0.98507
print(skm.precision_score(y_train_5, y_scores_forest))

print("The recall score is") # 0.81535
print(skm.recall_score(y_train_5, y_scores_forest))

print("The F1-score is") # 0.89221
print(skm.f1_score(y_train_5, y_scores_forest))


##***********************************************************
## Multiclass Classification
##***********************************************************

