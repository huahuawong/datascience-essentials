## Q1. How do to find thresholds for a classifier?
In general, the threshold used for a classifier is 0.5, but this could pose a problem when it's dealing with an imbalanced dataset. There are different ways to find the optimal threshold. A simple method is similar to "grid search" where you set different threshold and pay close attention to the TPR and FPR values and draw the ROC curve before comparing each one of them.

## Q2 What’s the difference between logistic regression and support vector machines? What's an example of a situation where you would use one over the other?
Logistic regression and SVM are both supervised ML methods that are used for classification task. The main difference between these 2 is that 
1. SVM tries to finds the “best” margin (distance between the line and the support vectors) that separates the classes and this reduces the risk of error on the data, while logistic regression does not, instead it can have different decision boundaries with different weights that are near the optimal point.

2. SVM is based on geometrical properties of the data while logistic regression is based on statistical approaches.

3. The risk of overfitting is less in SVM, while Logistic regression is vulnerable to overfitting.

One situaion where we would use SVM over logistic regression is when we have small numberof features and intermediate number of training samples. Or when we have a huge number of features with little training samples, that's when we use logistic regression

## Q3 Explain ICA and CCA. How do you get a CCA objective function from PCA?
PCA stands for principal component analysis, using principal components to reduce dimension of the data. ICA is Independent Components Analysis and CCA is Canonical Correlation (Covariance) Analysis. ICA is a rotation of components produced via PCA. The components are rotated under the condition they maximize statistical (probabilistic) independence. CCA is the Singular Value Decomposition of (X'X)-1 (X'Y) (Y'Y)-1, where you are trying to identify components that maximize the multivariate correlation between two data sets.

In PCA, a set of variables predict themselves: they model principal components which in turn model back the variables, you don't leave the space of the predictors and (if you use all the components) the prediction is error-free. In multiple regression, a set of variables predict one extraneous variable and so there is some prediction error. In CCA, the situation is similar to that in regression, but (1) the extraneous variables are multiple, forming a set of their own; (2) the two sets predict each other simultaneously (hence correlation rather than regression); (3) what they predict in each other is rather an extract, a latent variable, than the observed predictand of a regression 

## Q4 What is the relationship between PCA with a polynomial kernel and a single layer autoencoder? What if it is a deep autoencoder?
 
## Q5 What is "random" in random forest? If you use logistic regression instead of a decision tree in random forest, how will your results change? 

## Q6 What is backpropagation?
Well, usually when we design a neural network, we have intiial weights. And these weights may result in a significant error in the model output. So one way to reduce the error is through BP, where minimum value of the error function in weight space is calculated using gradient descent. This could increase the accuracy of the predicition using the model.

## Q7 What’s the difference between Generative and Discriminative models? 
Generative models consider the joint probability distribution while discriminative models consider the conditional probability distribution. In general, generative models try to figure out how to generate the data to perform classification, while discrimative models simply differentiates between data of different classes.

## Q8. When should we use feature scaling? And what's the difference between scaling and normalization?
Feature scaling is important when we use machine learning models that calculates distance, for instance, kNN and PCA. Tree-based model like Random Forest ot LDA doesn't really require rescaling. Difference between feature scaling and normalization is in scaling, the range of the data is changed, while in normalization, the distribution of the data is changed.

## Q9. What is active learning in deep learning models?
The motivation behind active learning is, if we have billions of images that needs labeling, how can we speed up the process? It can be really time-consuming if it is pure manual work, and this is where active learning comes in. The selected machine learning model is trained on the available, manually labeled data (usually only a small portion of the data)and then applied to the remaining data to automatically define their labels. The quality of the model is evaluated on a test set that has been extracted from the available labeled data. 
