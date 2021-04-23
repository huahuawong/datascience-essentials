### Q1. What is A/B testing?
An experiment technique to determine if a new design brings improvement based on a given metric.
We should formulate A?B Testing using Population, Intervention, Comparison, Outcome, Time = PICOT


### Q2. Explain what is p value?
Before we talk about what is p value, we should understand what is null hypothesis. Null hypothesis is a hypothesis that states that when we
introduce a change to a system, there will be zero to none effects. If there really is no effect, we can declare that the null 
hypothesis is true, and p value evaluate how well our sample data support the null hypothesis.

High P values (typically > 0.05) means that data is likely with a true null hypothesis
Low P values (typically ≤ 0.05) means that data is unlikely with a true null hypothesis

### Q3. What is bias?
Bias is the dfference between the model prediction and the actual values that it is trying to predict. Bias tends to oversimplifies the model

### Q4. What is variance?
Variance is the variability of the model prediction given a testing dataset. Models with high variance tend to focus on training data, but not testing. This results in overfitting.

### Q5. What is the bias and variance tradeoff?
The need to strike a balance between these two, A good balance would ensure that the model isn’t underfitting or overfitting

### Q6. What’s confidence interval?
Let's say we want to have an idea of the population mean, but we can't be 100% certain on the actual number, so we get a range of values that we're confident that the actual value will be within this range

### Q7. What are the factors that affect confidence interval?
1. Sample size - higher sample size would result in wider CI and vice-versa
2. Variation - Low variation would result in narrower CI and vice-versa

### Q8. What is maximum likelihood estimation? Could there be any case where it doesn’t exist?
MLE is essentially finding a parameter which maximizes the likelihood that the process described by the model produced the data that were actually observed.
It can readily be generalized to the case where we are trying to estimate the probability density of observing a single data point x, that is generated from a Gaussian distribution, i.e. P(x; mean, standard deviation). It can be used to estimate parameters for machine learning models such as Logistic Regression

It doesn't exist when we want to model that with a mixture of Gaussians, where the estimation diverges. Let's say if we only have one Gaussian on one of the 
training examples. During MLE, the σ of the model will decrease, increasing the likelihood. As a result, other instances would become less likely.

### Q9. Type 1 Error vs Type 2 Error
Type 1 Error refers to false positive while Type 2 Error refers to false negative

### Q10. What is the difference between convex and non-convex cost function; what does it mean when a cost function is non-convex?
We can think of convex cost function as a convex graph, i.e. a nice U shape curve, whereas non-convex means it isn't convex-shaped, and it's often been observed as having a wavy line. But what if you can't plot the graph of the cost function? If the function is twice differentiable, and the second derivative is always greater than or equal to zero for its entire domain, then the function is convex.

When a cost function is non-convex, it means that there’s a likelihood that the function may find local minima instead of the global minimum. Note that both l2 and l1 regularizations are convex, and one of the example of non-convex cost function is the neural network, which depending on the hidden layers, may be non-convex.

### Q11. What is *l0 norm*, *l1 norm* and *l2 norm* exactly?
First, what is a norm? In mathematics, a norm is a function from a real or complex vector space to the nonnegative real numbers that behaves in certain ways like the distance from the origin: it commutes with scaling, obeys a form of the triangle inequality, and is zero only at the origin

The *L0 norm* is the number of non-zero elements in a vector. *l1 norm* is the sum of the magnitudes of the vectors in a space, also known as Manhattan norm. *l2 norm* is the shortest distance to go from one point to another, also known as Eucliden norm.

### Q12. You are drawing from a normally distributed random variable X ~ N(0, 1) once a day. What is the approximate expected number of days until you get a value of more than 2?

The probability of drawing a value of more than 2 would be: θ≡P(Xi>2)=1−Φ(2)≈0.02275. (Refer to z-score probability table)

Notice that we are looking at geometric distribution here, so the expected number of days would be (1/θ) = 43.95. Geometric distribution means how many trials/ failures until you get one success. In this case, success means getting a value of more than 2.


### Q13. When do we use different statistical distribution?
It depends. Sometimes we can use normal/ Gaussian distribution to get some quick insights. For instance, in a normal distribution, 68% of the observations fall within +/- 1 standard deviation from the mean. For binary classification problems, binomial distribution can be used as the distribution foundation. If a regression model uses least squares cost functions (for instance, linear regression), it is assumed that the residuals would have a normal distribution with a mean of zero. 

### Q14. What is bootstrapping?
Bootstrapping is one of the resampling techniques and it is basically sampling with replacement. What is sampling with replacement? Think about drawing a ball from a basket with 5 balls. After each draw, you put the ball back into the basket. That means each draw is independent of one another.

There is an ensembled method called Bootstrap Aggregation or bagging. The Bootstrap Aggregation algorithm is done by creating multiple different models from a single training dataset.

### Q15. How do we determine outliers?
In Python, we can use boxplots to identify outliers. By math concept, multiplying the interquartile range (IQR) by 1.5 will give us a way to determine whether a certain value is an outlier. If we subtract 1.5 x IQR from the first quartile, any data values that are less than this number are considered outliers. Same thing goes for the third quartile, i.e. 1.5 x IQR + third quartile.
