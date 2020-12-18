---
layout: page
title: Will clients subscript a term deposit ?
cover-img:  /assets/img/bank.jpg
thumbnail-img: /assets/img/bank.png
share-img: /assets/img/bank.jpg
subtitle: Using machine learning methods to predict whether a client will subscribe a term deposit or not
---

## Table of Contents
#### 1. Introduction

#### 2. Datasets

#### 3. Data Wash

3.1 Data drop
3.2 Data Visualization and Feature Selection
3.3 Unknown Filling
3.4 Data Standardization

#### 4. Model Implementation

4.1 Data Oversample
4.2 Logistic Regression
4.3 Random Forest

#### Performance Comparision and Analysis
#### Conclusion

## 1. Introduction

Classic statistical models such as logistical regression are proven to perform poorly in predicting rare events like civil war onset. In the paper "Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data", the author compared the performance of random forest with three versions of logistic regression of different features and hyperparameters. The results show that random forest outperforms all the logistic regression models in terms of prediction accuracy as well as casual processes iterpretation. Therefore, it is worthwhile to expand the research to other fields to see if random forest woulc also provide more accurate predictions than logistic regression in out-of-sample data, as shown in the paper.

For this purpose, we proposed an extension project analyzing the performance of random forest on predicting rare events in marketing field. Specifically, our task is to predict whether a client will subscribe a term deposit or not based on information of the client. To achieve this, we first collect dataset from direct marketing campaign of a bank institution in Portugues and preprocess it. Since the dataset is highly imbalanced, it can be viewed as a rare event. We then apply random forest and logistic regression to make predictions and compare the results of these two methods. Concretely, we would like to adress the following research questions: 

- What is the accuracy of random forests in predicting the rare event in a new field?
- Can random forests predict the subscriptions of the bank term deposit efficiently and generally?
- To what extent does random forest outperform logistic regression method in predicting the rare events, subscription of the bank term deposit?
- Is there a causal effect of a client subscribing to a bank term deposit?
- What is the most important factor affecting a client subscription to a term deposit? Which clients are more likely to subscribe to a term deposit?

The data story is organized as follows: In section II, we will make a short description of our datasets. Section III will introduce the methods used in preprocessing the data. Section IV will implements logistic regression and random forest to the data. Section V compares and analyzes the performance of the models. Finally, we will conclude our project in section VI.

## 2. Datasets

We collect the Bank Marketing Data Set from the UCI Machine Learning Repository. The datasets is based on direct marketing campaign of a bank. Basically, the employees from the bank would make phone calls to the clients and convince them to make a term deposit with their bank. After the phone calls, the decision of the client will be noted - whether they accept the subscribtion or not, together with the information of the client. In out datasets, there are 41188 samples in total, and each with 20 attributes and a final decision: whether they subscribed the term deposit ("yes") or not ("no"). The attributes can be divided into the following categories:
 - Client's personal information: Age, job, marital status, education, loan status and so on;
 - Campaign activities: When and how to contact, the duration of the phone call;
 - Social and economic environment data: Employment variation rate, consumer price index, consumer confidence index, euribor 3 month rate, number of employees;

The dataset is highly imbalanced since only a few people would make the bank term deposit. The ratio of accepting the subscription (‘yes’) and rejecting the proposal ('no') is roughly 1:8 in the datasets. In addition, there are missing values appeared in some attributes marked with "unknown", since some people are unwilling to disclose their private information to the bank. Therefore, before we apply the statistical models, we have to preprocess the data and select useful features, which is also referred to as feature engineering.

## 3. Data Wash
### 3.1 Data drop 
The first step is to drop redundant and useless data. To do this, we first discard duplicated rows, and check if the data contains 'null' and mark them as unknowns. 

### 3.2 Data Visualization and Feature Selection
Next, we visualize the data, analyze it and select useful features. Beforehand, just by looking at the data information, we have full reason to discard the attribute "duration" -- the duration of each phone call, since it is unpredictable before each call. Besides, the duration is highly correlated to the final decision -- the longer the duration, the higher the possibility that a client would subscirbe. Therefore, to ensure a realistic predictive model, we would discard this attribute. 

Now we can start to analyze the features by visualizing the data. We first viusalize the distribution of the result. As we can see, most of the custom (88.7%) will reject the proposal of term deposit subscription. 

![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/yes_no.png?raw=true)

The next step is to visualize categorical variables. There are 10 categorical variables in total. We will choose some of them (most with unknowns), analyze the data and check if there is irrelevant or inappropriate feature to discard.

#### Job
The first features we analyze is job. As shown in the figure (and all the other figures), the left figure is the distribution of different categories, with x-axis being the categories and y-axis being the total count. The right figure shows how likely people with this category would accept (positive values in x-axis) or reject (negative values in x-axis) the subscription. The longer the bar, the higher the possibility. For example, from the visualization of job we can see that blue-collar workers are ver unlikely to subscribe a term deposit. Conversely, people who are retired, or students, or work in admin show a tendency to make a term deposit. Interestingly, people who work in services industry are tend to reject the term deposit. In addition, we can see there is a small portion of unknowns in the job attribute, but they do not seem to affect the result.

![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/jobs.png?raw=true)

#### Marital
Marital status is an effceting factor as well. People who are married or have been married before tend to reject the proposal, while singles would prefer to make a subscription. Again, there are a small portion of unknowns, but they do not take too much importance.

![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/marital_status.png?raw=true)

#### Education
From the distribution of education we can see that people with university degree are highly likely to make the subscription, which also make sense in real life. However, as we can see from the left figure, the proportion of unknowns is not negligible, and from the right figure we see that people with unknown education are tend to make the deposit term. This reminds us that we should deal with those unknowns carefully instead of directly discarding them.

- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/education.png?raw=true)

#### Default
The default variable shows whether the client has credit in default or not. Unlike before, the data distribution of this attribute is highly imbalanced. The number of unknowns take a significant amount that is unlikely to ignore. Besides, people with unknowns default are highly tend to rejecting the term deposit. Therefore, the unknowns of this attribute are too unpredictable so that it could be an inference to the final prediction. After discussing, we decide to discard this attribute.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/default.png?raw=true)

#### Housing and Loan
The attributes housing and loan can be analyzed together since they have similar properties. As we seen from the figures, people who have house and no loan are very likely to make a term deposit. Besides, the unknowns also take a small proportion and tendency in both attributes, but not negligible. 
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/housing.png?raw=true)
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/loan.png?raw=true)

### 3.3 Unknown Filling

The most challenging part of this project is to deal with the unknowns. There are many ways to handle it. Firstly, and most easily, we can directly delete all the samples that contain the unknowns. However, as we analyzed before, those unknowns take a noteworthy part in the whole dataset, and simply discarding them would definitely have negative impact on the prediction accuracy. The second method we try is to use decision tree to predict the unknowns. But the results also do not seem ideal (since the cross validation score for predicting unknowns is only 0.3-0.5). Finally, we decide to use the function "SimpleImputer" in sklearn to replace the unknowns with the most frequent value along each coloumn. As we can see later, the performance of our statistical models is satisfactory on our processed dataset!


### 3.4 Data Standardization

After filling the unknown data, we ploted the distribution of features with numerical values: 
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/before_standrization.png?raw=true)
- We can easily find that the distribution of the dataset are not uniform. For example, client's age is ranging from 15 to 90, whereas the values of "campaign" stays mostly within the interval 0 to 20. Therefore, standardization of the data is needed. In addition, the attribute "pdays" indicates the number of days that passed by after the client was last contacted from a previous campaign, and 999 means that the client was not previously contacted. For better interpretation of the data, we convert the numerical values into category by replacing all the value 999 with "no", and the rest with "yes". The result is shown below:
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/after_standrization_dis.png?raw=true)
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/after_standrization.png?raw=true)

## 4. Model Implementation
After preprocessing the data, we apply logistic regression and random forest model to the data to make predictions.

### 4.1 Data Oversample
As we showed before, the data is highly imbalanced. The model would therefore tend to to predict the outcome that has the lager portion in the dataset (in this case, model will tend to predict "no"). If we directly apply this model to make prediction, the bank would be at risk of losing their potential client. To address this problem, we use SMOTE algorithm to oversample the minority class. SMOTE algorithm generates new data based on the distribution of features using K-nearest neighbor. After oversampling, the ratio between majority class (not subscribing) and minority class (subscribing) will become 4:1, which is much more balanced than the original ratio 8:1. With the help of SMOTE, the potential of logistic regression model and random forest model can be better exploited.

### 4.2 Logistic Regression
We first apply logistic regression to the data. To better compare the performance of the linear and nonlinear models, we set different penalty methods. Here we use L1 and  L2 norm as our penalty terms. To find the best parameter for this model, grid search is also applied. Besides, we use 10-fold cross-validation to avoid overfitting. Finally, we calculate the FPR and TPR respectively. 

### 4.3 Random Forest
Next, we implement a random forest algorithm to the data to see its performance. As before, we use grid search to find the best hyperparameters (in this case, the number of trees and the max depth of each tree), and use 10-fold cross-validation to make predictions. After that, the FPR and TPR score will be calculated. 

## 4. Performance Comparison and Analysis
To analyze the performance of the models, we first plot the ROC curve to compare the 
#### 4.1 ROC Curve
The ROC curve is drawn according to TPR and FPR data from these two models.
The random forest can better predict our result with an accuracy of 90%, which is 10% percent better than linear regression.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/ROC_curve.png)
- Also, using the F1-score as the performance measure, we systematically compare the performance of Random Forests with logistic regression and L1-regularized logistic regression. In our experiments, we vary the ratio of the training set such that the percentage of the entire data used for training ranges between 0.2 and 0.8.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/F1_score.png)
#### 2.0 Importance of variables
The importance of different variables is another point we want to discuss. The mean decrease in the Gini Score is the predictive accuracy lost by omitting a given predictor from the tree used to generate predictions about the class.
Therefore, we could take the decrease in mean Gini score as an important factor that could be used to describe the importance of a single variable.  
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/Feature_importance.png)

#### 3.0 Result Compare
To understand how single variables can influence our prediction result, the partial dependence plot was used. By drawing these pictures, an intuitive story was told.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/Partial_Dependence.png)

### Which factor matters
From what the image shows, we can know whether a salesperson would make a business depends on multiple important variables. The features including 'euribor3m', 'age', 'campaign', 'nr.employed', 'job' and 'education' are the most important. The 'euribor3m' is the strongest predictor. According to the partial dependence plot, the lower euribor 3-month rate can attract clients to subscribe to the bank term deposits. 

Then, age can also be a strong predictor. According to the distribution of age and the partial dependence plot, the youth and the elderly will be more likely to subscribe to the bank term deposits.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/old_money.jpg)
- The 'campaign' feature indicates that a relatively low number of contacts performed during this campaign and for this client can increase the possibility of subscribing to the bank term deposits.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/campaign.png)
- 'nr. employed' is the number of employees - quarterly indicator. Therefore, a low number of employees can attract clients to subscribe to bank term deposits.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/empbanking.jpg)
- The 'job' and 'education' are the categorical variables. According to the distribution of the two variables, if the client's job is retired, the client is more likely to subscribe to the bank term deposits. Also, if a client has a university degree, the client is more likely to subscribe to the bank term deposits.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/Education_money.jpg)
According to the partial dependence plot, the categorical variables exhibit significant nonlinear relationships, so the changes of these categorical variables would not be captured by a linear logistic model. Also, the numeric variables except 'contact' present significant nonlinear relationships. Therefore, the logistic regression model provides a lower AUC score and F1 score.


