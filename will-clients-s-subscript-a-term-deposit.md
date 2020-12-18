---
layout: page
title: Will clients subscript a term deposit ?
subtitle: Using machine learning methods to predict whether a client will subscribe a term deposit or not
---

## Table of Contents
### 1. Introduction

### 2. Datasets

### 3. Data Wash

## 1. Introduction

Classic statistical models such as logistical regression are proven to perform poorly in predicting rare events like civil war onset. In the paper "Comparing Random Forest with Logistic Regression for Predicting Class-Imbalanced Civil War Onset Data", the author compared the performance of random forest with three versions of logistic regression of different features and hyperparameters. The results show that random forest outperforms all the logistic regression models in terms of prediction accuracy as well as casual processes iterpretation. Therefore, it is worthwhile to expand the research to other fields to see if random forest woulc also provide more accurate predictions than logistic regression in out-of-sample data, as shown in the paper.

For this purpose, we proposed an extension project analyzing the performance of random forest on predicting rare events in marketing field. Specifically, our task is to predict whether a client will subscribe a term deposit or not based on information of the client. To achieve this, we first collect dataset from direct marketing campaign of a bank institution in Portugues and preprocess it. Since the dataset is highly imbalanced, it can be viewed as a rare event. We then apply random forest and logistic regression to make predictions and compare the results of these two methods. Concretely, we would like to adress the following research questions: 

1. What is the accuracy of random forests in predicting the rare event in a new field?

2. Can random forests predict the subscriptions of the bank term deposit efficiently and generally?

3. To what extent does random forest outperform logistic regression method in predicting the rare events, subscription of the bank term deposit?

4. Is there a causal effect of a client subscribing to a bank term deposit?

5. What is the most important factor affecting a client subscription to a term deposit? Which clients are more likely to subscribe to a term deposit?

The data story is organized as follows: Firstly, we will make a short description of our datasets, and introduce the methods used in preprocessing the data...????????

## 2. Datasets

We collect the Bank Marketing Data Set from the UCI Machine Learning Repository. The datasets is based on direct marketing campaign of a bank. Basically, the employees from the bank would make phone calls to the clients and convince them to make a term deposit with their bank. After the phone calls, the decision of the client will be noted - whether they accept the subscribtion or not, together with the information of the client. In out datasets, there are 41188 samples in total, and each with 20 attributes and a final decision: whether they subscribed the term deposit ("yes") or not ("no"). The attributes can be divided into the following categories:
 - Client's personal information: Age, job, marital status, education, loan status and so on;
 - Campaign activities: When and how to contact, the duration of the phone call;
 - Social and economic environment data: Employment variation rate, consumer price index, consumer confidence index, euribor 3 month rate, number of employees;
 - Other attributes: ????????

The dataset is highly imbalanced since only a few people would make the bank term deposit. The ratio of accepting the subscription (‘yes’) and rejecting the proposal ('no') is roughly 1:8 in the datasets. In addition, there are missing values appeared in some attributes marked with "unknown", since some people are unwilling to disclose their private information to the bank. Therefore, before we apply the statistical models, we have to preprocess the data and select useful features, which is also referred to as feature engineering.

## 3. Data Wash
### 3.1 Data drop 
The first step is to drop redundant and useless data. To do this, we first discard duplicated rows, and check if the data contains 'null' and mark them as unknowns. 

### 3.2 Data Visualization and Feature Selection
Next, we visualize the data, analyze it and select useful features. Beforehand, just by looking at the data information, we have full reason to discard the attribute "duration" -- the duration of each phone call, since it is unpredictable before each call. Besides, the duration is highly correlated to the final decision -- the longer the duration, the higher the possibility that a client would subscirbe. Therefore, to ensure a realistic predictive model, we would discard this attribute. 

Now let's take a good look at our data. We first viusalize the distribution of the result. As we can see, most of the custom (88.7%) will reject the proposal of term deposit subscription. 

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

猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫

#### Default
The default variable shows whether the client has credit in default or not. Unlike before, the data distribution of this attribute is highly imbalanced. The number of unknowns take a significant amount that is unlikely to ignore. Besides, people with unknowns default are highly tend to rejecting the term deposit. Therefore, the unknowns of this attribute are too unpredictable so that it could be an inference to the final prediction. After discussing, we decide to discard this attribute.

猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫

#### Housing and Loan
The attributes housing and loan can be analyzed together since they have similar properties. As we seen from the figures, people who have house and no loan are very likely to make a term deposit. Besides, the unknowns also take a small proportion and tendency in both attributes, but not negligible. 

猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫猫

### 3.3 Unknown Filling

The most challenging part of this project is to deal with the unknowns. There are many ways to handle it. Firstly, and most easily, we can directly delete all the samples that contain the unknowns. However, as we analyzed before, those unknowns take a noteworthy part in the whole dataset, and simply discarding them would definitely have negative impact on the prediction accuracy. The second method we try is to use decision tree to predict the unknowns. But the results also do not seem ideal (since the cross validation score for predicting unknowns is only 0.3-0.5). Finally, we decide to use the function "SimpleImputer" in sklearn to replace the unknowns with the most frequent value along each coloumn. As we can see later, the performance of our statistical models is satisfactory on our processed dataset!


### 3.0 Data Standardization and  Nummerize 
- After filling unknown data, we ploted the distribution of data from different features.

- From the image above, we can easily find that the distribution of these dataset are uniform. For example, the range of data age is from 15 to 80. 
Compared with that, the data of campaign are stay within the interve 0 to 20.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/before_standrization.png?raw=true)
Therefore, we decide to standardize our dataset with mean of 0, and vriance of 1.
- After standrization we get the following table
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/after_standrization.png?raw=true)
- It worth to mention that the numeric variables 'pdays' are not continouly distributed, which can be divided into 2 categores smaller than 30 and 999. We convert the 'pdays' to categorical variable, in which, 999 means client was not previously contacted. 
Here we handle this problme by using 'yes' and 'no' to represent its values.
```
bank_additional_full.loc[bank_additional_full['pdays'] < 30, 'pdays'] = 'yes'
bank_additional_full.loc[bank_additional_full['pdays'] == 999, 'pdays'] = 'no'
bank_additional_full.pdays.astype("category")
```
- Besides, another problem we need to tackle is the text values, as text values can not be processed by classical data processing method.
Therefore, we should find an appropriate method to process these kinds of data.
The method we decide to take is encoding. We use number value to represent the text.
For example, the feature jobs contains 12 different kinds of jobs, then we can assign each job a value from 1 to 12. 
We applied this method to all text values.

```
#### Label Encoder
# Label the dependent varaible
df['y'] = df['y'].apply(lambda x: 1 if x=="yes" else 0)
#### Label Encoder for categorical varaibles
cat_cols = ['job','marital','education','loan','contact','month','poutcome',"day_of_week" ,"housing",'pdays' ]
new_df = bank_additional_full.copy()
le = preprocessing.LabelEncoder()
for col in cat_cols:
    #print(col)
    df[col] = le.fit_transform(df[col])
```

## Model and implementation
- After completing data preprocessing, we set up two models based on classical logistic regression and random forest model. 
Then we inport features as predictor and use y as target to indicate whether they would buy products. 
### Data oversample
As the distribution of y shows, there are too few people answer with yes.
We proposed another method to handle oversample our data, the SMOTE.
SMOTE is a Knn based algorithm, which use algorithm to extend our dataset, by generating different new data set with the distribution of features.
We used SMOTE algorithm to oversample the minority class. Then, the ratio between majority class and minority class will be 3 : 1.
```
smote = SMOTE(sampling_strategy = 0.25)
X, y = smote.fit_sample(X, y)
```
After data oversampling, the ratio of subscribed (‘no’) bank term deposit and not ('yes') subscribed in the data is roughly 4.000109481059777.
With the help of SMOTE, the potential of logistic regression model and random forest model can be better exploied.
### Logistic Regression
- The first model we used is logistic regression model, we used the sigmoid function as activation to process data. With the help of the sklearn library, we can easily get our prediction. In judge the performance of our model, we used 10-fold cross-validation and generate the probability as our result. 
Then we calculate the FPR and TPR respectively.
To better compare the performance of linear and unlinear model, we set different penalty parameter. Here, L1, L2 are all implemented.
And GridSearch method is used to search best parameter for this model.

## Model and implementation
- After completing data preprocessing, we set up two models based on classical logistic regression and random forest model. 
Then we input features as input and whether they buy products as y. 
### Logistic Regression
- Here we establish a logistic regression model, use the sigmoid function as activation to process data. With the help of the sklearn library, we can easily get our result of the prediction. In order to judge the performance of our model, we used 10-fold cross-validation and generate the probability as our result. 
Then we calculate the FTR and TFR respectively. We get their values are and .
### Random Forest
- Random forest is another method we choose to analyze our result. Due to its good performance to predict rare events, we believe Random Forest will give us a good prediction result.
Random forests are an ensemble learning method for classification, regression, and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees. We generate about XXX trees in this model.
### Data Analysis and Visulization
- After the model analysis, we get both the TPR and FPR from both models. 
#### 1.0 ROC Curve
- Here goes the ROC curve.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- As can be seen from roc curve, XXX.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
#### 2.0 Importance of variables
- Next, we decided to judge the importance of different variables. We judge the loss of accuracy by eliminating one of the elements.
- The figure below shows the average decrease in the prediction accuracy of the first X variables selected by the random forest, called the Gini score. The Gini score is calculated as follows.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- Then we sort the gini score of different features and draw the following picture.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
#### 3.0 Result Compare
- By comparing model performance from different models.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- We can find that ***********************************
- We think that .....
- Therefore, we have reason to believe that ******************************.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- From the feature importance and relevance with the result. We can easily find that the features like X, Y will exert a huge influence on the result prediction, where X is positively related to the result and Y is negative to that. Here, we can find that when ( Real Situation), people tend to subscript a new term deposit. If a man (real situation),...
- Moreover, time is an effective factor as well. If the seller tries to call someone on Monday, they are highly likely to be rejected. This is in line with our feeling, after all, who would be in the mood to answer the call from a salesperson during the busiest Monday. Another interesting point is that the month will influence the performance of the salesperson. May is quite an interesting month when people get a sales call, blablablabla. In contrast, if they get a call at %%%%, they are highly likely to XXXX.


[Link](url) and ![Image](src)


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/LIUQyou/ADA_P4/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
