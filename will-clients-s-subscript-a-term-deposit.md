---
layout: page
title: Will clients subscript a term deposit ?
subtitle: Using machine learning methods to predict whether a client will subscribe a term deposit or not
---

## Table of Contents
1. Introduction

2. Datasets

3. Data Wash
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

The dataset is highly imbalanced since only a few people would make the bank term deposit. The ratio of accepting the subscription (‘yes’) and rejecting the proposal ('no') is roughly 1:8 in the datasets. In addition, there are missing or unknown values appeared in some attributes because some people would not like to tell their private information to the bank. Therefore, before we apply the statistical models, we have to preprocess the data and select useful features, which is also referred to as feature engineering.

## 3. Data Wash
- As we can see in the dataset, there are 20 columns of information recorded. It contains information from different customers, 
some of which are number data and some others are text data. The first step to process data is to wash data.
### 3.1 Data drop 
The first step is to drop redundant and useless data. To do this, we check if there are any duplicated rows and discard them. We also check if the data contains 'null' and mark them as unknowns. 
The next thing we do is feature selection. Beforehand, just by looking at the information, we have full reason to discard the attribute "duration" -- the duration of each phone call, since it is unpredictable before each call. Besides, the duration is highly correlated to the final decision -- the longer the duration, the higher the possibility that a client would subscirbe. Therefore, to ensure a realistic predictive model, we would discard this attribute. 

### 1.0 Data Visualization and Selection
- Now, we see the data is roughly processible. Therefore, let's take a good at what we want predict. The following image describing the distribution of our result. Obviously, most of custom(88.7%) will reject the proposal of salesperson.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/yes_no.png?raw=true)
- Here are the categorical variables.
- The next step we do is to analyze the relevance of them between results. If the feature has nothing to do with the result, then we can drop it. Here goes the analysis result.
- The first features we analyze is the jobs. From that we can know, the tendency of subscrption has obvious difference amoung different occupancy, where blue collars are the people who are least likely to spend money on that.
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/jobs.png?raw=true)
- Marital status is a effceting factor as well. Single are likely to subscribe bank term deposit when getting a sell call, while married person is on the contrary. 
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/marital_status.png?raw=true)
- After analyzing all listed features, we find that 'y' depends on all the categorical varibles, and we do not need to drop any of them.


### 2.0 unknown filling
- Unknown data is another problem we are facing. Due to privacy sensitivity of some customers, they are unwilling to disclose more personal information, so a lot of data cannot be recorded. Therefore, there are blank information being marked as unknown. 'Unknown' category in ‘education’, ‘job’, ‘housing’, ‘loan’, ‘default’, and ‘marital’ can not help us use predictable model to determine which clients are more likely to subscribe a term deposit.
- 这里加一个unknown的扇形图！
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png?raw=true)
- As can be seen from the above figure, the unknown value accounts for a large proportion of the data. 
This is an unappropriate situation we want to see. Too many unknown factors will affect the result of judgment, 
so it is particularly important to find a reasonable method to fill in the unknown value.
Here, we decided to use imputer to fill the missing values based on the most frequenct category in the column
```
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
for i in missing_col:
    bank_additional_full[i] = imp.fit_transform(bank_additional_full[i].values.reshape(-1,1))
```

### 3.0 Data Standardization and  Nummerize 
- The last step before going into ML filed is the features processing. 

From the table, there are multiple text values, that can not be processed by classical data processing method.
Therefore, we should find an appropriate method to process these kinds of data.
The method we decide to take is to encoding. We use number value to represent the text.
For example, the feature jobs contains 12 different kinds of jobs name. 
We decide to use number to reprent the text here.
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
- Then we used the same method to process all text feature. 
- Then we take a look at distribution of other numeric features. It is obvious that the numeric variables are not uniformly distributed. Therefore, we will standardize them. In addition, the 'pdays' variables is distributed discretely. We convert the 'pdays' to categorical variable, in which, 999 means client was not previously contacted. Therefore, we use 'no' representing 999 and 'yes' representing others.
```
bank_additional_full.loc[bank_additional_full['pdays'] < 30, 'pdays'] = 'yes'
bank_additional_full.loc[bank_additional_full['pdays'] == 999, 'pdays'] = 'no'
bank_additional_full.pdays.astype("category")
```
- After standrization we get the following table
- ![Image](https://github.com/LIUQyou/LIUQyou.github.io/blob/master/assets/img/after_standrization.png?raw=true)

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
