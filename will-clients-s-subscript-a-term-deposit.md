---
layout: page
title: Will clients subscript a term deposit ?
subtitle: Why you'd want to go on a date with me
---


## Abstract

We propose to study the power of random forest in predicting rare events in another field. We will use a dataset recording direct marketing campaigns (bank term deposit) of a Portuguese banking institution. The subscriptions of the bank term deposit are based on phone calls, so this is a rare event. We plan to apply random forest and logistic regression method to predict if a client will subscribe a term deposit in the bank, and compare their predictive accuracies. Then, we will analyze the causal effects in the subscription of a term deposit from a bank. Finally, we could indicate which clients are more likely to subscribe for term deposits.

## Research questions
1. What is the accuracy of random forests in predicting the rare event in a new filed?

2. Can random forests predict the subscriptions of the bank term deposit efficiently and generally?

3. To what extent does random forest outperform logistic regression method in predicting the rare events, subscription of the bank term deposit?

4. Is there a causal effect of a client subscribing a bank term deposit?

5. What is the most important factor affecting a client subscribe a term deposit? Which clients are more likely to subscribe a term deposit?

## datasets
-	Bank-additional-full.csv from “Bank Marketing Data Set” UCI machine learning repository. There are 41188 observations each with 20 features such as a client’s age, sex and job. The observations are ordered by date (from May 2008 to November 2010). The dependent variable is ‘y’ indicating whether a client subscribe the bank term deposit. 
The dataset is highly imbalanced because few people subscribe the bank term deposit. The ratio of subscribed (‘Yes’) bank term deposit and not ('no') subscribed in the data is roughly 1:8. In addition, there are missing or unknown values such as ‘education’, ‘housing’ and ‘loan’ in the dataset because some people would not like to tell their private information to the banking representative. We will discard the rows containing missing values. 

## Data preprocessing
- As we can see in the dataset, there are 20 columns of information recorded. It contains information from different customers, 
some of which are number data and some others are text data. The first step to process data is to wash data.
### 1.0 Data Visualization and Selection
- Here are the categorical variables.
- The first step we do is to analyze the relevance of them between results. If the feature has nothing to do with the result, then we can drop it. Here goes the analysis result.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- From what we see above, not all the features will impact the result.
For example, the custom's XXX will not influence their decision to subscript a new term deposit.
While marital status is an important factor. Apparently, if a customer has already get married. He is highly likely to buy a new business product.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- The other influence factor contains XX,XX,YY,ZZ.
As we know, if one factor has nothing to do with the result. We can simply delete it to reduce computation complexity and avoid noise.
We drop unnecessary features and keep the rest. Now, we get about NN features for the next step.
### 2.0 Data Nummerize and Standardization
- Another problem we are facing now is the features processing. 
From the table, there are multiple text values, that can not be processed by classical data processing method.
Therefore, we should find an appropriate method to process these kinds of data.
The method we decide to take is to encoding. We use number value to represent the text.
For example, the feature jobs contains NN kinds of different jobs name. 
We decide to use number to reprent the text here.
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- Then we used the same method to process all text feature. 
### 3.0 unknown filling
- Unknown data is another problem we are facing. Due to privacy sensitivity of some customers, they are unwilling to disclose more personal information, so a lot of data cannot be recorded. Therefore, there are blank information being marked as unknown.
![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
- As can be seen from the above figure, the unknown value accounts for a large proportion of the data. 
This is an unappropriate situation we want to see. Too many unknown factors will affect the result of judgment, 
so it is particularly important to find a reasonable method to fill in the unknown value.
Here, we decided to use the random forest method to predict the exact eigenvalues to fill the data.
```py
// Javascript code with syntax highlighting.
var fun = function lang(l) {
  dateformat.i18n = require('./lang/' + l)
  return true;
}
```
- ![Image](https://github.githubassets.com/images/icons/emoji/octocat.png)
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
