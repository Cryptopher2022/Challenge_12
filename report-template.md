# Module 12 Report Template

## Overview of the Analysis

To include:
An overview of the analysis: Explain the purpose of this analysis.
The results: Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of both machine learning models.

A summary: Summarize the results from the machine learning models. Compare the two versions of the dataset predictions. Include your recommendation for the model to use, if any, on the original vs. the resampled data. If you don’t recommend either model, justify your reasoning.

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of this analysis is to refine the processing of real time data for the status of existing loans.  In a single column, there is a loan_status coded by either a "0" or a "1".  Zero means "non-risky" and One equals a "risky" loan.  There are many more non-risky loans than non-risky so being able to take in new data and process it correctly is paramount for this feature.  

* Explain what financial information the data was on, and what you needed to predict.
As mentioned above, the information was on a set of existing loans and the prediction is used to identify risky loans correctly, although there is much more non-risky loans than risky loans throughout the data set.  Being able to predict which ones are risky is a critical feature of this financial analysis.  

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
Out of a total package of loans totalling over 77,536 with 75,036 identified as non-risky and 2,500 identified as risky.  3.2% risky loans is a small amount to get correctly each time, thus the confirmation of the utility of the model is critical to the success of this endeavor.  

* Describe the stages of the machine learning process you went through as part of this analysis.

Read the lending_data.csv data from the Resources folder into a Pandas DataFrame.

Create the labels set (y) from the “loan_status” column, and then create the features (X) DataFrame from the remaining columns.

Note A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting.

Check the balance of the labels variable (y) by using the value_counts function.

Split the data into training and testing datasets by using train_test_split.

Create a Logistic Regression Model with the Original Data
Employ your knowledge of logistic regression to complete the following steps:

Fit a logistic regression model by using the training data (X_train and y_train).

Save the predictions on the testing data labels by using the testing feature data (X_test) and the fitted model.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.

Answer the following question: How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

Predict a Logistic Regression Model with Resampled Training Data
Did you notice the small number of high-risk loan labels? Perhaps, a model that uses resampled data will perform better. You’ll thus resample the training data and then reevaluate the model. Specifically, you’ll use RandomOverSampler.

To do so, complete the following steps:

Use the RandomOverSampler module from the imbalanced-learn library to resample the data. Be sure to confirm that the labels have an equal number of data points.

Use the LogisticRegression classifier and the resampled data to fit the model and make predictions.

Evaluate the model’s performance by doing the following:

Calculate the accuracy score of the model.

Generate a confusion matrix.

Print the classification report.



* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

First pass through I used the LogisticRegression only and the second pass through I used oversampling techniques along with LogisticRegression, which proved to be more accurate.  

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
                  pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384

precision is at 85%, where the recall is at 91% on just the LogisticRegression methodology.  


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384

Precision was lowered slightly to 84% but the recall was lifted to 99%.  To me, this methodology is better.  

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
Resampling and LogisticRegression together appear to perform best.  The classification shows that the precision dropped only 1% while the precision increased to 99%.  


* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Yes, we are trying to predict the 1's, the risky loans.  The methodology described above (the combined resampling and logistic regression) shows better overall result in finding the correct number of 1's, the primary goal of this endeavor.  

If you do not recommend any of the models, please justify your reasoning.
