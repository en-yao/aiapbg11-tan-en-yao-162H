### Name and email
Name: Tan En Yao

Email: t.enyao93@gmail.com

### Overview of folder structure
```
AIAP
│
├── .github
├── src/                  (Source code directory)
│   ├── main.py           (Main Python script) 
│   ├── pipe.py           (Pipeline script)  
│   ├── config.py         (Configuration settings)
│   ├── query.py          (Dataset query)
├── README.md             (Project documentation)
├── eda.ipynb             (Jupyter notebook)
├── requirements.txt      (Python dependencies)
├── run.sh                (Shell script)
```

### Instruction for executing the pipeline and modifying parameters
To run the pipeline, double click on the the bash script (run.sh).

To experiment with different algorithms and parameters, 
1. Add the algorithm name and their parameters to `param_grid` in *src/config.py*
2. Import the model and initialize it in the `_initialize_classifiers()` method in in the `ClassifierPipeline` class found in *src/pipe.py*

To remove algorithms and parameters, comment out the algorithms and their parameters in *src/config.py*.

### Description of flow of the pipeline

1. Data cleaning
- *Converting column types:* Ensures that data types are correctly interpreted for downstream processing and analysis
- *Removing duplicates:* Eliminates redundant rows to prevent bias and overrepresentation of certain data points
- *Removing outliers:* Filters outliers to reduce skew in the data distribution 
- *Feature selection:* Chooses only relevant features to improve model efficiency

2. Create steps
- *Imputer:* Replaces missing values in categorical data with the most frequent value
- *Scaler:* Standardizes features to have a mean of 0 and a standard deviation of 1 for models that rely on distance metrics
- *Encoder:* Transforms nominal categorical features into binary representations through one-hot encoding

3. Set up the pipeline
- *Steps:* Combines imputer, scaler, encoder, and classifier to streamline the data transformation and model training workflow
- *Undersampler:* Mitigates class imbalance by reducing the number of majority class samples, enabling more balanced learning across classes
- *Classifier model:* Specifies the model type to be trained on

4. Create RandomSearchCV object
- *Pipeline:* Combines data transformation steps with model training within cross-validation
- *Param_grid:* Specifies the range of hyperparameters for optimizing model performance
- *Cross validation:* Evaluates the model on multiple test sets to provide a more reliable estimate of model performance on new, unseen data
- *Make predictions:* Predictions are generated on a test or hold-out set to evaluate model performance

5. Compute and print performance
- *Precision:* Measures the model's accuracy in correctly identifying positives
- *Recall:* Measures model's ability to capture all true positives
- *F1 score:* Provides a balanced score that accounts for both precision and recall
- *ROC AUC score:* Measures the model’s ability to distinguish between classes across different thresholds
- *Confusion matrix:* Summarizes classification results, showing true positives, true negatives, false positives, and false negatives

6. Visualise model performance
- *Box plots:* Show cross-validation score distributions
- *Precision-recall curve:* Illustrates the trade-off between precision and recall for various thresholds
- *ROC AUC curve:*  Visualizes the relationship between the true positive rate and false positive rate across thresholds

### Overview of key findings from the EDA 
- A quick view of the dataset revealed composite values in features Model and Factory
- The units for values in the Temperature feature had to be converted from °F to °C
- The amount of missing values is less than 5% of the total number of rows
- There were negative values in the RPM feature
- Class distribution in all the Failure types were imbalanced
- Target columns are not correlated
- The amount of duplicated rows is less than 1% of the total number of rows
- The numerical data all have a right skewed distribution
- There is one outlier detected for temperature where the value is considered unreasonable
- There is no correlation among the numerical features
- There are no clear class boundaries among the binary classes across all the Failure types
- A stacked bar plot of Failure counts against Year reveals no progression with time

### Describe how features in the dataset are processed

| Process                     | Description                                                                                       |
|---------------------------- |---------------------------------------------------------------------------------------------------|
| Split composite values      | Separate composite values into its components                                                     |
| Extract decimal from string | Extract decimal value from a string value                                                         |
| Convert negative values     | Converts negative values to positive                                                              |
| Impute Missing Values       | Replace missing values in features with mode                                                      |
| Remove duplicate rows       | Remove duplicated rows in the dataset                                                             |
| Convert measurement units   | Standardise units of measurement to just one unit                                                 |
| Remove outlier              | Remove data points that do not make sense based on domain knowledge                               |
| Log normalisation           | Transforms the data that is highly skewed by applying a logarithmic function                      |
| Standardisation             | Transforms the data to have a mean of 0 and a standard deviation of 1                             |
| One-hot encoding            | Converts categorical data into binary format                                                      |
| Removing redundant features | Eliminating features that do not provide new or useful information                                |

### Explanation of your choice of models 
Exploratory Data Analysis (EDA) has revealed that there are no distinct boundaries between the binary classes across all failure types, indicating that parametric classification methods would likely be ineffective. As a result, non-parametric methods like decision trees and random forests are more suitable for this classification task as they do not make assumptions on the distributions of the features. Furthermore, each chosen model supports class weighting, which is advantageous for handling imbalanced data. The only non-parametric method intentionally excluded is K-Nearest Neighbors because the classifier base its decisions on the majority vote of the nearest neighbors, which easily skews the predictions toward the majority class.

`RandomForestClassifier`
Random Forest handles imbalanced binary data by assigning a higher weight to the minority class, hence the model gives more focus to instances of this class. Additionally, each tree in the forest sees only a subset of the data, which balances the influence of both classes across the ensemble.

`SVC`
Support vector classifiers handle imbalanced binary data by penalising the misclassification of the minority class, hence the more frequent the misclassification, the higher the penalty assigned to the minority class. This penalty becomes similar to the class weights used in Random Forest to improve classification of the minority class.

`DecisionTreeClassifier`
Decision trees also handles imbalanced datasets by assigning weights to classes, making them useful in contexts where there are distinct majority and minority classes.

`GradientBoostingClassifier`
Gradient boosting handles imbalanced data well because it iteratively learns from errors and as a result puts more emphasis on misclassified instances. 

`XGBClassifier`
XGBoost is particularly suited for imbalanced binary classification because different weights can be assigned to the minority and majority classes. Similar to Gradient Boost, XGboost iteratively corrects errors, which helps improve recall on the minority class

### Evaluation of the models developed
In the context of fault detection in cars, recall will be preferred over precision because it's priority is to capture as many actual faults as possible, even at the cost of identifying non-faults as faults. It is so for the following reasons:

*Safety precaution:*
Missing an actual fault may lead to serious repercussions, potentially endangering passengers and other road users. A high recall ensures that most of the actual faults are detected, hence minimizing the risk of undiagnosed issues.

*Risk Mitigation:*
Failure to detect a fault might be more costly and dangerous than misidentifying a non-fault. While misidentifynig a non-fault is acceptable, it is critical not to miss a fault that could lead to a breakdown or accident.

Out of the 5 models evaluated, SVC has the highest overall recall score across Failures A to E
- Failure A: SVC has the highest median recall score of 0.8
- Failure B: SVC has the highest median recall score of 0.7
- Failure C: SVC has the highest median recall score of 0.65
- Failure D: SVC has the highest median recall score of 0.75
- Failure E: SVC has the highest median recall score of 0.73

However, the trade-off is exceptionally poor precision scores which is evident in the low Average Precision (AP) score show in the precision-recall curves. This means that the model has a high number of false positives

### Other considerations for deploying the models developed
