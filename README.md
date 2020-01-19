# __DSCI_522_group-307__

## __Project Proposal__  

---

### __1. Data Set__  

The __Adult Data Set__ obtained from the University of California Irvine Machine Learning Repository [1] was selected to be used in this UBC MDS DSCI 522 data analysis project.  

The [download_data](https://github.com/UBC-MDS/DSCI_522_group-307/blob/master/scripts/download_data.py) script can be used to download and save the data into training, validation, and testing sets. The downloaded data files can be found in the __[\data](https://github.com/UBC-MDS/DSCI_522_group-307/tree/master/data)__ folder:  
- [training set](https://github.com/UBC-MDS/DSCI_522_group-307/blob/master/data/train.csv)  
- [validation set](https://github.com/UBC-MDS/DSCI_522_group-307/blob/master/data/validation.csv)  
- [testing set](https://github.com/UBC-MDS/DSCI_522_group-307/blob/master/data/test.csv)  

### __2. Research Question__   

__Predictive Research Question__  
- What features most strongly predict whether or not income will exceed $50k/year?  

_Additional subquestions may be added as project progresses_

### __3. Data Analysis__  

The data analysis will be developed in a systematic and reproducable manner:  
1) Import and split the data into training, validation, and testing sets  
2) Exploratory data analysis on the training set (generation of summary tables and plots)  
3) Preprocessing including addressing missing values and transforming continuous and categorical variables  
4) Classifier selection  
    - several classifiers will be tested in determining the final model   
    - classifiers to try include logistic classifier, random forest, kNN, SVM in addition to bagging and boosting methods
5) Hyperparameter optimization  
6) Model testing on testing data set

Once an optimized model is developed, scoring and feature weights will be summarized and reported. 

_Note: all scripts will be developed in Python and all machine learning analysis will be generated using the scikit-learn library_


### __4. EDA Tables & Figures__  

Tables will be generated summarizing the feature properties; this will aid in determining the type of imputer and transformation required during preprocessing.  
Histograms will also be generated to aid in evaluating the numerical features and the selection of the classifiers to test.

### __5. Summary__  

A final report will be generated after the optimized model has been created. This report will include an answer to the research question, justifications of classifiers and hyperparameter values, any assumptions made throughout model development, limitations of the model, and improvement recommendations. Summary tables and plots will be generated to effectively communicate model characteristics and effectiveness. 

[1] C. Blake and C. Merz. UCI repository of machine learningdatabases, 1998