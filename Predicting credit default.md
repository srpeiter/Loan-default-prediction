---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve as pr_curve
from sklearn.metrics import recall_score, precision_score, f1_score, auc, confusion_matrix, accuracy_score,  balanced_accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

```

```python
os.system("taskset -p 0xff %d" % os.getpid())
```

```python
font = {'weight' : 'normal',
        'size'   : 15}
mpl.rcParams['figure.figsize'] = (6., 6.0)
mpl.rc('font', **font)
mpl.rcParams['axes.unicode_minus']=False
mpl.rc('axes', linewidth = 1.5)
mpl.rc('xtick', labelsize=15)
mpl.rc('ytick', labelsize=15)
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.labelsize'] = 'large'
mpl.rcParams['xtick.major.size'] = 5.5     # major tick size in points
mpl.rcParams['xtick.major.width'] = 1.5     # major tick size in points
mpl.rcParams['ytick.major.size'] = 5.5     # major tick size in points
mpl.rcParams['ytick.major.width'] = 1.5     # major tick size in points
```

# Problem description


<font size="4">

Problem description
You are interested in investing in peer-to-peer loans on an online platform based in the USA. In order to calculate an expected return on your investment, you want to take into account the probability that a loan will default. Your goal in this assignment is to find the best possible model for predicting loan default, a binary outcome in the loan_status variable. This prediction should be made before a loan is issued. You should therefore, as far as is possible, make sure that you use only information available to investors before a loan is issued.

Make sure you define your goals for the model carefully beforehand, set expectations in terms of relevant metrics, and plan your model assessment steps. Consider various types of algorithms in your modelling effort, and take care to improve upon your initial efforts by learning more about the problem and data as you complete iterations of the modelling process.


```python
df = pd.read_csv('prediction_project/loans_train.csv')
print('training dataset shape: ',df.shape)
df.head()
```

```python
!head prediction_project/loans_test.csv
```

```python
df.columns
```

<font size="4">

    Now the question is what features to consider and what ML model to use?
    Lets explore the data first

```python
def search_missingval(df):
    '''
    Function that returns stats on missing value in datasets
    df: pandas Dataframe
    '''
    
    mis_val = df.isnull().sum()
    
    mis_val_table = pd.concat([mis_val,  100 * mis_val/len(df)], axis=1)
    print ('percentage of missing values in each column:')
    
    return mis_val_table
    
    
```

```python
search_missingval(df)
```

When it comes time to build our machine learning models, we will have to fill in these missing values (known as imputation). In later work, we will use models such as XGBoost that can handle missing values with no need for imputation. Another option would be to drop columns with a high percentage of missing values, although it is impossible to know ahead of time if these columns will be helpful to our model. Therefore, we will keep all of the columns for now.

```python
# Plotting numbers of credit default
order = ['Fully Paid', 'Charged Off']
cat_obj = df.loan_status.astype('category')
cat_obj.cat.reorder_categories(order, ordered=True,inplace=True)
cat_obj.cat.codes.hist()

```

Note that we have an imbalance here. People tend to pay back their debt more often compared to people defaulting. So this is something to account for in our model. Keep in mind, we have assigned defaults as 1 (positive) as we want to investigate whether our algorithm to correctly classify default instances despite this imbalance problem.
Because there are more instances of people paying back their debt and less cases where people default, the algorithm will do better around the first and less around the latter. One way to account for this problem is to penalize more for default instances.

It is therefore important to use the right metric for this purpose.


<font size="4">
Lets investigate the data type of the dataset


```python
df.dtypes.value_counts()
```

```python
# Number of unique classes in each object column
df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
```

There seems to be a thing about when to use label encoder or one hot encoder for encoding categorial classes.
One hot encoder increases the dimension of dataset if the number of classes is high, while label encoder is assigns an arbitrary number to a class. Label encoder is compatible with algorithm such as decision trees and random forest.
https://datascience.stackexchange.com/questions/9443/when-to-use-one-hot-encoding-vs-labelencoder-vs-dictvectorizor


```python
# Lets split the data into training set and results
df_train = df.drop('loan_status',axis = 1)
result = df['loan_status']
```

<font size="4">

Let's make some plots

```python
# Use describe to quickly scan the data
df_train.describe()
```

Let's look at the income distribution to see if we can detect outliers. I think annual income plays is an important 
indicator in determing whether someone will be able to pay off their debt.

```python
df1 = df[df.annual_inc !=0]
```

```python
df1.describe()
```

```python
sns.displot(
    df1, x="annual_inc", col='loan_status',
    facet_kws=dict(margin_titles=True),height=10,log_scale=True
)
plt.xlim(0.01e6,11e6)
# plt.ylim(0,20)

```

- It seems that income above 0.2e7 do not default on their debt. The question now is can we throw away this data?
Although, these datapoints occur naturally, they do not add new information to our model.
- Ok, lets not throw away anything, because these cannot be considered outliers or faulty values

```python
#just checking how many annual income is above 1 million. Common sense suggest that these incomes 
# dont risk to default on their debt
df[df.annual_inc > 0.1e7]
```

# Checking for feature correlations

```python
df
```

Converting categorial employment length to numerical values using label encoder. I used category from pandas

```python
def cat_to_num(df,columns_order):
    '''
    Function that converts categorial classes into 
    numerical values while preserving ordinality
    '''
    df2 = df.copy()
    
    for key, values in columns_order.items():
        _ = df[key].astype('category')
        __=_.cat.reorder_categories(values, ordered=True)
        df2[key] = __.cat.codes
        
        if np.sum(df2[key].unique() == -1) == 1:
            df2[key].replace({-1:np.nan},inplace=True)
    
    return df2

def OHE(df,columns):
    '''
    Function to convert categorial values into binary array
    df: pandas dataframe
    columns: features to turn into categorial values
    Returns updated array
    '''
    df2 = df.copy()
    
    df2.drop(columns,axis=1,inplace=True)
    
    arr = df.loc[:,columns]
    
    dummy_bin = pd.get_dummies(arr)
    
    return pd.concat([df2,dummy_bin],axis=1), list(dummy_bin.columns)
    
    
```

```python
order_emp = ['< 1 year','1 year',  '2 years', '3 years', '4 years','5 years','6 years', 
         '7 years', '8 years', '9 years', '10+ years' ]
order_stat = ['Fully Paid', 'Charged Off']

cat_columns = ['home_ownership','verification_status']

col_order = {'emp_length':order_emp,
            'loan_status':order_stat}

df2 = cat_to_num(df,col_order)

df2, cat_features = OHE(df2,cat_columns)
```

```python
df2.corr()['loan_status'].sort_values()
```

We see correlations with respect to the 'loan_status' variable. 'recoveries' has a high correlation, but this feature can not be determined beforehand, when the loan is provided. The same applies for 'total payment'. Lets check this

```python
# KDE plot of loans that were repaid on time
sns.kdeplot(df2.loc[df2['loan_status'] == 0, 'recoveries'], label = 'target == 0',color='r')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df2.loc[df2['loan_status'] == 1, 'recoveries'], label = 'target == 1')


```

The error says it all. Recoveries of zero means people pay off their debt, while a nonzero recovery means dafualt. This data cannot be known beforehand.

```python
df2.int_rate.hist()
```

```python
# Interest rate seems interesting 
# KDE plot of loans that were repaid on time

sns.kdeplot(df2.loc[df2['loan_status'] == 0, 'int_rate'], label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df2.loc[df2['loan_status'] == 1, 'int_rate'], label = 'target == 1')

# Labeling of plot
plt.xlabel('Interest rate (%)'); plt.ylabel('Density'); plt.title('Distribution of Interest rates');
plt.legend()
```

kde plot for default skews sligthtly towards higher interest rates, which makes sense.

```python
plt.figure(figsize = (17, 17))
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df2.corr(), dtype=bool))


sns.heatmap(df2.corr(), mask=mask, cmap = plt.cm.RdYlBu_r, vmin = -1, annot = True,fmt='.2f', vmax = 1)
plt.title('Correlation Heatmap');
```

```python
mp = df2.groupby('emp_length').mean()
mp
```

```python
plt.bar(mp.index, 100 - 100* mp['loan_status'])
plt.xticks(rotation = 75)
```

We see that the highest default rate is with people with 1 year employment and the lowest is with 10+years. But the  difference between the employement durations is within 3%. So for now, we can leave this feature out.

```python
emp_cut = df2[df2['loan_status']==1].groupby('emp_length').count()
```

```python
emp_cut
```

```python
tot_def = emp_cut.loan_status.sum()
tot_def
```

```python
plt.bar(mp.index, 100* emp_cut['loan_status']/tot_def)
plt.xticks(rotation = 75)
```

This is interesting, because from all the defaults, people who are employed the longest (10+ years) perform worse. But there is a trend here. In general, default is smaller with people who are employed longer, but in the category of 10+ years it explodes.

```python
def conv(x, N):
    '''
    Function to smooth out data
    '''
    x_conv = np.convolve(x,np.ones(N),'valid')
    return x_conv/N


```

```python
# plottin default risk with respect to loan amount
a = df2.groupby('loan_amnt').mean()
x = conv(a.index.values,80)
y=conv(1-a['loan_status'],80)
plt.plot(x,y,'*')
plt.xlabel('Loan amount')
plt.ylabel('default ratio (%)')

```

There seems to be no trend or pattern here. We do see higher default above 15000, but no specific pattern.
So we see a general trend here, when the loan amount increases, the default risk also increases. 

```python
# Interest rate seems interesting 
# KDE plot of loans that were repaid on time

sns.kdeplot(df2.loc[df2['loan_status'] == 0, 'dti'], label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(df2.loc[df2['loan_status'] == 1, 'dti'], label = 'target == 1')

# Labeling of plot
plt.xlabel('dti'); plt.ylabel('Density'); plt.title('Distribution of dti');
plt.xlim(0,100)
plt.legend()
```

Ok, there seems to be no difference in the KDE plot of the DTI ratio.


Let's choose the features:
- loan_amnt
- int_rate:
I noticed that that the grade class is related to the interest rate and loans with the same grade have the more or less the interest rate.
- home_ownership
- emp_length:
converted this array into numerical value using label encoding
- annual_inc
- verification_status
- dti
- delinq_2yrs
- open_acc
- pub_rec
- fico_range_low: this correlated with the interest rate
- installment
- zip_code (let's leave it out now, because of no ordinality). We can use one hot encoding but dimension of dataset will explode 

```python
df2.groupby('zip_code').mean()['loan_status'].plot()
```

```python
df2.groupby('addr_state').count()
```

```python
df2.groupby('zip_code').mean()
```

```python
df2.info()
```

# Logistic Regression 


Let's try a simple logistic regression scheme

```python
from sklearn.pipeline import FeatureUnion
```

```python
def select_features(df,feature):
    
    return df.loc[:,features]

def apply_transforms(df,oper):
    '''
    applies transforms to datasets with sklearn
    df: pandas dataframe
    oper: dictionary of operators to apply in order
    
    '''
    union = FeatureUnion(list(oper.items()))
    return union.fit_transform(df)
```

```python
# preparing the data
features = ['loan_amnt', 'int_rate', 'emp_length', 'annual_inc', 
            'dti', 'delinq_2yrs', 'open_acc', 'pub_rec','fico_range_low', 'installment','earliest_cr_line_year']

feature_ = features.copy()
[feature_.append(items) for items in cat_features]
# actually fico score and interest rate are related 
df_train = df2.loc[:,feature_]

target = df2.loan_status
```

```python
features
```

```python
df_train.info()
```

```python
# training set operations
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))
opers = {'imp': imputer,
        'minmaxsc': scaler }

# df_train_sc = apply_transforms(df_train,opers)
df_train_sc = imputer.fit_transform(df_train)
df_train_sc = scaler.fit_transform(df_train_sc)
```

```python
# test set operations
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))
df_test = pd.read_csv('prediction_project/loans_test.csv')
print('testing dataset shape: ',df_test.shape)
# df_test = df_test.loc[:,features]


order_emp = ['< 1 year','1 year',  '2 years', '3 years', '4 years','5 years','6 years', 
         '7 years', '8 years', '9 years', '10+ years' ]
order_stat = ['Fully Paid', 'Charged Off']

cat_columns = ['home_ownership','verification_status']

col_order = {'emp_length':order_emp}
df_test_num = cat_to_num(df_test,col_order)
df_test_num,_ = OHE(df_test_num,cat_columns)

df_test_num = df_test_num.loc[:,feature_]


# df_train_sc = apply_transforms(df_train,opers)
df_test_sc = imputer.fit_transform(df_test_num)
df_test_sc = scaler.fit_transform(df_test_sc)

df_test_num.head()
print(df_test_num.shape)
```

```python
df_label = pd.read_csv('prediction_project/loans_test_labels.csv')
# df_label.loc[:,('loan_status')] = df_label.loan_status.map(lambda x: 0 if x == 'Fully Paid' else 1)
order_stat = ['Fully Paid', 'Charged Off']
col_order = {'loan_status':order_stat}
df_label = cat_to_num(df_label,col_order)
labels = df_label['loan_status']
```

```python
labels[:11]
```

```python
log_reg = LogisticRegression(C=0.01)

# Train on the training data
log_reg.fit(df_train_sc, target)
```

```python
log_reg.coef_
```

```python
log_reg_pred = log_reg.predict_proba(df_test_sc)
log_reg_pred[:10]
```

Now what is the metric to use to measure the performance of our model?
We have to be carefull here, because our data is highly imbalanced with 14% instances of default.
Let's see what the ROC AUC does


https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18 <br>
Nice blog explaning how to handle and benchmark imbalanced datasets


![image.png](attachment:image.png)


Ref: https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/

```python
fpr, tpr, thresholds = roc_curve(labels,log_reg_pred[:,1])
```

```python
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
```

```python
print('AUC of ROC:', roc_auc_score(labels,log_reg_pred[:,1]))
```

```python
prre = pr_curve(labels.values,log_reg_pred[:,1])
```

```python
plt.plot(prre[2],prre[0][:-1],label='precision')
plt.plot(prre[2],prre[1][:-1],'k',label='Recall')
# plt.plot(pr,re)
plt.xlabel('Threshold')
# plt.ylabel('P')
plt.axhline(0.14,color='r')
plt.xlim(0,1)
plt.legend()
```

```python
print('AUC of precision-recall curve:',auc(prre[1],prre[0]))
```

```python
print('recall score:',recall_score(labels.values,log_reg.predict(df_test_sc)))
```

```python
print('precision score:',precision_score(labels.values,log_reg.predict(df_test_sc)))
```

```python
print('f1 score (harmonic mean between precision and recall):',
      f1_score(labels.values,log_reg.predict(df_test_sc)))
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,log_reg.predict(df_test_sc))

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

Looks very bad! Let's try to improve our model with a polynomial logistic regression with higher order terms


# Higher order logistic regression

```python
# Make a new dataframe for polynomial features
poly_features = imputer.fit_transform(df_train)
poly_features_test = imputer.fit_transform(df_test_num)


poly_target = target


# Need to impute missing values
# poly_features = imputer.fit_transform(poly_features)
# poly_features_test = imputer.transform(poly_features_test)

                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 2)#,include_bias=False)
```

```python
#  Train the polynomial features
poly_transformer.fit(poly_features)

# Transform the features
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape)

```

```python
print('Polynomial Features shape: ', poly_features_test.shape)
```

```python
# all features with interaction terms
poly_transformer.get_feature_names_out(feature_)
```

```python
# Create a dataframe of the features 
poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(feature_))

# # Add in the target
poly_features['target'] = target

# Find the correlations with the target
poly_corrs = poly_features.corr()['target'].sort_values()
```

```python
# Display most negative and most positive
print(poly_corrs.head(15))
print(poly_corrs.tail(10))
```

```python
poly_features.drop('target',axis=1,inplace=True)
poly_features_sc = scaler.fit_transform(poly_features)
poly_features_test_sc = scaler.fit_transform(poly_features_test)
# poly_features_test_sc = np.insert(poly_features_test_sc, 0, np.ones(len(poly_features_test_sc)),axis=1)
```

```python
# Now fit the data
log_reg = LogisticRegression(C = 0.01,max_iter=1000)#,tol=1e-6)

# Train on the training data
log_reg.fit(poly_features_sc, target)
```

```python
log_reg_pred = log_reg.predict_proba(poly_features_test_sc)
log_reg_pred[:10]
```

```python
poly_features_sc.shape
```

```python
fpr, tpr, thresholds = roc_curve(labels,log_reg_pred[:,1])
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
```

```python
print('AUC of ROC:', roc_auc_score(labels,log_reg_pred[:,1]))
```

The AUC of ROC increases with 0.001 but this is not a huge improvement. However the recall score is doubled. This is a better indicator for our imbalanced dataset. Adding higher order polynomial terms seems to improve the predicting capabilities of our classifier, but not a lot. Let's investigate how the randomforest classifier performs on our imbalanced dataset

```python
print('recall score:',recall_score(labels.values,log_reg.predict(poly_features_test_sc)))
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,log_reg.predict(poly_features_test_sc))

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
prre = pr_curve(labels.values,log_reg_pred[:,1])
```

```python
plt.plot(prre[2],prre[0][:-1],label='precision')
plt.plot(prre[2],prre[1][:-1],'k',label='Recall')
# plt.plot(pr,re)
plt.xlabel('Threshold')
# plt.ylabel('P')
plt.axhline(0.14,color='r')
plt.xlim(0,1)
plt.legend()
```

# Random Forest classifier


Lets see how the random forest clasifier performs with our datasets

```python
def plot_feature_importance(features,values):
    feature_fr = pd.DataFrame({'features': features,
                               'importance': values})
    
    feature_fr.sort_values(by='importance',inplace=True)
    
    plt.figure(figsize = (10, 6))
    ax = plt.subplot()
    
    # Need to reverse the index to plot most important on top
    ax.barh(feature_fr['features'], 
            feature_fr['importance']/feature_fr['importance'].sum(), 
            align = 'center', edgecolor = 'k')
    
    
    
    # Plot labeling
    plt.xlabel('Importance'); plt.title('Feature Importances')
    plt.show()
    

```

```python
rf = RandomForestClassifier(n_estimators = 10, max_depth = None, bootstrap = False, max_features=None,
                            class_weight= 'balanced', verbose = 1, n_jobs = -1, max_samples = None, 
                            oob_score=False)
```

```python
rf.fit(df_train_sc,target)
```

```python
plot_feature_importance(feature_,rf.feature_importances_)
```

```python
predictions = rf.predict_proba(df_test_sc)[:, 1]
predictions_cl = rf.predict(df_test_sc)
```

```python
predictions[:10]
```

```python
predictions_cl[:15]
```

```python
labels[:15]
```

```python
fpr, tpr, thresholds = roc_curve(labels,predictions)
plt.plot(fpr,tpr)
plt.plot(fpr,fpr,'--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
```

```python
print('AUC of ROC:', roc_auc_score(labels,predictions))
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,predictions_cl)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
print('recall score:',recall_score(labels.values,predictions_cl))
```

```python
print('f1 score (harmonic mean between precision and recall):',
      f1_score(labels.values,predictions_cl))
```

```python
prre = pr_curve(labels.values,predictions)
plt.plot(prre[2],prre[0][:-1],label='precision')
plt.plot(prre[2],prre[1][:-1],'k',label='Recall')
plt.plot(prre[1],prre[0],'m',label='Recall-PR')
# plt.plot(pr,re)
plt.xlabel('Threshold')
# plt.ylabel('P')
# plt.axhline(0.14,color='r')
plt.xlim(0,1)
plt.legend()
```

Ok. Now random forest improves the recall-precion score (f1) compared to logistic regression, but we still want to improve the predicting capabilities of our classifier. I have tried turning bootstrapping on/off, the amount of trees, max features etc. Randomforest class also has a class_weight attribute, which applies a weight to the class. That comes in handy for our imbalanced dataset. I found that the F1 score increases to 0.15, but the ROC-AUC decreases. Instead of hyperparameter tuning, lets turn again to feature engineering.


## Random Forest with poly features

```python
rf.fit(poly_features_sc,target)
```

```python
poly_pred = rf.predict(poly_features_test_sc)
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,poly_pred)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
balanced_accuracy_score(labels.values,predictions_cl)
```

Using the dataset with higher order terms do not improve the TP, but the FN for the given threshold.
Let's add new features to the new data set, such loan/unnual income and 



# Adding new features to dataset

```python
# Let's find new features which might be usefull
# This is dangerous! Do not assign a panda dataframe to another variable without using the copy method.
# If you omit this, changing the variable will also change the original dataframe. The variable is a pointer 
# to the original dataframe.
df_train_newfeat = df_train.copy()
df_train_newfeat['inc_cred_pct'] = df_train.annual_inc/df_train.loan_amnt
df_train_newfeat['inc_inst_pct'] = df_train.annual_inc/df_train.installment
```

```python
# aligning datasets such that test dataset also has the new features
# we just added in the training data
df_train_newfeat, df_test_newfeat = df_train_newfeat.align(df_test_num,axis=1)
```

```python
df_test_newfeat['inc_cred_pct'] = df_test.annual_inc/df_test.loan_amnt
df_test_newfeat['inc_inst_pct'] = df_test.annual_inc/df_test.installment
```

```python
# replacing missing values and scaling data with sklearn
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))

df_train_newfeat_sc = imputer.fit_transform(df_train_newfeat)
df_train_newfeat_sc = scaler.fit_transform(df_train_newfeat_sc)

df_test_newfeat_ = imputer.fit_transform(df_test_newfeat)
df_test_newfeat_sc = scaler.fit_transform(df_test_newfeat_)
```

```python
rf = RandomForestClassifier(n_estimators = 1, max_depth = None, bootstrap = True, max_features=None,
                            class_weight= 'balanced_subsample', verbose = 1, n_jobs = -1, max_samples = 14000, 
                            oob_score=False)
rf.fit(df_train_newfeat_sc ,target)
```

```python
new_pred = rf.predict_proba(df_test_newfeat_sc)[:, 1]
new_pred_cl = rf.predict(df_test_newfeat_sc)
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,new_pred_cl)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

Ok. We do see that the recall increases here (0.17). But something is weird here? The randomforest with a single estimator gives the highest recall, so basically a single decisiontree. Why is that? What I notice is that we get high recall values (0.22) whenever n_estimator * n_sample < total sample number. I think this has to do with the fact that our dataset is highly imbalanced. Whenever more that one estimator (tree), a majority vote is taken after a test instance is passed through the tree. Since we have a highly imbalanced set, it is highly probable that the majority of trees are going to prefer the majority class. This has to do maybe with the fact that features are chosen randomly at each node. Does the 'class_weight' attribute not account for this already? If not, is there a way to turn this random feature picking off?

```python
plot_feature_importance(df_test_newfeat.columns,rf.feature_importances_)
```

This is interesting: the new features we have created turn out to be important in our random forest classifier. How does it do that in detail? Using Gini impurity...wtf is that? But for now let's not waste a lot of time understanding it and save it for later
In this blog, they explain this: <br>
https://towardsdatascience.com/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3


```python
prre = pr_curve(labels.values,new_pred)
print('AUC of precision-recall curve:',auc(prre[1],prre[0]))
```

```python
print('AUC of ROC:', roc_auc_score(labels,new_pred))
```

```python
recall_score(labels,new_pred_cl)
```

```python
data_pairplot = df_train_newfeat.copy()
data_pairplot['target'] = target
g = sns.PairGrid(data_pairplot[:10000], hue="target",diag_sharey=False,corner=True,vars=features)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.scatterplot,alpha = 0.5)

```

Hmmm, I can't see significant correlations between features by eye, except (loan_amnt-int_rate, inc_inst_pct-inc_cred_pct and maybe installment-int_rate). But little distiction in classes. This is not ideal, since our dataset is highly imbalanced!


# Over/Undersampling to balance dataset


    Over and undersampling the minority and majority class respectively makes the dataset more balanced. Therefore, we dont need to provide a class_weight parameter to the classifier. The disadvantage of Random forest with regards to our dataset is that the trees are all independent. Since there is no clear difference in the distribution of the minority and majority classes, the decision tree will have a hard time in correctly predicting the classes. Also each tree has an equal weight in to final outcome, so the "incorrect" trees are equally weighted as the correct tree, which is not what we want.
   BTW, apparently scaling is important. When leaving this out, I notice that the classifier either predicts all positive or all negative (Not true, I forget to also unscale the validtion data)
When placing the scaler before under and oversampling, I noticed that it improves the recall score (by 1/13 *100%)!
```python
# Let's investigate what over and undersampling does to the distribution
data_for_samplng = df_train_newfeat.copy()

data_for_samplng = imputer.fit_transform(data_for_samplng)
over = SMOTE(sampling_strategy=0.25,k_neighbors=10)
adasyn = ADASYN(sampling_strategy=.25,n_neighbors=5)
under = RandomUnderSampler(sampling_strategy=.5)

sampled_data_over, y_reso = over.fit_resample(data_for_samplng,target)
sampled_data_under, y_resu = under.fit_resample(sampled_data_over,y_reso)
sampled_data = pd.DataFrame(sampled_data_under,columns=df_train_newfeat.columns)
sampled_data['target'] = y_resu
```

```python
plt.hist(y_resu)
```

```python
# Interest rate seems interesting 
# KDE plot of loans that were repaid on time

sns.histplot(sampled_data.loc[sampled_data['target'] == 0, 'inc_inst_pct'], label = 'target == 0',color='k')

# KDE plot of loans which were not repaid on time
sns.histplot(sampled_data.loc[sampled_data['target'] == 1, 'inc_inst_pct'], label = 'target == 1')

# Labeling of plot
plt.xlabel('Interest rate (%)'); plt.ylabel('Density'); plt.title('Distribution of Interest rates');
plt.legend()
plt.xlim(0,200)
```

Scaling works as expected.

```python

g = sns.PairGrid(sampled_data[:10000], hue="target",diag_sharey=False,corner=True,vars=features)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.scatterplot,alpha = 0.5)


```

```python
pd.DataFrame(sampled_data_under,columns=df_train_newfeat.columns)
```

```python
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))
over = SMOTE(sampling_strategy=0.25,k_neighbors=10)
under = RandomUnderSampler(sampling_strategy=.5)
model = RandomForestClassifier(n_estimators = 10, max_depth = None, bootstrap = False, max_features=None,
                            class_weight= 'balanced', verbose = 1, n_jobs = -1, max_samples = None, 
                            oob_score=False)
#  ('scaler',scaler),
steps = [('imp',imputer),('over',over), ('under',under),  ('model', model)]
pipeline = Pipeline(steps=steps)
pipeline.fit(df_train_newfeat,target)
```

```python
# Let's plot the confusion matrix
conf_matrix = confusion_matrix(labels.values,model.predict(df_test_newfeat_))

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
print('AUC of ROC:', roc_auc_score(labels,model.predict_proba(df_test_newfeat_)[:,1]))
```

```python
prre = pr_curve(labels.values,model.predict_proba(df_test_newfeat_)[:,1])
print('AUC of precision-recall curve:',auc(prre[1],prre[0]))
```

```python
print('recall score:',recall_score(labels.values,model.predict(df_test_newfeat_)))
```

```python
plt.plot(prre[2],prre[0][:-1],label='precision')
plt.plot(prre[2],prre[1][:-1],'k',label='Recall')
plt.plot(prre[1],prre[0],'m',label='Recall-PR')
# plt.plot(pr,re)
plt.xlabel('Threshold')
# plt.ylabel('P')
# plt.axhline(0.14,color='r')
plt.xlim(0,1)
plt.legend()
```

```python
plot_feature_importance(df_test_newfeat.columns,model.feature_importances_)
```

Over and undersampling helps to increase the recall (TP) at the expense of the precision (more False postive). False Positive are not good either, because it prevents people from obtaining loans, which is bad. Ideally we want to minimize false positive and maximize true positives.


# Let's try gradient boosting

```python
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))
over = SMOTE(sampling_strategy=.5,k_neighbors=10)
under = RandomUnderSampler(sampling_strategy=.5)
model_gbc = GradientBoostingClassifier(n_estimators = 200,max_depth=5,verbose=1)
#    ('under',under), ('scaler',scaler)
steps = [('imp',imputer),('over',over) ,('model', model_gbc)]
pipeline = Pipeline(steps=steps)
pipeline.fit(df_train_newfeat,target)
```

```python
# Let's plot the confusion matrix


conf_matrix = confusion_matrix(labels.values,model_gbc.predict(df_test_newfeat_))

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
prre = pr_curve(labels.values,model_gbc.predict_proba(df_test_newfeat_)[:,1])
print('AUC of precision-recall curve:',auc(prre[1],prre[0]))
```

```python
# # Lets do a parameter sweep
# parameters = {'model__n_estimators':[100,1000,3000], 'model__learning_rate':[0.001,0.1,10],
#              'model__max_depth':[1,5,9]}

# clf = GridSearchCV(pipeline, parameters)

# clf.fit(df_train_newfeat,target)
```

# XGB classifier


As a last resort, let's see how extreme gradient boosting performs!

```python
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
scaler = MinMaxScaler(feature_range = (0, 1))
over = SMOTE(sampling_strategy= .25,k_neighbors=10)
under = RandomUnderSampler(sampling_strategy=.8)
model_xgb = XGBClassifier(scale_pos_weight=1,subsample=0.6,eta=0.6, disable_default_eval_metric=1)
#    ('under',under), ('scaler',scaler), ('over',over) , ('under',under),
steps = [('imp',imputer), ('under',under), ('model', model_xgb)]
pipeline = Pipeline(steps=steps)
pipeline.fit(df_train_newfeat,target)

```

```python
# Let's plot the confusion matrix


conf_matrix = confusion_matrix(labels.values,model_xgb.predict(df_test_newfeat_))

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i,j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j,i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()
```

```python
prre = pr_curve(labels.values,model_xgb.predict_proba(df_test_newfeat_)[:,1])
print('AUC of precision-recall curve:',auc(prre[1],prre[0]))
```

```python
recall_score(labels.values,model_xgb.predict(df_test_newfeat_))
```

```python

```
