#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load dataset
import numpy as np
import pandas as pd
dataset = pd.read_csv('term-deposit-marketing-2020.csv')


# In[2]:


dataset


# In[3]:


#there is a class imbalance; 37104 nonsubscribes and 2896 subscribes
dataset[dataset['y']=='yes'].shape, dataset[dataset['y']=='no'].shape


# In[4]:


#take a look at basic statistics
dataset.describe()


# # Exploratory Data Analysis

# ## Look at counts for each feature

# In[5]:


import matplotlib.pyplot as plt
# Create a histogram for a specific column
#bins = int(2 * (len(dataset['age'])**(1/3))) #rice rule for number of bins for age
plt.hist(dataset['age'], bins=15)

# Labeling the axes and showing the plot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()
#here we see largest age range for customers is 30-40, and age range 40-60 is well-represented as well


# In[6]:


#bar chart for job
# Get the count of each category
category_counts = dataset['job'].value_counts()

# Create a bar chart for job
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Job')
plt.ylabel('Frequency')
plt.title('Bar Chart of Job')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[7]:


#Most of the jobs represented are blue-collar, management, technician, admin, and services
#while retired, self-employed, enterpreneur, unemployed, housemaid, student are least-represented.


# In[8]:


#bar chart for marital
# Get the count of each category
category_counts = dataset['marital'].value_counts()

# Create a bar chart for marital
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Marital status')
plt.ylabel('Frequency')
plt.title('Bar Chart of Marital')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[9]:


#most of the customers are married, about half the number of married are single, and half the number of single are divorced


# In[10]:


#bar chart for education
# Get the count of each category
category_counts = dataset['education'].value_counts()

# Create a bar chart for education
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Education')
plt.ylabel('Frequency')
plt.title('Bar Chart of Education')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[11]:


#most customers have secondary education (think high school), then tertiary (think college), then primary (think elementary school)


# In[12]:


#bar chart for default
# Get the count of each category
category_counts = dataset['default'].value_counts()

# Create a bar chart for default
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Default')
plt.ylabel('Frequency')
plt.title('Bar Chart of Default')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[13]:


#most do not have credit in default


# In[14]:


import matplotlib.pyplot as plt
# Create a histogram for average yearly balance
plt.hist(dataset['balance'], bins=20)

# Labeling the axes and showing the plot
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.title('Histogram of Balance')
plt.show()

counts, bin_edges, patches = plt.hist(dataset['balance'], bins=20)
counts, bin_edges


# In[15]:


#most of the balances are between -2500 and 3000 euros


# In[16]:


#bar chart for housing
# Get the count of each category
category_counts = dataset['housing'].value_counts()

# Create a bar chart for housing
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Housing')
plt.ylabel('Frequency')
plt.title('Bar Chart of Housing')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[17]:


#more customers have a housing loan vs not, but the numbers are not that different


# In[18]:


#bar chart for loan
# Get the count of each category
category_counts = dataset['loan'].value_counts()

# Create a bar chart for loan
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Loan')
plt.ylabel('Frequency')
plt.title('Bar Chart of Loan')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[19]:


#most do not have a personal loan


# In[20]:


#bar chart for contact
# Get the count of each category
category_counts = dataset['contact'].value_counts()

# Create a bar chart for contact
plt.bar(category_counts.index, category_counts.values)

# Labeling the axes and showing the plot
plt.xlabel('Contact')
plt.ylabel('Frequency')
plt.title('Bar Chart of Contact')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[21]:


#most customers have cellular or unknown for mode of contact


# In[22]:


#Plot positive and negative counts over time (in months)
monthly_counts = dataset.groupby(['month', 'y']).size().unstack(fill_value=0)

# Create a new figure
plt.figure(figsize=(10,6))

# Plot the counts of positive and negative target values
plt.plot(monthly_counts.index, monthly_counts['yes'], label='Positive Counts', marker='o')
plt.plot(monthly_counts.index, monthly_counts['no'], label='Negative Counts', marker='o')

# Add labels and legend
plt.title('Monthly Counts of Positive and Negative Target Values')
plt.xlabel('Month')
plt.ylabel('Count')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# In[23]:


#Plot positive and negative counts over time (in days)
daily_counts = dataset.groupby(['day', 'y']).size().unstack(fill_value=0)

# Create a new figure
plt.figure(figsize=(10,6))

# Plot the counts of positive and negative target values
plt.plot(daily_counts.index, daily_counts['yes'], label='Positive Counts', marker='o')
plt.plot(daily_counts.index, daily_counts['no'], label='Negative Counts', marker='o')

# Add labels and legend
plt.title('Daily Counts of Positive and Negative Target Values')
plt.xlabel('Day')
plt.ylabel('Count')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()


# In[24]:


import matplotlib.pyplot as plt
# Create a histogram for contact duration
plt.hist(dataset['duration'], bins=20)

# Labeling the axes and showing the plot
plt.xlabel('Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Duration')
plt.show()

counts, bin_edges, patches = plt.hist(dataset['duration'], bins=20)
counts, bin_edges


# In[25]:


#many of the call durations were less than 250 seconds with the second largest bin of call durations between 250 and 500 seconds


# In[26]:


# Determine the range for your data
data_min = int(np.floor(dataset['campaign'].min()))  # floor to the nearest lower integer
data_max = int(np.ceil(dataset['campaign'].max()))   # ceil to the nearest higher integer

# Create a sequence of bin edges from data_min to data_max
bin_edges = np.arange(data_min, data_max + 1, step=1)  # step=1 for integer steps

# Create the histogram using the specified bin edges
counts, bin_edges, patches = plt.hist(dataset['campaign'], bins=bin_edges)

# Labeling the axes and showing the plot
plt.xlabel('Number of Contacts')
plt.ylabel('Frequency')
plt.title('Histogram of Campaign')
plt.show()

# Output the counts and bin edges
print('Counts:', counts)
print('Bin edges:', bin_edges)


# ## Look for relationships between each feature and the target

# In[27]:


#compare age group with subscription rate
#first, group the ages into bins of length 10 years
dataset['age'] = pd.cut(dataset['age'], bins=range(0, 101, 10), right=False, labels=[f'{i}-{i+9}' for i in range(0, 100, 10)])

# Group by age_group and subscribed, then calculate the size of each group
age_counts = dataset.groupby(['age', 'y']).size().unstack(fill_value=0)

# Calculate subscription rate for each age group
age_counts['subscription_rate'] = age_counts['yes'] / (age_counts['no'] + age_counts['yes'])

# Sort age groups by subscription rate for better visualization
sorted_age_counts = age_counts.sort_values(by='subscription_rate', ascending=False)

# Print or plot the subscription rate
print(sorted_age_counts['subscription_rate'])

# Plot the subscription rates
sorted_age_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Age Group')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Age Group')
plt.show()


# In[28]:


#although people aged over 80 tend to have a higher subscription rate, their numbers are low;
#so, the more promising age groups are 60-69, 20-29, 30-39, 40-49, and 50-59
age_counts


# In[29]:


#compare job type with subscribe
job_counts = dataset.groupby(['job', 'y']).size().unstack(fill_value=0)
job_counts['subscription_rate'] = job_counts['yes'] / (job_counts['no'] + job_counts['yes'])

# Sort job types by subscription rate for better visualization
sorted_job_counts = job_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_job_counts['subscription_rate'])

# Or plot the subscription rates
sorted_job_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Job Type')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Job Type')
plt.show()


# In[30]:


#the job types with highest subscription rate are student, retired, unemployed, etc.


# In[31]:


#compare marital status with subscribe
marital_counts = dataset.groupby(['marital', 'y']).size().unstack(fill_value=0)
marital_counts['subscription_rate'] = marital_counts['yes'] / (marital_counts['no'] + marital_counts['yes'])

# Sort marital types by subscription rate for better visualization
sorted_marital_counts = marital_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_marital_counts['subscription_rate'])

# Or plot the subscription rates
sorted_marital_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Marital Status')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Marital Status')
plt.show()


# In[32]:


#the subscription rate doesn't seem to differ much between the marital statuses


# In[33]:


#compare education level with subscribe
education_counts = dataset.groupby(['education', 'y']).size().unstack(fill_value=0)
education_counts['subscription_rate'] = education_counts['yes'] / (education_counts['no'] + education_counts['yes'])

# Sort education levels by subscription rate for better visualization
sorted_education_counts = education_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_education_counts['subscription_rate'])

# Or plot the subscription rates
sorted_education_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Education Level')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Education Level')
plt.show()


# In[34]:


#the education level doesn't seem to matter much in regards to subscription rate


# In[35]:


#compare default with subscribe
default_counts = dataset.groupby(['default', 'y']).size().unstack(fill_value=0)
default_counts['subscription_rate'] = default_counts['yes'] / (default_counts['no'] + default_counts['yes'])

# Sort default types by subscription rate for better visualization
sorted_default_counts = default_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_default_counts['subscription_rate'])

# Or plot the subscription rates
sorted_default_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Default Status')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Default Status')
plt.show()


# In[36]:


#default status doesn't seem to matter


# In[37]:


#compare balance with subscription rate

# Binning the balance column in intervals of 5000
# Create bins
bins = range(-8000, 103000, 5000)

# Generate labels based on bins
labels = [f'{i}-{i+4999}' for i in bins[:-1]]  # Exclude the last bin edge

# Apply pd.cut()
dataset['balance'] = pd.cut(dataset['balance'], bins=bins, right=False, labels=labels)

# Group by balance group and subscribed, then calculate the size of each group
balance_counts = dataset.groupby(['balance', 'y']).size().unstack(fill_value=0)

# Calculate subscription rate for each balance group
balance_counts['subscription_rate'] = balance_counts['yes'] / (balance_counts['no'] + balance_counts['yes'])

# Sort balance groups by subscription rate for better visualization
sorted_balance_counts = balance_counts.sort_values(by='subscription_rate', ascending=False)

# Print or plot the subscription rate
print(sorted_balance_counts['subscription_rate'])

# Plot the subscription rates
sorted_balance_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Balance Group')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Balance Group')
plt.show()


# In[38]:


sorted_balance_counts


# In[39]:


#As you can see,not all balance ranges have large counts;
#We only want to consider ranges with enough counts
#As you can see, the balance groups with highest subscription rates are
#7000-12000, 2000-7000, and -3000-2000.

# Filter the DataFrame to only include rows where the sum of 'no' and 'yes' is greater than 1000
filtered_balance_counts = sorted_balance_counts[(sorted_balance_counts['no'] + sorted_balance_counts['yes']) > 500]

# Plot the filtered subscription rates
filtered_balance_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Balance Group')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Balance Group')
plt.show()


# In[40]:


#compare housing loan with subscribe
housing_counts = dataset.groupby(['housing', 'y']).size().unstack(fill_value=0)
housing_counts['subscription_rate'] = housing_counts['yes'] / (housing_counts['no'] + housing_counts['yes'])

# Sort housing types by subscription rate for better visualization
sorted_housing_counts = housing_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_housing_counts['subscription_rate'])

# Or plot the subscription rates
sorted_housing_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Housing Loan Status')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Housing Loan Status')
plt.show()


# In[41]:


sorted_housing_counts


# In[42]:


#The housing loan status doesn't seem to make a big difference


# In[43]:


#compare personal loan status with subscribe
loan_counts = dataset.groupby(['loan', 'y']).size().unstack(fill_value=0)
loan_counts['subscription_rate'] = loan_counts['yes'] / (loan_counts['no'] + loan_counts['yes'])

# Sort loan status types by subscription rate for better visualization
sorted_loan_counts = loan_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_loan_counts['subscription_rate'])

# Or plot the subscription rates
sorted_loan_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Personal Loan Status')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Personal Loan Status')
plt.show()


# In[44]:


#there isn't a huge difference between personal loan status types


# In[45]:


#compare contact type level with subscribe
contact_counts = dataset.groupby(['contact', 'y']).size().unstack(fill_value=0)
contact_counts['subscription_rate'] = contact_counts['yes'] / (contact_counts['no'] + contact_counts['yes'])

# Sort contact types by subscription rate for better visualization
sorted_contact_counts = contact_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_contact_counts['subscription_rate'])

# Or plot the subscription rates
sorted_contact_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Contact Type')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Contact Type')
plt.show()


# In[46]:


#cellular is the best contact type, with telephone being second best


# In[47]:


#compare last contact month with subscribe
month_counts = dataset.groupby(['month', 'y']).size().unstack(fill_value=0)
month_counts['subscription_rate'] = month_counts['yes'] / (month_counts['no'] + month_counts['yes'])

# Sort contact month by subscription rate for better visualization
sorted_month_counts = month_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_month_counts['subscription_rate'])

# Or plot the subscription rates
sorted_month_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Contact Month')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Contact Month')
plt.show()


# In[48]:


sorted_month_counts


# In[49]:


#the best months to contact are oct, mar, apr.


# In[50]:


#compare last contact day with subscribe
day_counts = dataset.groupby(['day', 'y']).size().unstack(fill_value=0)
day_counts['subscription_rate'] = day_counts['yes'] / (day_counts['no'] + day_counts['yes'])

# Sort contact day by subscription rate for better visualization
sorted_day_counts = day_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_day_counts['subscription_rate'])

# Or plot the subscription rates
sorted_day_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Contact Day')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Contact Day')
plt.show()


# In[51]:


sorted_day_counts


# In[52]:


#the best days to contact seem to the either the first of the month of the last of the month


# In[53]:


#compare contact duration to subscription rate

# Define bin edges using range, starting at 0 and ending at 5000 with a step of 250
bin_edges = range(0, 5000 + 1, 250)  # +1 to include the max value of 4918

# Create labels for these bins
labels = [f'{i}-{i+249}' for i in bin_edges[:-1]]  # Exclude the last bin edge

# Bin the data using pd.cut
dataset['duration'] = pd.cut(dataset['duration'], bins=bin_edges, labels=labels, right=False)

# Group by duration group and subscribed, then calculate the size of each group
duration_counts = dataset.groupby(['duration', 'y']).size().unstack(fill_value=0)

# Calculate subscription rate for each duration group
duration_counts['subscription_rate'] = duration_counts['yes'] / (duration_counts['no'] + duration_counts['yes'])

# Sort duration groups by subscription rate for better visualization
sorted_duration_counts = duration_counts.sort_values(by='subscription_rate', ascending=False)

# Print or plot the subscription rate
print(sorted_duration_counts['subscription_rate'])

# Plot the subscription rates
sorted_duration_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Duration Group')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Duration Group')
plt.show()


# In[54]:


#if we take a look at only the cases where there are at least 100 subscriptions,
#the best duration ranges between 0 to 1500 seconds with higher conversion rates 
#for 500-1250 seconds
sorted_duration_counts[sorted_duration_counts['yes']>100]


# In[55]:


import pandas as pd
import matplotlib.pyplot as plt



# Filter the DataFrame to only include rows where 'yes' is at least 100
filtered_duration_counts = sorted_duration_counts[sorted_duration_counts['yes'] >= 100]

# Plot the filtered subscription rates
filtered_duration_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Duration Group')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Duration Group')
plt.show()


# In[56]:


#compare campaign count with subscribe

# Define bin edges using range, starting at 0 and ending at 35 with a step of 3
bin_edges = range(0, 35 + 1, 3)

# Create labels for these bins
labels = [f'{i}-{i+2}' for i in bin_edges[:-1]]  # Exclude the last bin edge

# Bin the data using pd.cut
dataset['campaign'] = pd.cut(dataset['campaign'], bins=bin_edges, labels=labels, right=False)

campaign_counts = dataset.groupby(['campaign', 'y']).size().unstack(fill_value=0)
campaign_counts['subscription_rate'] = campaign_counts['yes'] / (campaign_counts['no'] + campaign_counts['yes'])

# Sort contact day by subscription rate for better visualization
sorted_campaign_counts = campaign_counts.sort_values(by='subscription_rate', ascending=False)

# Now you can print or plot the subscription rate
print(sorted_campaign_counts['subscription_rate'])

# Or plot the subscription rates
sorted_campaign_counts['subscription_rate'].plot(kind='bar')
plt.xlabel('Campaign Count')
plt.ylabel('Subscription Rate')
plt.title('Subscription Rate by Campaign Count')
plt.show()


# In[57]:


#The more contacts made, the less successful for subscription; so less contacts made is better


# # Machine Learning: Classification Models

# ## Data Preprocessing

# In[58]:


dataset = pd.read_csv('term-deposit-marketing-2020.csv')
#I have three binary features default, housing, and loan.  I will convert these into 0's and 1's
#I will also convert the 'y' column into 0's and 1's
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
dataset['default'] = le.fit_transform(dataset['default'])  # Assumes values are 'yes' and 'no'
dataset['housing'] = le.fit_transform(dataset['housing']) 
dataset['loan'] = le.fit_transform(dataset['loan']) 
dataset['y'] = le.fit_transform(dataset['y']) 

#there are several categorical features: job, marital, education, contact, month
#I will one-hot encode 'job', 'marital', 'education', 'contact'

dataset_encoded = pd.get_dummies(dataset, columns=['job'], prefix='job')
dataset_encoded = pd.get_dummies(dataset_encoded, columns=['marital'], prefix='marital')
dataset_encoded = pd.get_dummies(dataset_encoded, columns=['education'], prefix='education')
dataset_encoded = pd.get_dummies(dataset_encoded, columns=['contact'], prefix='contact')
#dataset_encoded = pd.get_dummies(dataset_encoded, columns=['month'], prefix='month')

#for month, just convert to numbers between 1 and 12
month_dict = {
    'jan': 1, 'feb': 2, 'mar': 3,
    'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9,
    'oct': 10, 'nov': 11, 'dec': 12
}
dataset_encoded['month'] = dataset['month'].str.lower().replace(month_dict)


# In[59]:


dataset_encoded


# ## Build, Apply, and Assess Models

# In[60]:


#Perform logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import cross_val_score

#Define X and Y using dataset
X = dataset_encoded.drop(columns=['y']).values
Y = dataset_encoded['y'].values

#Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 2)

#Feature Scaling
column_names=dataset_encoded.drop(columns=['y']).columns
X_train = pd.DataFrame(X_train, columns=column_names)
X_test = pd.DataFrame(X_test, columns=column_names)
from sklearn.preprocessing import StandardScaler

# Define columns to scale
columns_to_scale = ['age', 'balance', 'duration']

# Create the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both training and test data
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

X_train=X_train.values
X_test=X_test.values


#Fitting Logistic Regression to the Training set
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train,Y_train)

# #Predicting the Test set results
# Y_pred = classifier.predict(X_test)

# #Making the confusion matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, Y_pred)
# print(f'accuracy:{cm.trace()/len(Y_test)}')
# print(f'confusion matrix:\n{cm}')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(classifier, X_train, Y_train, cv=5)

# The cv_scores array will contain the accuracy for each fold
mean_cv_score = cv_scores.mean()

# Output the mean accuracy across the 5 folds
print(f'Mean CV Accuracy: {mean_cv_score * 100:.2f}%')

#Predicting the Test set results
Y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
print(f'accuracy:{cm.trace()/len(Y_test)}')
print(f'confusion matrix:\n{cm}')

#check the precision, recall, and f1-score
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))

#Although the accuracy is very high,
#the precision, recall, and f1-score are high for class 0 but low for class 1.  This is not good.  We want high scores for both.
# Recall for minority class is how many of the subscribers the model correctly identifies out of all the subscribers.
#Precision for minority, on the other hand, is how many of the model's predictions of subscriber are correct.
#We want to maximize both precision and recall since we don't want to count someone as a subscriber when they're not;
#this could be a waste of effort to try and convert them
#Secondly, we don't want to skip someone if they are a subscriber since those are the people we want to target;
#if customers who are potential subscribers are rare or difficult to find,
#it may suggest that recall is more important than precision


# In[61]:


import shap

# Convert to DataFrames with column names
X_train = pd.DataFrame(X_train, columns=dataset_encoded.drop(columns=['y']).columns)
X_test = pd.DataFrame(X_test, columns=dataset_encoded.drop(columns=['y']).columns)
# create the explainer
explainer = shap.LinearExplainer(classifier, X_train)

# compute SHAP values
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[62]:


#To overcome the imbalanced data issue, we can apply oversampling and undersampling
get_ipython().system('pip install -U imbalanced-learn')


# In[63]:


import imblearn
# Oversampling and under sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter


# In[64]:


#Oversampling
#convert X_train, X_test back to np arrays
X_train=X_train.values
X_test=X_test.values
# Randomly over-sample the minority class
ros = RandomOverSampler(random_state=42, sampling_strategy='minority')
X_train_ros, Y_train_ros= ros.fit_resample(X_train, Y_train)
# Check the number of records after over sampling
print(sorted(Counter(Y_train_ros).items()))


# In[65]:


#There are 29648 nonsubscribers in Y_train; after oversampling, we now have 29648 in each of the
# two classes--subscriber and nonsubscriber.


# In[66]:


#Fitting Logistic Regression to the randomly over-sampled Training set
classifier_ros = LogisticRegression(max_iter=1000)
classifier_ros.fit(X_train_ros,Y_train_ros)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(classifier_ros, X_train_ros, Y_train_ros, cv=5)

# The cv_scores array will contain the accuracy for each fold
mean_cv_score = cv_scores.mean()

# Output the mean accuracy across the 5 folds
print(f'Mean CV Accuracy: {mean_cv_score * 100:.2f}%')

#Predicting the Test set results
Y_pred_ros = classifier_ros.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_ros)
print(f'accuracy:{cm.trace()/len(Y_test)}')
print(f'confusion matrix:\n{cm}')

#check the precision, recall, and f1-score
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_ros))


# In[67]:


#the recall greatly improved but precision went down; the accuracy is about 87%


# In[68]:


import shap
# Convert to DataFrames with column names
X_train_ros = pd.DataFrame(X_train_ros, columns=dataset_encoded.drop(columns=['y']).columns)
X_test = pd.DataFrame(X_test, columns=dataset_encoded.drop(columns=['y']).columns)
# create the explainer
explainer = shap.LinearExplainer(classifier_ros, X_train_ros)

# compute SHAP values
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[69]:


#Perform Random Forest with oversampling
#convert X_train, X_test back to np arrays
X_train_ros=X_train_ros.values
X_test=X_test.values
#Fitting random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)
classifier.fit(X_train_ros,Y_train_ros)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(classifier, X_train_ros, Y_train_ros, cv=5)

# The cv_scores array will contain the accuracy for each fold
mean_cv_score = cv_scores.mean()

# Output the mean accuracy across the 5 folds
print(f'Mean CV Accuracy: {mean_cv_score * 100:.2f}%')

#Predicting the Test set results
Y_pred_rf = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rf)
print(f'accuracy:{cm.trace()/len(Y_test)}')
print(f'confusion matrix:\n{cm}')

#check the precision, recall, and f1-score
from sklearn.metrics import classification_report
print(f'classification report:\n{classification_report(Y_test,Y_pred_rf)}')


# In[70]:


#precision improved, but recall went down


# In[71]:


import shap

# Convert to DataFrames with column names
X_train_ros = pd.DataFrame(X_train_ros, columns=dataset_encoded.drop(columns=['y']).columns)
X_test = pd.DataFrame(X_test, columns=dataset_encoded.drop(columns=['y']).columns)
# create the explainer
explainer = shap.TreeExplainer(classifier, X_train_ros, n_jobs=-1)

# compute SHAP values

subset = X_test.sample(frac=0.1, random_state=1)  # Adjust frac to control the size of the subset
shap_values = explainer.shap_values(subset,check_additivity=False)
shap.summary_plot(shap_values, subset, plot_type='bar')


# In[72]:


#Undersampling approach

# Randomly under sample the majority class
rus = RandomUnderSampler(random_state=42)
X_train_rus, Y_train_rus= rus.fit_resample(X_train, Y_train)
# Check the number of records after under sampling
print(sorted(Counter(Y_train_rus).items()))


# In[73]:


#After undersampling, we now have 2352 of each class--subscriber and nonsubscriber


# In[74]:


#Fitting Logistic Regression to the randomly under-sampled Training set
#convert X_train, X_test back to np arrays
X_test=X_test.values

classifier_rus = LogisticRegression(max_iter=1000)
classifier_rus.fit(X_train_rus,Y_train_rus)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(classifier_rus, X_train_rus, Y_train_rus, cv=5)

# The cv_scores array will contain the accuracy for each fold
mean_cv_score = cv_scores.mean()

# Output the mean accuracy across the 5 folds
print(f'Mean CV Accuracy: {mean_cv_score * 100:.2f}%')

#Predicting the Test set results
Y_pred_rus = classifier_rus.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rus)
print(f'accuracy:{cm.trace()/len(Y_test)}')
print(f'confusion matrix:\n{cm}')

#check the precision, recall, and f1-score
from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred_rus))


# In[75]:


import shap
# Convert to DataFrames with column names
X_train_rus = pd.DataFrame(X_train_rus, columns=dataset_encoded.drop(columns=['y']).columns)
X_test = pd.DataFrame(X_test, columns=dataset_encoded.drop(columns=['y']).columns)
# create the explainer
explainer = shap.LinearExplainer(classifier_rus, X_train_rus)

# compute SHAP values
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')


# In[76]:


#Perform Random Forest with undersampling
#convert X_train, X_test back to np arrays
X_train_rus=X_train_rus.values
X_test=X_test.values
#Fitting random forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy', n_estimators=100, random_state=0)
classifier.fit(X_train_rus,Y_train_rus)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(classifier, X_train_rus, Y_train_rus, cv=5)

# The cv_scores array will contain the accuracy for each fold
mean_cv_score = cv_scores.mean()

# Output the mean accuracy across the 5 folds
print(f'Mean CV Accuracy: {mean_cv_score * 100:.2f}%')

#Predicting the Test set results
Y_pred_rf = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred_rf)
print(f'accuracy:{cm.trace()/len(Y_test)}')
print(f'confusion matrix:\n{cm}')

#check the precision, recall, and f1-score
from sklearn.metrics import classification_report
print(f'classification report:\n{classification_report(Y_test,Y_pred_rf)}')


# In[77]:


#Both precision and recall improved


# In[78]:


import shap

# Convert to DataFrames with column names
X_train_rus = pd.DataFrame(X_train_rus, columns=dataset_encoded.drop(columns=['y']).columns)
X_test = pd.DataFrame(X_test, columns=dataset_encoded.drop(columns=['y']).columns)
# create the explainer
explainer = shap.TreeExplainer(classifier, X_train_rus, n_jobs=-1)

# compute SHAP values

subset = X_test.sample(frac=0.1, random_state=1)  # Adjust frac to control the size of the subset
shap_values = explainer.shap_values(subset, check_additivity=False)
shap.summary_plot(shap_values, subset, plot_type='bar')


# In[79]:


# get importance
importances = classifier.feature_importances_
from matplotlib import pyplot as plt
plt.title('Feature Importances')
plt.xlabel('Relative Importance')

feat_importances = pd.Series(classifier.feature_importances_, index=dataset_encoded.iloc[:,0:-1].columns)
feat_importances.nlargest(10).plot(kind='barh')


# In[80]:


#According to the above, the top four features are duration, balance, day, and age.


# In[ ]:





# # Perform Statistical Tests for Feature Significance

# In[81]:


dataset


# In[82]:


#use chi-square test for the categorical variables job, marital, education, default, housing, loan, contact, month
import pandas as pd
from scipy.stats import chi2_contingency

# Assume dataset is your DataFrame
dataset = dataset  # Replace with your DataFrame
target_variable = 'y'  # Replace with your target column name

# List of categorical features
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month']  # Replace with your list of features

# Dictionary to store results
chi2_results = {}

# Loop through each categorical feature
for feature in categorical_features:
    # Create a contingency table
    contingency_table = pd.crosstab(dataset[feature], dataset[target_variable])
    
    # Perform the Chi-Square test
    chi2_stat, p_value, dof, ex = chi2_contingency(contingency_table)
    
    # Store the results in the dictionary
    chi2_results[feature] = {
        'Chi2 Statistic': chi2_stat,
        'P-value': p_value
    }

# Now chi2_results contains the Chi2 Statistic and P-value for each feature
# Optionally, print the results
for feature, result in chi2_results.items():
    print(f"{feature}: Chi2 Statistic = {result['Chi2 Statistic']}, P-value = {result['P-value']}")


# In[83]:


#all the categorical features are significant except for default


# In[84]:


#perform t-test for all numeric features
import pandas as pd
from scipy import stats

# Assume dataset is your DataFrame
dataset = dataset  # Replace with your DataFrame
target_variable = 'y'  # Replace with your target column name

# List of numerical features
numerical_features = ['age', 'balance', 'day', 'duration', 'campaign']  # Replace with your list of features

# Dictionary to store results
t_test_results = {}

# Segregate data based on the binary target variable
group1 = dataset[dataset[target_variable] == 0]
group2 = dataset[dataset[target_variable] == 1]

# Loop through each numerical feature
for feature in numerical_features:
    # Perform the t-test
    t_statistic, p_value = stats.ttest_ind(group1[feature], group2[feature], equal_var=False)  # Assuming unequal variances
    
    # Store the results in the dictionary
    t_test_results[feature] = {
        'T-Statistic': t_statistic,
        'P-value': p_value
    }

# Now t_test_results contains the T-Statistic and P-value for each feature
# Optionally, print the results
for feature, result in t_test_results.items():
    print(f"{feature}: T-Statistic = {result['T-Statistic']}, P-value = {result['P-value']}")


# In[85]:


#the numeric features age, balance, duration, and campaign are significant


# In[ ]:




