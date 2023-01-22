#!/usr/bin/env python
# coding: utf-8

# # Data Science Intern At LetsGrowMore

# ## LGMVIP - JAN 2023

# ## TASK 1 - MUSIC RECOMMENDATION SYSTEM

# Music recommender systems can suggest songs to users based on their listening patterns. 
# 
# Dataset Link : https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data

# ### Author - Diwakar Chaurasia

# ## Preparation

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing Dateaset

# In[6]:


ntr = 7000
nts = 3000
data_path = r"C:\Users\DIWAKAR DC\Desktop\Python\Task 4 LGM\train.csv"
train = pd.read_csv(data_path, nrows=ntr) # The "nrows" parameter is set to "ntr" which is 7000, so it will only read the first 7000 rows of the CSV file.
It then assigns the column names to a list called "names".
names = ['msno', 'song_id', 'source_system_tab', 'source_screen_name', 'source_type', 'target']


# In[11]:


test_1 = pd.read_csv(data_path, names = names, skiprows=ntr, nrows=nts) #The "skiprows" parameter set to "ntr" (7000) and "nrows" parameter set to "nts" (3000),
                                                                          #so it will only read the next 3000 rows of the CSV file after skipping the first 7000 rows.
songs = pd.read_csv(r"C:\Users\DIWAKAR DC\Desktop\Python\Task 4 LGM\songs.csv")
members = pd.read_csv(r"C:\Users\DIWAKAR DC\Desktop\Python\Task 4 LGM\members.csv")


# ### Train dataset

# In[12]:


train.head(10) #It will display the first 10 rows of the DataFrame "train"


# In[13]:


train.sample(10) #It will display 10 random rows of the DataFrame "train"


# In[15]:


train.shape #The tuple contains the number of rows and columns in the DataFrame.


# In[17]:


train.info() #It displays a summary of the DataFrame "train" on the output, including the column names, data types, and number of non-null values for each column, and the memory usage of the DataFrame.


# In[18]:


train.columns #It displays the name of all columns 


# In[19]:


train.describe() #It will display the descriptive statistics of the DataFrame "train"


# In[20]:


train.isnull().sum() #It will display the number of missing values in each column of the DataFrame "train" in the output.


# ### Songs Dataset

# In[22]:


songs.head(10) #It will display the first 10 rows of the DataFrame "songs"


# In[23]:


songs.sample(10) #It will display 10 random rows of the DataFrame "songs"


# In[24]:


songs.shape #The tuple contains the number of rows and columns in the DataFrame.


# In[25]:


songs.info() #It displays a summary of the DataFrame "songs" on the output, including the column names, data types, and number of non-null values for each column, and the memory usage of the DataFrame.


# In[29]:


songs.columns #It displays the name of all columns


# In[30]:


songs.describe() #It will display the descriptive statistics of the DataFrame "songs"


# In[31]:


songs.isnull().sum() #It will display the number of missing values in each column of the DataFrame "songs" in the output.


# ### Members Dataset

# In[32]:


members.head(10) #It will display the first 10 rows of the DataFrame "members"


# In[33]:


members.sample(10) #It will display 10 random rows of the DataFrame "members"


# In[34]:


members.shape #The tuple contains the number of rows and columns in the DataFrame.


# In[35]:


members.info() # It displays a summary of the DataFrame "members" on the output, including the column names, data types, and number of non-null values for each column, and the memory usage of the DataFrame.


# In[37]:


members.columns #It displays the name of all columns


# In[38]:


members.describe() #It will display the descriptive statistics of the DataFrame "members"


# In[39]:


members.isnull().sum() #It will display the number of missing values in each column of the DataFrame "members" in the output.


# ## Performing Data Visualization

# In[59]:


#DATA VISUALIZATION
plt.figure(figsize =(14,8))
red_palette = sns.color_palette("Reds", 4)
sns.countplot(x = train['source_system_tab'], hue=train['source_system_tab'], palette=red_palette)


# In[65]:


plt.figure(figsize =(14,8))
green_palette = sns.color_palette("Greens", 2)
sns.countplot(x = train['source_system_tab'], hue=train['target'], palette=green_palette)


# In[64]:


plt.figure(figsize =(14,8))
blue_palette = sns.color_palette("Blues", 2)
sns.countplot(x = train['source_screen_name'], hue=train['target'], palette=blue_palette, data = train, orient = 'v')
plt.xticks(rotation = 90)
plt.show()


# In[87]:


plt.figure(figsize=(14,8))
sns.set(font_scale=2, style='ticks')
sns.set_palette('hot')
sns.countplot(x='source_type', hue='source_type', data = train, edgecolor = sns.color_palette('hot',5))
plt.grid(visible=True, linestyle='-.')
plt.ylabel('Count', fontsize = 20)
plt.xticks(rotation = '45')
plt.title('Count plot types for listening music', fontsize = 25)
plt.legend().set_visible(False)
plt.tight_layout()


# 1. Source_system_tab indicates the name of the tab where the event was triggered. System tabs are used to categorize KKBOX mobile apps functions.
# 2. It can be depicted from the above plot that people repeat songs from their library or discover tabs.
# 3. From notifications or settings tab people are not interested to repeat songs.

# In[91]:


plt.figure(figsize=(14,8))
sns.set(font_scale=2)
sns.countplot(y = 'source_screen_name', data = train, facecolor = (0,0,0,0), linewidth = 2, edgecolor = sns.color_palette('hot',4))
sns.set(style = "darkgrid")
plt.xlabel ('Source types', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
plt.xticks(rotation = '45')
plt.title('Count plot for screen using', fontsize = 25)
plt.tight_layout()


# In[100]:


def count_plot(data, x, hue, type):
  plt.figure(figsize = (15, 8))
  sns.set(font_scale = 2)
  sns.countplot(x = x, hue = hue, data = data)
  plt.xlabel(x, fontsize = 30)
  plt.ylabel('Count', fontsize = 30)
  plt.xticks(rotation = '90')
  plt.title('Count plot for {0} in {1} data'.format(x, type), fontsize = 30)
  plt.tight_layout()


# • It will take the dataframe, column name of x-axis, hue, and type of data as input and will create a countplot of the data on x-axis and hue of the data.
# 
# • It will also set the labels, title, and adjust the subplot parameters for better visualization.
# 
# • It will display the plot on the console/output.

# In[109]:


def count_plot_function(data, x):
  plt.figure(figsize = (15,8))
  sns.set(font_scale = 2)
  sns.countplot(x = x, data = data)
  plt.xlabel(x, fontsize = 30)
  plt.ylabel('Count', fontsize = 30)
  plt.xticks(rotation = '90')
  plt.title('Count plot', fontsize = 30)
  plt.tight_layout()


# • The Count_plot function will creates a figure with a specified size, sets the font scale, creates a countplot using the seaborn library, sets the x-label, y-label, rotation of x-axis ticks and title of the plot, and adjusts the subplot parameters.

# In[128]:


plt.figure(figsize =(14,8))
sns.countplot(x = train['source_type'],palette=sns.color_palette("husl",2),hue=train['target'],data = train,orient='v')
plt.xticks(rotation =90)
plt.show()


# • The countplot will show the count of different 'source_type' and it will color the bars of the 'source_type' according to the values in the 'target' column.

# ## Performing Data Visualization on members.csv

# In[123]:


import matplotlib as mpl

mpl.rcParams['font.size'] = 40.0
labels = ['Male','Female']
colors = ['#66b3ff','#ff9999']
plt.figure(figsize = (9, 9))
sizes = pd.value_counts(members.gender)
patches, texts, autotexts = plt.pie(sizes,labels=labels,colors=colors, autopct='%.0f%%',
                                    shadow=False, radius=1,startangle=90)
for t in texts:
    t.set_size('smaller')
plt.legend()
plt.show()


# The Pie chart displays the proportion of Male and Female in the members dataframe. And the highest as we can see clearly is of male.

# In[129]:


plt.figure(figsize =(14,8))
orange_palette = sns.color_palette("Oranges", 2)
sns.countplot(x = songs['language'],data =train,hue=songs['language'],palette = orange_palette,orient='h')


# In[141]:


plt.figure(figsize =(14,8))
Purples_palette = sns.color_palette("Purples", 5)
sns.countplot(x = 'registered_via', data = members, palette = Purples_palette)
plt.title("Count Plot")


# • Most of the registrations happened via method '4', '7' and '9'.
# 
# • A few users have registered themselves via '13' and '16' methods.

# In[144]:


plt.figure(figsize =(14,8))
rainbow_palette = sns.color_palette("rainbow", 5)
sns.countplot(x = 'city', data = members, palette = rainbow_palette)
plt.title("Count Plot")


# • Most of the people who used to listen songs are from city labelled '1'. Other cities have very few people who prefer listening music via this music app.

# ## Performing stats test on members.csv

# In[150]:


plt.figure(figsize = (14, 8)) 
sns.histplot(members.registration_init_time, color = 'orange', kde=True)
sns.set(font_scale=2)
plt.xlabel('registration time', fontsize=25)
plt.title('Probability density function for Registration', fontsize=25)


# This code will create a PDF plot of the registration time of the members dataframe, it shows the probability density of registration_time in the dataset. 
# The x-axis shows the registration time, and the y-axis shows the density.

# In[160]:


def plot_pdf_cdf(x, flag):
    plt.figure(figsize=(14, 8))
    if flag:
        sns.histplot(x, cumulative=True, element='bars', color='purple')
        plt.title('CDF for age')
    else:
        sns.histplot(x, element='bars', color='purple')
        plt.title('PDF for age')
    sns.set(font_scale=2)


# In[162]:


plot_pdf_cdf(members['bd'], False)


# In[163]:


plot_pdf_cdf(members['bd'], True)
plt.show()


#  'CDF' stands for Cumulative Distribution Function, it gives the cumulative probability of a random variable taking on a value less than or equal to x.

# In[164]:


np.percentile(members['bd'].values, 98)


# 1. 98th percentile user is of 47 age. The 98th percentile can be defined as the value below which 98% of the ages in the dataset fall.
# 2. Means most of the user are below 50.
# 3. We can also observe via above CDF that almost 99% values are below 50. There are also some outliers like 1030, -38, -43, 1051, etc. As age cannot be negative value or more than 100 for humans.

# ## Now Performing Data Preprocessing & Cleaning
# 

# In[167]:


test = test_1.drop(['target'],axis=1)
ytr = np.array(test_1['target'])


# In[169]:


test_name = ['id','msno','song_id','source_system_tab',
             'source_screen_name','source_type']
test['id']=np.arange(nts)
test = test[test_name]


# In[170]:


song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')


# The above code is merging the train and test data with the songs data on the 'song_id' column. It is using the 'left' join type, which means that all rows from the train and test data will be kept and any missing values for the 'song_id' column in the songs data will be filled with NaN.

# In[171]:


members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))


# In the above provided code the lambda function is used to extract the year, month and date from the registration_init_time column in a concise and readable way.

# In[172]:


members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)


# In the above provided code the lambda function is used to extract the year, month and date from the expiration_date column in a concise and readable way. And,finally it drops the 'registration_init_time' column from the members dataframe using the .drop() method, passing the column name as the first argument and specifying axis=1 to indicate that we want to drop a column and not a row.
# 
# 
# 
# 
# 

# In[173]:


train = train.fillna(-1)
test = test.fillna(-1)


# The .fillna() method is used to fill any missing values in a dataframe with a specific value. In this case, the missing values are being filled with -1.

# In[174]:


import gc #garbage collector
del members, songs; gc.collect()


# This is done to free up the memory space occupied by the unnecessary dataframes and to prevent memory overflow.

# In[175]:


colm = list(train.columns)
colm.remove('target')


# The above code is creating a list of columns from the train dataframe, and then using the list method 'remove' to remove the column 'target' from the list. This is done because the target variable is the variable that we are trying to predict, it should not be included in the list of features to be used for training the model.

# In[176]:


from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
for col in tqdm(colm):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])


# The process is done for all columns in the dataset that have an object data type, which typically corresponds to categorical variables.It is important as most machine learning models take numerical inputs, so it is necessary to convert categorical variables to numerical variables before feeding the data to the model.

# In[177]:


unique_songs = range(max(train['song_id'].max(), test['song_id'].max()))
song_popularity = pd.DataFrame({'song_id': unique_songs, 'popularity':0})

train_sorted = train.sort_values('song_id')
train_sorted.reset_index(drop=True, inplace=True)
test_sorted = test.sort_values('song_id')
test_sorted.reset_index(drop=True, inplace=True)


# In[178]:


pip install lightgbm


# ## Model Building 

# In[179]:


from sklearn.model_selection import train_test_split
import lightgbm as lgb
X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

del train, test; gc.collect();

X_train, X_valid, y_train, y_valid = train_test_split(X, y,     test_size=0.1, random_state = 12)
    
del X, y; gc.collect();

d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


# The above code is splitting the training data into training and validation sets using the train_test_split function from sklearn.model_selection, with a test size of 0.1, which means 10% of the data will be used for validation and the remaining 90% will be used for training. The LightGBM library's Dataset object is then created for the training and validation sets, which is used for training the model. The watchlist variable is a list of Dataset objects that is used to track the model's performance during training.

# ## Trying out some basic classification models

# In[182]:


def predict(m1_model):
    model = m1_model.fit(X_train,y_train)
    print('Training Score : {}'.format(model.score(X_train,y_train)))
    y_pred = model.predict(X_valid)
    v_test = model.predict(X_test)
    yhat = (v_test>0.5).astype(int)
    comp = (yhat==ytr).astype(int)
    acc = comp.sum()/comp.size*100
    print("Accuracy on test data for the model", acc)


# ## Logistic Regression & Random Forest Classifier

# In[183]:


from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression()
predict(m1)


# In[184]:


from sklearn.ensemble import RandomForestClassifier
m2 = RandomForestClassifier()
predict(m2)


# In[187]:


pip install xgboost


# In[188]:


from xgboost import XGBClassifier
m3 = XGBClassifier()
predict(m3)


# ## Prediction using lightgbm

# In[204]:


params = {}
params['learning_rate'] = 0.3
params['application'] = 'binary'
params['max_depth'] = 15
params['num_leaves'] = 2**8
params['verbosity'] = 0
params['metric'] = 'auc'

model1 = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, callbacks=[lgb.early_stopping(10), lgb.log_evaluation(period=10)])


# In[192]:


p_test = model1.predict(X_test)


# ## Printing accuracy of lgbm model on test data

# In[205]:


yhat = (p_test>0.5).astype(int)
comp = (yhat==ytr).astype(int)
acc = comp.sum()/comp.size*100
print('The accuracy of lgbm model on test data is: {0:f}%'.format(acc))


# In[ ]:




