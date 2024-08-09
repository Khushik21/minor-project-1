#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error


# In[2]:


import pandas as pd

# Read the CSV file
data = pd.read_csv(r'C:\Users\Khushi\Desktop\world.csv', decimal=',')

# Display the number of missing data
print('Number of missing data:')
print(data.isnull().sum())

# Display summary statistics
print('Summary statistics:')
print(data.describe(include='all'))


# In[3]:


data.groupby('Region')[['GDP ($ per capita)','Literacy (%)','Agriculture']].median()
for col in data.columns.values:
    if data[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = data.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = data.groupby('Region')[col].median()
    for region in data['Region'].unique():
        data[col].loc[(data[col].isnull())&(data['Region']==region)] = guess_values[region]


# In[4]:


fig, ax = plt.subplots(figsize=(16,6))
#ax = fig.add_subplot(111)
top_gdp_countries = data.sort_values('GDP ($ per capita)',ascending=False).head(20)
mean = pd.DataFrame({'Country':['World mean'], 'GDP ($ per capita)':[data['GDP ($ per capita)'].mean()]})
gdps = pd.concat([top_gdp_countries[['Country','GDP ($ per capita)']],mean],ignore_index=True)

sns.barplot(x='Country',y='GDP ($ per capita)',data=gdps, palette='Set3')
ax.set_xlabel(ax.get_xlabel(),labelpad=15)
ax.set_ylabel(ax.get_ylabel(),labelpad=30)
ax.xaxis.label.set_fontsize(16)
ax.yaxis.label.set_fontsize(16)
plt.xticks(rotation=90)
plt.show()


# In[5]:


plt.figure(figsize=(16,12))
sns.heatmap(data=data.iloc[:,2:].corr(),annot=True,fmt='.2f',cmap='coolwarm')
plt.show()


# In[6]:


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20,12))
plt.subplots_adjust(hspace=0.4)

corr_to_gdp = pd.Series()
for col in data.columns.values[2:]:
    if ((col!='GDP ($ per capita)')&(col!='Climate')):
        corr_to_gdp[col] = data['GDP ($ per capita)'].corr(data[col])
abs_corr_to_gdp = corr_to_gdp.abs().sort_values(ascending=False)
corr_to_gdp = corr_to_gdp.loc[abs_corr_to_gdp.index]

for i in range(2):
    for j in range(3):
        sns.regplot(x=corr_to_gdp.index.values[i*3+j], y='GDP ($ per capita)', data=data,
                   ax=axes[i,j], fit_reg=False, marker='.')
        title = 'correlation='+str(corr_to_gdp[i*3+j])
        axes[i,j].set_title(title)
axes[1,2].set_xlim(0,102)
plt.show()


# In[7]:


data.loc[(data['Birthrate']<14)&(data['GDP ($ per capita)']<10000)]


# In[8]:


LE = LabelEncoder()
data['Region_label'] = LE.fit_transform(data['Region'])
data['Climate_label'] = LE.fit_transform(data['Climate'])
data.head()


# In[9]:


train, test = train_test_split(data, test_size=0.3, shuffle=True)
training_features = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'Literacy (%)', 'Phones (per 1000)',
       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Region_label',
       'Climate_label','Service']
target = 'GDP ($ per capita)'
train_X = train[training_features]
train_Y = train[target]
test_X = test[training_features]
test_Y = test[target]


# In[10]:


model = LinearRegression()
model.fit(train_X, train_Y)
train_pred_Y = model.predict(train_X)
test_pred_Y = model.predict(test_X)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
msle_train = mean_squared_log_error(train_pred_Y, train_Y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
msle_test = mean_squared_log_error(test_pred_Y, test_Y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)


# In[11]:


model = RandomForestRegressor(n_estimators = 50,
                             max_depth = 6,
                             min_weight_fraction_leaf = 0.05,
                             max_features = 0.8,
                             random_state = 42)
model.fit(train_X, train_Y)
train_pred_Y = model.predict(train_X)
test_pred_Y = model.predict(test_X)
train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)

rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
msle_train = mean_squared_log_error(train_pred_Y, train_Y)
rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
msle_test = mean_squared_log_error(test_pred_Y, test_Y)

print('rmse_train:',rmse_train,'msle_train:',msle_train)
print('rmse_test:',rmse_test,'msle_test:',msle_test)


# In[12]:


plt.figure(figsize=(18,12))

train_test_Y = train_Y.append(test_Y)
train_test_pred_Y = train_pred_Y.append(test_pred_Y)

data_shuffled = data.loc[train_test_Y.index]
label = data_shuffled['Country']

colors = {'ASIA (EX. NEAR EAST)         ':'red',
          'EASTERN EUROPE                     ':'orange',
          'NORTHERN AFRICA                    ':'gold',
          'OCEANIA                            ':'green',
          'WESTERN EUROPE                     ':'blue',
          'SUB-SAHARAN AFRICA                 ':'purple',
          'LATIN AMER. & CARIB    ':'olive',
          'C.W. OF IND. STATES ':'cyan',
          'NEAR EAST                          ':'hotpink',
          'NORTHERN AMERICA                   ':'lightseagreen',
          'BALTICS                            ':'rosybrown'}

for region, color in colors.items():
    X = train_test_Y.loc[data_shuffled['Region']==region]
    Y = train_test_pred_Y.loc[data_shuffled['Region']==region]
    ax = sns.regplot(x=X, y=Y, marker='.', fit_reg=False, color=color, scatter_kws={'s':200, 'linewidths':0}, label=region) 
plt.legend(loc=4,prop={'size': 12})  

ax.set_xlabel('GDP ($ per capita) ground truth',labelpad=40)
ax.set_ylabel('GDP ($ per capita) predicted',labelpad=40)
ax.xaxis.label.set_fontsize(24)
ax.yaxis.label.set_fontsize(24)
ax.tick_params(labelsize=12)

x = np.linspace(-1000,50000,100) # 100 linearly spaced numbers
y = x
plt.plot(x,y,c='gray')

plt.xlim(-1000,60000)
plt.ylim(-1000,40000)

for i in range(0,train_test_Y.shape[0]):
    if((data_shuffled['Area (sq. mi.)'].iloc[i]>8e5) |
       (data_shuffled['Population'].iloc[i]>1e8) |
       (data_shuffled['GDP ($ per capita)'].iloc[i]>10000)):
        plt.text(train_test_Y.iloc[i]+200, train_test_pred_Y.iloc[i]-200, label.iloc[i], size='small')


# In[13]:


from sklearn.tree import DecisionTreeClassifier

# Define the features and target variable
training_features = ['Population', 'Area (sq. mi.)',
       'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
       'Net migration', 'Infant mortality (per 1000 births)',
       'Literacy (%)', 'Phones (per 1000)',
       'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
       'Deathrate', 'Agriculture', 'Industry', 'Service', 'Region_label',
       'Climate_label','Service']
target = 'Region'  # Assuming you want to classify countries into regions

# Define and train the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(train_X, train_Y)

# Predict on the training and testing sets
train_pred_Y = classifier.predict(train_X)
test_pred_Y = classifier.predict(test_X)

# Evaluate the model
accuracy_train = classifier.score(train_X, train_Y)
accuracy_test = classifier.score(test_X, test_Y)

print("Accuracy on training set:", accuracy_train)
print("Accuracy on testing set:", accuracy_test)


# In[14]:


import scipy.stats as stats
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix,recall_score,precision_score,f1_score,mean_squared_error, mean_absolute_error,classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


# In[17]:


from sklearn.tree import plot_tree

class_names = [str(cls) for cls in classifier.classes_]

plt.figure(figsize=(20, 10))
plot_tree(classifier, filled=True, feature_names=training_features, class_names=class_names)
plt.show()




# In[ ]:


# Hyperparameter Tuning Using GridSearchCV


# In[21]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Define the parameter grid
param_grid = {
    'max_depth': [3, 6, 9, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Decision Tree Classifier
classifier = DecisionTreeClassifier(random_state=42)

# Initialize GridSearchCV with 3-fold cross-validation
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Print the best cross-validation score
print("Best cross-validation score:", grid_search.best_score_)




# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define the range of maximum depths to explore
depths = range(1, 21)  # Adjust the range as needed

# Initialize lists to store accuracies
train_accuracies = []
test_accuracies = []

# Iterate over different maximum depths
for depth in depths:
    # Initialize Decision Tree Classifier with the current maximum depth
    classifier = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # Train the classifier
    classifier.fit(train_X, train_Y)
    
    # Predict on training and testing sets
    train_pred_Y = classifier.predict(train_X)
    test_pred_Y = classifier.predict(test_X)
    
    # Calculate accuracy for training and testing sets
    train_accuracy = accuracy_score(train_Y, train_pred_Y)
    test_accuracy = accuracy_score(test_Y, test_pred_Y)
    
    # Append accuracies to the lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the accuracies vs. depths
plt.figure(figsize=(10, 6))
plt.plot(depths, train_accuracies, label='Train Accuracy')
plt.plot(depths, test_accuracies, label='Test Accuracy')
plt.xlabel('Depth of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Depth of Decision Tree Classifier')
plt.legend()
plt.grid(True)
plt.show()


# In[23]:


# Define the range of values for min_samples_split
min_samples_splits = range(2, 21)  # Adjust the range as needed

# Initialize lists to store accuracies
train_accuracies = []
test_accuracies = []

# Iterate over different values of min_samples_split
for min_samples_split in min_samples_splits:
    # Initialize Decision Tree Classifier with the current min_samples_split
    classifier = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)
    
    # Train the classifier
    classifier.fit(train_X, train_Y)
    
    # Predict on training and testing sets
    train_pred_Y = classifier.predict(train_X)
    test_pred_Y = classifier.predict(test_X)
    
    # Calculate accuracy for training and testing sets
    train_accuracy = accuracy_score(train_Y, train_pred_Y)
    test_accuracy = accuracy_score(test_Y, test_pred_Y)
    
    # Append accuracies to the lists
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot the accuracies vs. min_samples_split
plt.figure(figsize=(10, 6))
plt.plot(min_samples_splits, train_accuracies, label='Train Accuracy')
plt.plot(min_samples_splits, test_accuracies, label='Test Accuracy')
plt.xlabel('Min Samples Split of Decision Tree')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Min Samples Split of Decision Tree Classifier')
plt.legend()
plt.grid(True)
plt.show()


# #  Decision Regressor Tree

# In[25]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Initialize Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)

# Train the regressor
regressor.fit(train_X, train_Y)

# Predict on training and testing sets
train_pred_Y = regressor.predict(train_X)
test_pred_Y = regressor.predict(test_X)

# Calculate RMSE for training and testing sets
rmse_train = np.sqrt(mean_squared_error(train_Y, train_pred_Y))
rmse_test = np.sqrt(mean_squared_error(test_Y, test_pred_Y))

print('RMSE for training set:', rmse_train)
print('RMSE for testing set:', rmse_test)
# Calculate R2 score for training and testing sets
r2_train = r2_score(train_Y, train_pred_Y)
r2_test = r2_score(test_Y, test_pred_Y)

print('R2 score for training set:', r2_train)
print('R2 score for testing set:', r2_test)


# In[26]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [3, 6, 9, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Decision Tree Regressor
regressor = DecisionTreeRegressor(random_state=42)

# Initialize GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Print the best cross-validation score
print("Best cross-validation score:", grid_search.best_score_)


# In[28]:


from sklearn.metrics import r2_score

# True target values
y_true = [3, -0.5, 2, 7]

# Predicted target values
y_pred = [2.5, 0.0, 2, 8]

# Calculate R2 score
r2 = r2_score(y_true, y_pred)

print("R2 score:", r2)



# # Ensemble Methods
# 

# # Bagging

# In[35]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Bagging Classifier
bagging_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    n_estimators=500,
    max_samples=0.5,
    bootstrap=True,
    random_state=42
)

# Train the Bagging Classifier
bagging_classifier.fit(train_X, train_Y)

# Predict on training and testing sets
train_pred_Y_bagging = bagging_classifier.predict(train_X)
test_pred_Y_bagging = bagging_classifier.predict(test_X)

# Calculate accuracy for training and testing sets
train_accuracy_bagging = accuracy_score(train_Y, train_pred_Y_bagging)
test_accuracy_bagging = accuracy_score(test_Y, test_pred_Y_bagging)

print('Accuracy for training set (with bagging):', train_accuracy_bagging)





# # Pasting:

# In[36]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Pasting Classifier
pasting_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    n_estimators=500,
    max_samples=0.5,
    bootstrap=False,  # Setting bootstrap to False for pasting
    random_state=42
)

# Train the Pasting Classifier
pasting_classifier.fit(train_X, train_Y)

# Predict on training and testing sets
train_pred_Y_pasting = pasting_classifier.predict(train_X)
test_pred_Y_pasting = pasting_classifier.predict(test_X)

# Calculate accuracy for training and testing sets
train_accuracy_pasting = accuracy_score(train_Y, train_pred_Y_pasting)
test_accuracy_pasting = accuracy_score(test_Y, test_pred_Y_pasting)

print('Accuracy for training set (with pasting):', train_accuracy_pasting)
print('Accuracy for testing set (with pasting):', test_accuracy_pasting)


# # Random Subspaces:

# In[37]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Random Subspaces Classifier
random_subspaces_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    n_estimators=500,
    max_features=0.5,  # Considering random subsets of features
    bootstrap=False,  # Setting bootstrap to False for Random Subspaces
    random_state=42
)

# Train the Random Subspaces Classifier
random_subspaces_classifier.fit(train_X, train_Y)

# Predict on training and testing sets
train_pred_Y_random_subspaces = random_subspaces_classifier.predict(train_X)
test_pred_Y_random_subspaces = random_subspaces_classifier.predict(test_X)

# Calculate accuracy for training and testing sets
train_accuracy_random_subspaces = accuracy_score(train_Y, train_pred_Y_random_subspaces)
test_accuracy_random_subspaces = accuracy_score(test_Y, test_pred_Y_random_subspaces)

print('Accuracy for training set (with Random Subspaces):', train_accuracy_random_subspaces)
print('Accuracy for testing set (with Random Subspaces):', test_accuracy_random_subspaces)


# # Random Patches:

# In[38]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Random Patches Classifier
random_patches_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    n_estimators=500,
    max_samples=0.5,  # Considering random subsets of samples
    max_features=0.5,  # Considering random subsets of features
    bootstrap=True,  # Setting bootstrap to True for Random Patches
    random_state=42
)

# Train the Random Patches Classifier
random_patches_classifier.fit(train_X, train_Y)

# Predict on training and testing sets
train_pred_Y_random_patches = random_patches_classifier.predict(train_X)
test_pred_Y_random_patches = random_patches_classifier.predict(test_X)

# Calculate accuracy for training and testing sets
train_accuracy_random_patches = accuracy_score(train_Y, train_pred_Y_random_patches)
test_accuracy_random_patches = accuracy_score(test_Y, test_pred_Y_random_patches)

print('Accuracy for training set (with Random Patches):', train_accuracy_random_patches)
print('Accuracy for testing set (with Random Patches):', test_accuracy_random_patches)


# # OOB(Out of Bag) Score:

# In[39]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Bagging Classifier
bagging_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    n_estimators=500,
    oob_score=True,  # Enable calculation of OOB score
    random_state=42
)

# Train the Bagging Classifier
bagging_classifier.fit(train_X, train_Y)

# OOB score
oob_score = bagging_classifier.oob_score_
print('OOB score:', oob_score)


# # Applying GridSearchCV:

# In[42]:


from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

# Initialize base classifier (Decision Tree Classifier in this case)
base_classifier = DecisionTreeClassifier()

# Initialize Bagging Classifier
bagging_classifier = BaggingClassifier(
    base_estimator=base_classifier,
    random_state=42
)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_samples': [0.5, 0.7, 1.0],
    'max_features': [0.5, 0.7, 1.0],
}

# Initialize KFold cross-validation with a smaller number of splits
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Initialize GridSearchCV with KFold
grid_search = GridSearchCV(estimator=bagging_classifier, param_grid=param_grid, cv=kf)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_bagging_classifier = grid_search.best_estimator_

# Predict on testing set using the best model
test_pred_Y = best_bagging_classifier.predict(test_X)

# Calculate accuracy for the best model
accuracy = accuracy_score(test_Y, test_pred_Y)
print('Accuracy:', accuracy)


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

# Initialize Random Forest Classifier
random_forest_classifier = RandomForestClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=random_forest_classifier, param_grid=param_grid, cv=5)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_random_forest_classifier = grid_search.best_estimator_

# Predict on testing set using the best model
test_pred_Y = best_random_forest_classifier.predict(test_X)

# Calculate accuracy for the best model
accuracy = accuracy_score(test_Y, test_pred_Y)
print('Accuracy:', accuracy)



# # Random Forest Regressor

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest Regressor
random_forest_regressor = RandomForestRegressor(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=random_forest_regressor, param_grid=param_grid, cv=5)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_random_forest_regressor = grid_search.best_estimator_

# Predict on testing set using the best model
test_pred_Y = best_random_forest_regressor.predict(test_X)

# Calculate RMSE for the best model
rmse = np.sqrt(mean_squared_error(test_Y, test_pred_Y))
print('RMSE:', rmse)



# In[ ]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extract one of the decision trees from the Random Forest
one_tree = best_random_forest_regressor.estimators_[0]

# Plot the decision tree
plt.figure(figsize=(20, 10))
plot_tree(one_tree, filled=True, feature_names=training_features)
plt.show()


# # Adaboost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

# Initialize AdaBoost Classifier
adaboost_classifier = AdaBoostClassifier(random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.1, 0.5, 1.0]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=adaboost_classifier, param_grid=param_grid, cv=5)

# Perform grid search
grid_search.fit(train_X, train_Y)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_adaboost_classifier = grid_search.best_estimator_

# Predict on testing set using the best model
test_pred_Y = best_adaboost_classifier.predict(test_X)

# Calculate accuracy for the best model
accuracy = accuracy_score(test_Y, test_pred_Y)
print('Accuracy:', accuracy)


# # XGBoost

# In[ ]:


import xgboost as xgb

# Convert data into DMatrix format required by XGBoost
dtrain = xgb.DMatrix(train_X, label=train_Y)
dtest = xgb.DMatrix(test_X)

# Define hyperparameters
params = {
    'max_depth': 3,
    'eta': 0.1,
    'objective': 'reg:squarederror',  # Use 'reg:squarederror' for regression tasks
    'eval_metric': 'rmse'
}

# Train the XGBoost model
num_rounds = 100  # You can adjust this number
xgb_model = xgb.train(params, dtrain, num_rounds)

# Predict on the testing set
test_pred_Y = xgb_model.predict(dtest)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_Y, test_pred_Y))
print('RMSE:', rmse)


# In[ ]:


#Accuracy vs Number of Estimators for XGBoost Classifier


# In[ ]:


import xgboost as xgb
import matplotlib.pyplot as plt

# Initialize lists to store accuracy and number of estimators
accuracies = []
num_estimators_list = [50, 100, 150, 200, 250]  # You can adjust this list

# Convert data into DMatrix format required by XGBoost
dtrain = xgb.DMatrix(train_X, label=train_Y)
dtest = xgb.DMatrix(test_X)

for num_estimators in num_estimators_list:
    # Define hyperparameters
    params = {
        'max_depth': 3,
        'eta': 0.1,
        'objective': 'multi:softmax',  # Use 'multi:softmax' for multiclass classification tasks
        'num_class': len(train_Y.unique())  # Number of classes in your target variable
    }

    # Train the XGBoost model
    xgb_model = xgb.train(params, dtrain, num_estimators)

    # Predict on the testing set
    test_pred_Y = xgb_model.predict(dtest)

    # Calculate accuracy
    accuracy = accuracy_score(test_Y, test_pred_Y)
    accuracies.append(accuracy)

# Plot Accuracy vs Number of Estimators
plt.figure(figsize=(10, 6))
plt.plot(num_estimators_list, accuracies, marker='o')
plt.title('Accuracy vs Number of Estimators for XGBoost Classifier')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()


# # K Nearest Neighbours

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

# Create k-NN regressor model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed

# Train the model
knn_model.fit(train_X, train_Y)

# Make predictions
train_pred_Y_knn = knn_model.predict(train_X)
test_pred_Y_knn = knn_model.predict(test_X)

# Evaluate the model
rmse_train_knn = np.sqrt(mean_squared_error(train_pred_Y_knn, train_Y))
msle_train_knn = mean_squared_log_error(train_pred_Y_knn, train_Y)
rmse_test_knn = np.sqrt(mean_squared_error(test_pred_Y_knn, test_Y))
msle_test_knn = mean_squared_log_error(test_pred_Y_knn, test_Y)

print('KNN Regression Results:')
print('-----------------------')
print('RMSE (Train):', rmse_train_knn)
print('MSLE (Train):', msle_train_knn)
print('RMSE (Test):', rmse_test_knn)
print('MSLE (Test):', msle_test_knn)



# # Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB

# Create Naive Bayes classifier model
naive_bayes_model = GaussianNB()

# Train the model
naive_bayes_model.fit(train_X, train_Y)

# Make predictions
train_pred_Y_nb = naive_bayes_model.predict(train_X)
test_pred_Y_nb = naive_bayes_model.predict(test_X)

# Evaluate the model (if you have labels for the test data)
accuracy_train_nb = naive_bayes_model.score(train_X, train_Y)
accuracy_test_nb = naive_bayes_model.score(test_X, test_Y)

print('Naive Bayes Classifier Results:')
print('------------------------------')
print('Train Accuracy:', accuracy_train_nb)
print('Test Accuracy:', accuracy_test_nb)


# # Support Vector Machine

# In[ ]:


from sklearn.svm import SVR

# Create SVM regressor model
svm_model = SVR(kernel='linear')  # You can choose different kernels like 'linear', 'rbf', 'poly', etc.

# Train the model
svm_model.fit(train_X, train_Y)

# Make predictions
train_pred_Y_svm = svm_model.predict(train_X)
test_pred_Y_svm = svm_model.predict(test_X)

# Evaluate the model
rmse_train_svm = np.sqrt(mean_squared_error(train_pred_Y_svm, train_Y))
msle_train_svm = mean_squared_log_error(train_pred_Y_svm, train_Y)
rmse_test_svm = np.sqrt(mean_squared_error(test_pred_Y_svm, test_Y))
msle_test_svm = mean_squared_log_error(test_pred_Y_svm, test_Y)

print('SVM Regression Results:')
print('-----------------------')
print('RMSE (Train):', rmse_train_svm)
print('MSLE (Train):', msle_train_svm)
print('RMSE (Test):', rmse_test_svm)
print('MSLE (Test):', msle_test_svm)


# # Support Vector Regressor

# In[ ]:


from sklearn.svm import SVR

# Create SVR model
svr_model = SVR(kernel='rbf')  # You can choose different kernels like 'linear', 'poly', 'rbf', etc.

# Train the model
svr_model.fit(train_X, train_Y)

# Make predictions
train_pred_Y_svr = svr_model.predict(train_X)
test_pred_Y_svr = svr_model.predict(test_X)

# Evaluate the model
rmse_train_svr = np.sqrt(mean_squared_error(train_pred_Y_svr, train_Y))
msle_train_svr = mean_squared_log_error(train_pred_Y_svr, train_Y)
rmse_test_svr = np.sqrt(mean_squared_error(test_pred_Y_svr, test_Y))
msle_test_svr = mean_squared_log_error(test_pred_Y_svr, test_Y)

print('SVR Regression Results:')
print('-----------------------')
print('RMSE (Train):', rmse_train_svr)
print('MSLE (Train):', msle_train_svr)
print('RMSE (Test):', rmse_test_svr)
print('MSLE (Test):', msle_test_svr)


# # Principle Component Analysis

# In[ ]:


from sklearn.decomposition import PCA

# Initialize PCA
pca = PCA(n_components=10)  # Specify the number of principal components to retain

# Fit PCA to the training data and transform both training and testing data
train_X_pca = pca.fit_transform(train_X)
test_X_pca = pca.transform(test_X)

# Now you can use train_X_pca and test_X_pca as the transformed features
# and apply any regression or classification algorithm on them


# In[ ]:




