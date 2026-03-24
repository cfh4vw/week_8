# %% [markdown]
# # Section Q3- Wine Quality
# %%
# necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
# %% [markdown]
# *************************************************
# Question 1
# *************************************************

# %%
# Load in Wine quality dataset
df = pd.read_csv('winequality-red.csv')
df.info()

# I chose this dataset because we used similar ones in our Design class,
# and everyone loves wine! Pulled from Kaggle
# can be found at this link: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 
# %% [markdown]
# *************************************************
# Question 2
# *************************************************

# %%
# EDA- 
# no missing values yay!
# opened in Data Wrangler- all numeric features
#   lots of skewed vars though
#   right skewed: residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, sulphates, alcohol
#   the rest are roughly symmetric so we will leave them be
#   
#   our target variable will be quality--> how well can a linear model predict wine quality?

# apply log transformations
skew_cols = ['residual sugar', 'chlorides', 'free sulfur dioxide', 
             'total sulfur dioxide', 'sulphates', 'alcohol']
for col in skew_cols:
    df[f'log_{col}'] = np.log(df[col] + 1)

# drop redundant columns
df = df.drop(columns=skew_cols)
df.head()

# view a correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# looks like with our target variable, the most highly correlated vars
# are 'log_alcohol', 'volatile acidity', 'log_sulphates', and 'citric acid'
#
# we'll do a few models: with the vars above, with every variable, and 
# with some randomly selected variables

# separate target variable
yall = df['quality']
xall = df.drop(columns=['quality'])
# we have all the features in this right now, but we'll create
# more specialized models later!
# %% [markdown]
# *************************************************
# Question 3
# *************************************************

# %%
# split the data
X_train, X_test, y_train, y_test = train_test_split(
    xall, yall, test_size=0.2, random_state=42
)

# %% [markdown]
# *************************************************
# Question 4
# *************************************************

# %%
# regression model 1 with the highest correlated vars
corr_cols = ['log_alcohol', 'volatile acidity', 'log_sulphates', 'citric acid']

model1 = LinearRegression().fit(X_train[corr_cols],y_train)
y_pred = model1.predict(X_test[corr_cols])

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

# This isn't great honestly, but let's try some other vars and see 
# which is best
# %%
# regression model 2 with the other vars, to see how they perform
features = X_train.drop(columns=corr_cols).columns

model2 = LinearRegression().fit(X_train[features], y_train)
y_pred_others = model2.predict(X_test[features])

mse = mean_squared_error(y_test, y_pred_others)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_others)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

# This is way worse, as expected, but lets combine them and see
# how the model does with all the variables.
# %%
# regression model 3 with all vars
model3 = LinearRegression().fit(X_train, y_train)
y_pred_all = model3.predict(X_test)

mse = mean_squared_error(y_test, y_pred_all)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_all)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

# not bad! R^2 isn't perfect and neither is the RMSE, but it's the 
# best so far.
# %% [markdown]
# *************************************************
# Question 5
# *************************************************

# %% [markdown]
# The 3rd model I made performed the best, I think because it had
# the most variables to base its predictions off of. While not all
# of these variables were super helpful, as shown in model 2, 
# having some correlation with the target variable quality made
# a small difference in the R2 and RMSE. 
#
# However, since there was only essentially a 0.01 difference in R2
# and in RMSE from models 1 to 3, we can infer that adding the 
# other variables added only a little information to the model and 
# also a lot of noise. 

# %% [markdown]
# *************************************************
# Question 6
# *************************************************

# %% [markdown]
# From exploring this dataset, I first understood much deeper
# how much a correlation matrix can help in selecting the most
# predictive features. Also, in creating the models, I learned
# that more features isn't always necessarily better. Specifically,
# model 2 had many more features to regress on and performed far worse
# than model 1. 
#
# Specific to this dataset, I conclude that my best regression model
# based on all the aspects of a wine only account for 40.67% of 
# the variation in wine quality. 
#
# As you can see in the residual plot below, the data is slightly
# skewed to the left, meaning that my model has a tendency to 
# overpredict wine quality. 
#
# I am also realizing as I write this that maybe a classification
# algorithm would have been better for this data. :(
# Nonetheless, I now realize the importance of being slightly
# picky when choosing the correct models to use for Machine Learning.
# This data should have probably been interpreted as a discrete
# target variable, not a continuous one, since the quality ratings
# are each a whole number. I had assumed that they would have decimals
# but you know what they say about assuming...
# %%
# residual plot
resid = y_test - y_pred_all
resid.plot.kde(color='pink')
plt.show()

# %%
