# %% [markdown]
# # Section Q1- Housing
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

# %%
# load the data
df = pd.read_csv('https://raw.githubusercontent.com/DS3001/linearRegression/refs/heads/main/data/Q1_clean.csv')
df.head()
# %% [markdown]
# *************************************************
# Question 1
# *************************************************

# %%
# Compute the average prices and scores by `Neighborhood `
avg_price = df.groupby("Neighbourhood ")['Price'].mean()
avg_score = df.groupby("Neighbourhood ")['Review Scores Rating'].mean()
# which borough is the most expensive on average? 
print(f"Average Price: \n{avg_price}")
print("On average, Manhattan is the most expensive borough.\n")
# Create a kernel density plot of price and log price, grouping by `Neighborhood `.

df_with_log = df.copy()
log_price = np.log(df['Price'] + 1)
df_with_log['log_price'] = log_price

avg_log_price = df_with_log.groupby("Neighbourhood ")['log_price'].mean()
print(f"Average Log Price: \n{avg_log_price}\n")

fig, ax = plt.subplots(1, 2, figsize=(10,4))
df.groupby('Neighbourhood ')['Price'].plot.kde(ax=ax[0])
df_with_log.groupby('Neighbourhood ')['log_price'].plot.kde(ax=ax[1])

ax[0].set_title('KDE of price grouped by neighborhood')
ax[1].set_title('KDE of log price grouped by neighborhood')
plt.legend()
plt.show()
# %% [markdown]
# *************************************************
# Question 2
# *************************************************

# %%
# one-hot encoding
df_dummies = pd.get_dummies(df_with_log, columns=['Neighbourhood ', 'Property Type', 'Room Type'], drop_first=False, prefix=['Hood', 'Prop_Type', 'Room_Type'])
# don't drop first, no need to do this since we aren't including an intercept

# create predictor and target vars
X_cols = df_dummies[['Hood_Bronx','Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island']]
y = df_dummies['log_price']

# create model without intercept
model_without = LinearRegression(fit_intercept=False).fit(X_cols, y)

# print coefficients
print(f"Without Intercept: Coefficient = {model_without.coef_}, R² = {model_without.score(X_cols, y):.4f}")

# When comparing the coefficients to the table from part 1,
# they are the same as the average log prices per neighborhood.
# I didn't include the Bronx since apparently there were no 
# entries with the neighborhood listed as the Bronx. 
# This shows that the coefficients of a regression of a continuous
# variable onto one categorical variable is the same as the 
# category means.
# %% [markdown]
# *************************************************
# Question 3
# *************************************************
# %%
# now with the intercept, we want to set drop_first=True
# to avoid the dummy variable trap
df_dummies_intercept = pd.get_dummies(df_with_log, columns=['Neighbourhood ', 'Property Type', 'Room Type'], drop_first=True, prefix=['Hood', 'Prop_Type', 'Room_Type'])

# create predictor and target vars
X_cols = df_dummies_intercept[['Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island']]
y = df_dummies_intercept['log_price']

# create model without intercept
model_with = LinearRegression(fit_intercept=True).fit(X_cols, y)

# print coefficients
print(f"Without Intercept: Coefficient = {model_with.coef_},Intercept = {model_with.intercept_:.2f}, R² = {model_with.score(X_cols, y):.4f}")

# I dropped the first dummy variable to avoid multicollinearity.
# To do this, set drop_first=True and DON'T include 'Hood_Bronx' 
# in the X_cols list, since it doesn't exist in the index. 
# 
# When we include the intercept, we can see that the intercept
# of our model is actually equal to the previous coefficient
# for the first dummy, the Bronx (4.22)!
#
# If you want to get the coefficients from part 2 (or the log means),
# just add the NEW coefficients to the intercept of this model.
# %% [markdown]
# *************************************************
# Question 4
# *************************************************
# %%
# select mlr features
mlr_features = ['Review Scores Rating', 'Neighbourhood ']

df_mv = df_with_log[mlr_features + ['log_price']]

df_mv = pd.get_dummies(df_mv, columns=['Neighbourhood '], drop_first=True, prefix=['Hood',])
df_mv.head()

X_mv = df_mv[['Review Scores Rating','Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island']]
y_mv = df_mv['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X_mv, y_mv, test_size=0.2, random_state=42
)

model_mv = LinearRegression().fit(X_train, y_train)
y_pred = model_mv.predict(X_test) 

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# print coefficients
features = ['Review Scores Rating','Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island']
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model_mv.coef_})
print(coef_df.to_string(index=False))

# The coefficient for Review Scores Rating is 0.007832.
# The most expensive property you can rent is in Manhattan.
# %% [markdown]
# *************************************************
# Question 5
# *************************************************
# %%
mlr_features = ['Review Scores Rating', 'Neighbourhood ', 'Property Type']

df_mv = df_with_log[mlr_features + ['log_price']]

df_mv = pd.get_dummies(df_mv, columns=['Neighbourhood ', 'Property Type'], drop_first=True, prefix=['Hood','Prop_type'])
df_mv.head()

prop_cols = [col for col in df_mv.columns if col.startswith('Prop_type_')]
X_mv = df_mv[['Review Scores Rating','Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island']+prop_cols]
y_mv = df_mv['log_price']

X_train, X_test, y_train, y_test = train_test_split(
    X_mv, y_mv, test_size=0.2, random_state=42
)

model_mv = LinearRegression().fit(X_train, y_train)
y_pred = model_mv.predict(X_test) 

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# print coefficients
features = ['Review Scores Rating','Hood_Brooklyn','Hood_Manhattan','Hood_Queens','Hood_Staten Island'] + prop_cols
coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model_mv.coef_})
print(coef_df.to_string(index=False))

# The coefficient for Review Scores Rating is 0.007630.
# The most expensive property TYPE you can rent is a Bungalow,
# and the most expensive neighborhood is still, unsurprisingly, Manhattan.
# %% [markdown]
# *************************************************
# Question 6
# *************************************************
# %% [markdown]
# When the coefficient of Review Scores Rating changes between parts 
# 5 and 6, it means that the effectiveness of RSR as a predictor
# variable changes. 
#
# In this example, its predictive value gets worse relative to the
# other variables when property type is taken into consideration.
# However, since both coefficients are extremely low, we can assume
# the RSR is a bad predictor variable in general.


# %% [markdown]
# # Section Q2- Cars

# %% [markdown]
# *************************************************
# Question 1
# *************************************************

# %%
# load the data
df = pd.read_csv('cars_hw.csv')
df.head() # maybe look at price and mileage to potentially transform them

#df["Mileage_Run"].plot.kde(color='steelblue')
#plt.title('Kernel Density Plot of Mileage') #Looks pretty normal... 
# It looks pretty skewed in data wrangler though as well, lets do another 
# log transformation with this one too.

#df["Price"].plot.kde(color='steelblue')
#plt.title('Kernel Density Plot of Price') #definitely skewed to the right..
# We'll do a log transformation on price

# apply transformations
df['log_price'] = np.log(df['Price'] + 1)
df['log_mileage'] = np.log(df['Mileage_Run'] + 1)

# Drop unnecessary/redundant columns
df = df.drop(columns=['Unnamed: 0','Price','Mileage_Run'])
df.head()

# %%[markdown]
# *************************************************
# Question 2
# *************************************************

# %%
# summary of price
print(df.groupby('Make')['log_price'].describe())
df.groupby('Make')['log_price'].plot.kde()
plt.legend()

# The most expensive car brands are MG Motors, Kia, and Jeep.
# In general, the log prices don't look too skewed, and most brands 
# have a relatively small standard deviation. From the graph, we can 
# see that Datsun is a bit of an outlier, since they only have 3
# observations in the dataset. 
#
# There is a central tendence around the log price of 13-13.5, 
# and looks like there are very few cars that are way above or way below
# that value.
# %%[markdown]
# *************************************************
# Question 3
# *************************************************

# %%
# splitting the data
y_all = df['log_price']
X_all = df.drop(columns=['log_price'])
 
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)
# %%[markdown]
# *************************************************
# Question 4
# *************************************************

# %%
# numeric regression only
numeric_features = X_train.select_dtypes(include=[np.number]).columns

model_num = LinearRegression().fit(X_train[numeric_features],y_train)
y_num_pred = model_num.predict(X_test[numeric_features])
y_num_pred_train = model_num.predict(X_train[numeric_features])

mse = mean_squared_error(y_test, y_num_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_num_pred)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

mse_train = mean_squared_error(y_train, y_num_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_num_pred_train)
print(f"Root Mean Squared Error (Train Set): {rmse_train:.2f}")
print(f"R² Score (Train Set): {r2_train:.4f}")
# %%
# categorical variable regression
cat_features = X_train.select_dtypes(exclude=[np.number]).columns
X_train_cat = pd.get_dummies(X_train[cat_features], drop_first=True, prefix=cat_features)
X_test_cat = pd.get_dummies(X_test[cat_features], drop_first=True, prefix=cat_features)

X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0)
# need to re-index because of there may be mismatch in the data- Datsun only had 3 entries

model_cat = LinearRegression().fit(X_train_cat,y_train)
y_cat_pred_test = model_cat.predict(X_test_cat)
y_cat_pred_train = model_cat.predict(X_train_cat)

mse = mean_squared_error(y_test, y_cat_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_cat_pred_test)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

mse_train = mean_squared_error(y_train, y_cat_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_cat_pred_train)
print(f"Root Mean Squared Error (Train Set): {rmse_train:.2f}")
print(f"R² Score (Train Set): {r2_train:.4f}")

# This model using the categorical variables performs better,
# since it has a lower RMSE and a higher R^2 score.
# %%
# combining both numeric and categorical vars
X_train_both = pd.get_dummies(X_train, columns=cat_features, drop_first=True, prefix=cat_features)
X_test_both = pd.get_dummies(X_test, columns=cat_features, drop_first=True, prefix=cat_features)
X_test_both = X_test_both.reindex(columns=X_train_both.columns, fill_value=0)
# same thing, reindex here because of weird categorical columns

model_both = LinearRegression().fit(X_train_both,y_train)
y_pred_test = model_both.predict(X_test_both)
y_pred_train = model_both.predict(X_train_both)

mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)
print(f"Root Mean Squared Error (Test Set): {rmse:.2f}")
print(f"R² Score (Test Set): {r2:.4f}")

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)
print(f"Root Mean Squared Error (Train Set): {rmse_train:.2f}")
print(f"R² Score (Train Set): {r2_train:.4f}")

# The joint model performs a lot better! On the test set, the RMSE
# is 0.07 lower, and the R^2 score is 0.1604 higher. This model
# accounts for 81.39% of the variation in the log of price.
# %%[markdown]
# *************************************************
# Question 5
# *************************************************

# %%
from sklearn.preprocessing import PolynomialFeatures

results = {}
for degree in [1,2,3,4,5,6,7,8,9,10]:
    pf  = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_p_num = pf.fit_transform(X_train[numeric_features])
    X_test_p_num = pf.transform(X_test[numeric_features])
    X_train_poly = np.hstack([X_train_p_num, X_train_cat])
    X_test_poly = np.hstack([X_test_p_num, X_test_cat])

    model_p = LinearRegression().fit(X_train_poly, y_train)
    y_p_test = model_p.predict(X_test_poly)
    y_p_train = model_p.predict(X_train_poly)

    mse_test = mean_squared_error(y_test, y_p_test)
    rmse_test = np.sqrt(mse_train)
    r2  = model_p.score(X_test_poly, y_test)
    results[f'degree_{degree}'] = r2
    print(f"Degree {degree}  |  Test R²: {r2:.4f} | Test RMSE: {rmse_test:.4f}")

# Interestingly, my R^2 never goes negative. Even if I make my model 
# go up to degree 10, the value never goes negative. RMSE also 
# doesn't change. This could mean that there are just simply
# not enough numeric features, or the numeric features do not
# predict price well in the first place, so adding polynomial
# features wouldn't help much. This is supported by the fact that
# the combination model in question 4 of all variables performs much
# better than this polynomial model.
# %%[markdown]
# *************************************************
# Question 6
# *************************************************

# %%
# predicted vs. true vals for num + cat features from problem 4
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5) #y_pred_test is the var name for the best model, only used that time
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values')
plt.show()

# The predicted values rougly line up with the true values along the 
# diagnonal line.
# %%
# residual plot
resid = y_test - y_pred_test
resid.plot.kde(color='pink')
plt.title('Residual Plot')

# It looks like my residuals are relatively bell-shaped centered 
# around zero. This means that my errors are normally distributed
# and therefore my model does not have much bias and it consistently
# predicts pretty accurately.

# %% [markdown]
# # Section Q3- Wine Quality

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
