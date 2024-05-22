import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv("./master.csv")

# Drop columns and prepare target variable
X = data.drop(['suicides/100k pop', 'suicides_no', 'country-year'], axis=1)
y = data['suicides/100k pop']

# Convert 'gdp_for_year' to float by removing commas
X['gdp_for_year'] = X['gdp_for_year'].str.replace(',', '').astype(float)

# Identify numeric and categorical features
numeric_features = ['year', 'population', 'hdi_for_year', 'gdp_for_year', 'gdp_per_capita']
categorical_features = ['country', 'sex', 'age', 'generation']

# Define the numeric transformer pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Define the categorical transformer pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Create the final pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit the pipeline on the data
clf.fit(X, y)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
clf.fit(X_train, y_train)

# Predict using the model
y_pred_test = clf.predict(X_test)
y_pred_train = clf.predict(X_train)

# Evaluate the model
from sklearn.metrics import mean_squared_error

rms_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
rms_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

print('The RMSE of the training set is: ' + str(rms_train))
print('The RMSE of the test set is: ' + str(rms_test))

# Plot heatmap of correlations for numeric features
numeric_columns = X.select_dtypes(include=[np.number]).columns
corr = X[numeric_columns].corr()
sns.heatmap(corr, annot=True, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()

# Optional: example scatter plot after transformation (customize as needed)
plt.scatter(X['gdp_per_capita'], X['gdp_for_year'])
plt.xlabel('GDP per Capita ($)')
plt.ylabel('GDP for Year ($)')
plt.show()
