import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Numpy is a package in Python used for Scientific Computing
# matplotlib.pyplot is a plotting library used for 2D graphics
# Pandas is the most popular python library that is used for data analysis.
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("./master.csv")
X = data.drop(['suicides/100k pop' , 'suicides_no', 'country-year'], axis=1)
#delete some "dublicate" features
y=data['suicides/100k pop']


# Exclude non-numeric columns before calculating correlation
numeric_columns = X.select_dtypes(include=[np.number]).columns
corr = X[numeric_columns].corr() #It creates a matrix
print(corr)

""" 
                        year  population  HDI for year  gdp_per_capita ($)
year                1.000000    0.008850      0.366786            0.339134
population          0.008850    1.000000      0.102943            0.081510
HDI for year        0.366786    0.102943      1.000000            0.771228
gdp_per_capita ($)  0.339134    0.081510      0.771228            1.000000

"""


# Plot heatmap
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()


#plt.scatter(X['gdp_per_capita($)'], X['gdp_for_year($) '])


#---------------------------------------------------
#data.info()   # print the concise summery of the datasetpi
'''RangeIndex: 27820 entries, 0 to 27819 
Data columns (total 12 columns):      
 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   country             27820 non-null  object
 1   year                27820 non-null  int64
 2   sex                 27820 non-null  object
 3   age                 27820 non-null  object
 4   suicides_no         27820 non-null  int64
 5   population          27820 non-null  int64
 6   suicides/100k pop   27820 non-null  float64
 7   country-year        27820 non-null  object
 8   HDI for year        8364 non-null   float64
 9    gdp_for_year ($)   27820 non-null  object
 10  gdp_per_capita ($)  27820 non-null  int64
 11  generation          27820 non-null  object
dtypes: float64(2), int64(4), object(6)
memory usage: 2.5+ MB
'''
#------------------------------------------------------

#data.head(2) # look at 1st 5 data points

#data.describe()
""" 
plt.scatter(X['gdp_per_capita'], X['gdp_for_year'])
plt.show() 


plt.scatter(X['population'], X['gdp_for_year'])
plt.show()

plt.scatter(X['gdp_per_capita'], X['HDI for'])
plt.show()


print(X.columns.values)# attributeları yazdırıyor.

X = X[Y< 125]  Bunu dağiştirelim. Y ve X eksenlerini düzenlemek gerek.
Y = Y[Y<125]
plt.scatter(X['gdp_for_year'], X['hdi_for_year'])
plt.show()
"""



#Data Preprocessing


X['gdp_for_year'] = X['gdp_for_year'].str.replace(',','').astype(float)
#We replace the commas from the values for the data to be converted as float.


numeric_features = ['year' , 'hdi_for_year' , 'gdp_for_year', 'population', 'gdp_per_capita']
categorical_features = ['country' , 'sex' , 'age', 'generation']


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
 #these all appear to come because HDI wasn't available prior to 2

numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(missing_values = 'NaN', strategy= 'mean')), ('scaler', StandardScaler())])

#categorical_transtormer = Pipeline(steps=[('onehot',OneHotEncoder())])
# Define the categorical transformer pipeline
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer ( transformers= [('num', numeric_transformer,numeric_features), ('cat', categorical_transformer ,categorical_features )])

clf = Pipeline(steps=[('preprocessor', preprocessor) ])
X = clf. fit_transform(X)
print(X)

#git status
#git add .
#git commit -m "commit-message"
#git push


#Splitting the Dataset


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=False)
regressor.fit(X_train , y_train)

LinearRegression( copy_X=True, fit_intercept=False, n_jobs=None, normalize = False)

y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
rms_test = np.sqrt(mean_squared_error(y_test,y_pred_test))
rms_train = np.sqrt( mean_squared_error(y_train,y_pred_train))

print('The RMSE of the training set is: '+ rms_train.astype(str))
print('The RMSE of the test set is: '+ rms_test.astype(str))
