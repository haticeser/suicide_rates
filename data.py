# Importing necessary libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import sklearn
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np  # linear algebra
#split edip train ettikten sonra 

import os # accessing directory structure
df =  pd.read_csv("./master.csv")

#print(df)

#print(df.info() )  # print the concise summery of the datasetpi

#print(df.columns) #attributeları yazdırıyor.

#print(df.describe()) #min max bulduk dfyı açıkladık.
#--------------
#checking the df for null or missing values

#print(df.isnull().sum())
#---------------------------
#Checking the df type of each column

#print(df.dtypes)
#----------------------------------------------

#bu kısımda bazı sorulara cevap veriyoruz.
## What was the highest number of deaths due to Suicide in a year?
maximum = max(df.suicide_count)
#print(maximum)

## What is the total death-count due to suicide occured over the years 1985 to 2016 ?
total = (df.suicide_count).count()
#print(total)

##What is the average death rate?
average = (df.suicide_count).mean()
#print(average)



#---------------------------------------------------


X = df.drop(['suicide_rate' , 'suicide_count', 'country-year'], axis=1)
plt.scatter(X['gdp_per_capita'], X['gdp_for_year'])
#plt.show()


plt.scatter(X['gdp_per_capita'], X['gdp_for_year'])
#plt.show() 


plt.scatter(X['population'], X['gdp_for_year'])
#plt.show()

plt.scatter(X['gdp_per_capita'], X['hdi_for_year'])
#plt.show()


#print(X.columns.values)# attributeları yazdırıyor.


plt.scatter(X['gdp_for_year'], X['hdi_for_year'])
#plt.show()


# Exclude non-numeric columns before calculating correlation
numeric_columns = X.select_dtypes(include=[np.number]).columns
corr = X[numeric_columns].corr() #It creates a matrix
#print(corr)

#--------------------------------------------------------

# Plot heatmap
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
#plt.show()


#-----------------------------------------




#belli bir kısmını almak için
ftr_albania =(df['country'].str.contains('Albania')&
              df['sex'].str.contains('male') &
              df['age'].str.contains('15-24 years'))
#print(df[ftr_albania][['country','sex','age','suicide_count']])




ftr_russia =(df['country'].str.contains('Russia')&
              df['sex'].str.contains('female') &
              df['age'].str.contains('15-24 years'))
#print(df[ftr_russia][['country','sex','age','suicide_count','year']])


year = df[ftr_russia]['year']
suicides = df[ftr_russia]['suicide_count']

plt.figure(figsize=(10, 8))
plt.plot(year,suicides)
plt.xticks(rotation= 'vertical')
plt.title('Russian 15-24 years old Female Suicides Cases from 1980 to 2015 ',fontsize=18)
plt.xlabel('Year')
plt.ylabel('Suicides Cases')
#plt.show()


#DROP ETME İŞLEMİNİ ANLATIYORUM. KODUN DEVAMINDA BU DROP EDİLMİŞ DATASET ÜZERİNDEN İŞLME YAPACAĞIM.
#print(df.shape)
#(27820, 12) drop yapmadan önce
#------------------------------------------
#dropping the HDI for year column

df = df.drop(['hdi_for_year'], axis = 1)
#print(df.shape)
#print(df.columns) #attribute düşürünce kalan attribute listesi
#(27820, 11) bir attribute düşürünce

#-----------------------------------------------
#dropping the country-year for year column

df = df.drop(['country-year'], axis = 1)
#print(df.shape)
#(27820, 10) bir tane daha düşürünce geriye kaldı 10 attribute
#---------------------------------------------------
#droppinf off any null rows (is any)

df = df.dropna()
#print(df.shape)
#(27820, 10) 

#---------------------------



"""
The non-numerical labeled columns - -> country -> year -> gender -> age_group -> generation

---are to be converted to numerical labels that can be worked upon by using SkLearn's LabelEncoder.
"""

#encoding the categorical features with LabelEncoder
from sklearn.preprocessing import LabelEncoder

categorical = ['country', 'year','age', 'sex', 'generation']
le = sklearn.preprocessing.LabelEncoder()

for column in categorical:
    df[column] = le.fit_transform(df[column])
    #print(le.fit_transform(df["sex"])[0:5])  # [1 0 0 0 1]  (1: male, 0: female)



# Converting the column 'gdp_for_year' to float from object

df['gdp_for_year'] = df['gdp_for_year'].str.replace(',','').astype(float)


#Scaling the numerical df columns with RobustScalar

numerical = ['suicide_count', 'population', 'suicide_rate', 
              'gdp_for_year','gdp_per_capita']


from sklearn.preprocessing import RobustScaler

rc = RobustScaler()
df[numerical] = rc.fit_transform(df[numerical])

#print(df.head) #burda sıradan yazdırıyo







#splitting the df

# Sepratating & assigning features and target columns to X & y

y = df['suicide_rate']
X = df.drop('suicide_rate',axis=1)
#print(X.shape)
#print(y.shape)
"""
(27820, 9)
(27820,)
"""




# Split the data into training and test sets (e.g., 80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#Further split the training data into training and validation sets (e.g., 75% training, 25% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=32)


# Convert DataFrame to numpy array
X_train_array = X_train.values
X_test_array = X_test.values
X_val_array = X_val.values
y_val = y_val.values
y_train = y_train.values
y_test = y_test.values

# Reshape the numpy array to a 2D array [num_samples, num_features]
X_train = X_train_array.reshape(X_train_array.shape[0], X_train_array.shape[1])
X_test = X_test_array.reshape(X_test_array.shape[0], X_test_array.shape[1])
X_val = X_val_array.reshape(X_val_array.shape[0], X_val_array.shape[1])

#print(X_train.shape)
#print(y_train.shape)
#(16692, 1, 9)
#(16692,)

#print(df.head(5)) #ilk 5 değeri  yazdırıyor.


#---------------------------------------------
#Trying Support Vector Regression
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train)
y_pred_test = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
rms_test = np.sqrt(mean_squared_error(y_test , y_pred_test))
rms_train =np.sqrt(mean_squared_error(y_train, y_pred_train))

print('The RMSE of the training set is: '+rms_train.astype(str))
print('The RMSE of the test set is: '+rms_test.astype(str))

#-------------------------------------------------
#Trying Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)
y_pred_test1 = regressor.predict(X_test)
y_pred_train1 = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
rms_test = np.sqrt(mean_squared_error(y_test , y_pred_test1))
rms_train =np.sqrt(mean_squared_error(y_train, y_pred_train1))

#print('The RMSE of the training set is: '+rms_train.astype(str))
#print('The RMSE of the test set is: '+rms_test.astype(str))
#--------------------------------------------------
#Trying Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=30)
regressor.fit(X_train, y_train)
y_pred_test2 = regressor.predict(X_test)
y_pred_train2 = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
rms_test = np.sqrt(mean_squared_error(y_test , y_pred_test2))
rms_train =np.sqrt(mean_squared_error(y_train, y_pred_train2))

#print('The RMSE of the training set is: '+rms_train.astype(str))
#print('The RMSE of the test set is: '+rms_test.astype(str))



#git status
#git add .
#git commit -m "commit-message"
#git push