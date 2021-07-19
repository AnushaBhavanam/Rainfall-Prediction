import pandas as pd
import pickle
rainfall_data=pd.read_csv("..//Dataset//weatherAUS.csv")
rainfall_data.head()
rainfall_data.shape
rainfall_data.info()
rainfall_data.isna().sum()
rainfall_data.drop('Date',inplace=True,axis=1)
rainfall_data.drop('Location',inplace=True,axis=1)

rainfall_data['RainToday'].value_counts()
rainfall_data['RainTomorrow'].value_counts()
rainfall_data['RainToday'].replace({'No':0,'Yes':1},inplace=True)
rainfall_data['RainTomorrow'].replace({'No':0,'Yes':1},inplace=True)
rainfall_data['RainTomorrow'].value_counts()

from sklearn.utils import resample
no = rainfall_data[rainfall_data.RainTomorrow == 0]# stores all the '0' values in it
yes = rainfall_data[rainfall_data.RainTomorrow == 1]# stores all the '1' values in it

# as we are oversampling the minority class i.e 'yes' values by using the resample()

yes_oversample = resample(yes, replace=True , n_samples=len(no) , random_state = 123)
oversample = pd.concat([yes_oversample , no])
total = oversample.isnull().sum().sort_values(ascending=False)
percent = (oversample.isnull().sum()/oversample.isnull().count()).sort_values(ascending=False)
missing = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing.head(23)
oversample.select_dtypes(include=['object']).columns
oversample['WindGustDir'] = oversample['WindGustDir'].fillna(oversample['WindGustDir'].mode()[0])
oversample['WindDir9am'] = oversample['WindDir9am'].fillna(oversample['WindDir9am'].mode()[0])
oversample['WindDir3pm'] = oversample['WindDir3pm'].fillna(oversample['WindDir3pm'].mode()[0])
## We will convert the categorical features to continuous features with label encoding

## lbr =  LabelEncoder()
## rainfall_data['Date']=lbr.fit_transform(rainfall_data['Date'])

from sklearn.preprocessing import LabelEncoder
lencoders = {}
for col in oversample.select_dtypes(include=['object']).columns:
    lencoders[col] = LabelEncoder()
    oversample[col] = lencoders[col].fit_transform(oversample[col])
    
import warnings
warnings.filterwarnings("ignore")

## Multiple Imputation by Chained Equations

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
MiceImputed = oversample.copy(deep=True)
mice_imputer = IterativeImputer()
MiceImputed.iloc[:, :] = mice_imputer.fit_transform(oversample)
Q1 = MiceImputed.quantile(0.25)
Q3 = MiceImputed.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
## Removing the outliers from the dataset

MiceImputed = MiceImputed[~((MiceImputed<(Q1-1.5*IQR))|(MiceImputed>(Q3+1.5*IQR))).any(axis=1)]

MiceImputed.shape

# Standardizing data
# initialising the minmax scaler function in to r_scaler
# scaling the dataset keeping the columns name

from sklearn import preprocessing
r_scaler = preprocessing.MinMaxScaler()
#r_scaler.fit(MiceImputed)
modified_data = pd.DataFrame(r_scaler.fit_transform(MiceImputed),index=MiceImputed.index,columns=MiceImputed.columns)
x = MiceImputed[['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 
                       'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                       'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 
                       'RainToday']]
y = MiceImputed['RainTomorrow']

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(class_weight="balanced")

lr.fit(x,y)
y_pred =lr.predict(x)
from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred)
with open('model.pkl','wb') as f:
    pickle.dump(lr,f)