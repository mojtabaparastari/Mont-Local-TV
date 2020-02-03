import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
import math
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier;
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso
import pandas as pd
from sklearn.metrics import confusion_matrix,silhouette_samples, silhouette_score
import datetime
import matplotlib.cm as cm
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import HuberRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor



data = pd.read_csv("D:\class\data science\montral predict/data.csv")
test = pd.read_csv("D:\class\data science\montral predict/test (1).csv")
datasets = [data,test]

print(data.shape)
print(data.dtypes)
print(data.columns)
print(data.isna().sum())
print(data.head())

columns = data.columns

def impute_Starttime(dataset):
    na_list_StartTime = dataset.index[dataset['Start_time'].isna()]
    for i in na_list_StartTime:
        dataset.loc[i, ["Start_time"]] = dataset.iloc[i - 1]["End_time"]


def impute_Endtime(dataset):
    na_list_EndTime = dataset.index[dataset['End_time'].isna()]
    for i in na_list_EndTime:
        dataset.loc[i,["End_time"]] = dataset.iloc[i+1]["Start_time"]



for dataset in datasets:
    print(dataset["Season"].unique())
    dataset["Temperature in Montreal during episode"].fillna(dataset.groupby("Season")["Temperature in Montreal during episode"].transform("mean"), inplace=True)
    print(dataset.isna().sum())
    dataset['Start_time'] = dataset['Start_time'].agg(pd.Timestamp)
    dataset['End_time'] = dataset['End_time'].agg(pd.Timestamp)

datasets = [data , test]

for dataset in [data,test]:
    impute_Starttime(dataset)
    impute_Endtime(dataset)

test["Start_time"].fillna(test.iloc[1]["Start_time"],inplace=True)
test["End_time"].fillna(test.iloc[1]["End_time"],inplace=True)

datasets = [data , test]
for dataset in datasets:
    dataset["duration"] = dataset['End_time'] - dataset['Start_time']
data = data.dropna(subset=['Start_time', 'End_time',"duration"])


datasets = [data , test]




### divide time
for dataset in datasets:
    dataset["Start_time"] = pd.to_datetime(dataset["Start_time"]).dt.time
    dataset["End_time"] = pd.to_datetime(dataset["End_time"]).dt.time


datasets = [data , test]

## drop some unimportant variables:

data = data.drop(["Name of episode"],axis=1)
data=data.drop(["Unnamed: 0"],axis = 1)
test = test.drop(["Name of episode"],axis=1)
test=test.drop(["Unnamed: 0"],axis = 1)






datasets = [data , test]
channel_maping = {'General Channel':0 , 'Specialty Channel':1}
for dataset in datasets:
    dataset["Channel Type"] = dataset["Channel Type"].map(channel_maping)



## we should encode data set for models

enc = LabelEncoder()
for i in ["Episode","Station","Channel Type","Season","Day of week","Name of show","End_time","Start_time","Date"
          ,"duration","Genre","First time or rerun","Movie?","Game of the Canadiens during episode?",'# of episode in the season']:
    enc.fit(data[i])
    data[i]=enc.fit_transform(data[i])
for i in ["Episode","Station","Channel Type","Season","Day of week","Name of show","End_time","Start_time","Date"
          ,"duration","Genre","First time or rerun","Movie?","Game of the Canadiens during episode?",'# of episode in the season']:
    enc.fit(test[i])
    test[i]=enc.fit_transform(test[i])



 #### Delete outlier by using box plot and mean and 6sigma
data["Market Share_total"].agg(np.mean)
data["Market Share_total"].agg(np.std)
data["Market Share_total"].plot.box()
outlier = data.index[data['Market Share_total']>18]
data = data.drop(index = outlier)

## create response variable and independent variable:
X = data.drop(["Market Share_total"],axis = 1)
y = data["Market Share_total"]

## train and test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## some methods
huber = HuberRegressor(fit_intercept=True, alpha=0.9, max_iter=100, epsilon=1.2)
ridge = Ridge(fit_intercept=True, alpha=0.1, random_state=0, normalize=True)
linearreg = LinearRegression()
detr = DecisionTreeRegressor(max_depth=100)
boostr = AdaBoostRegressor()

### select best method:
for method in [huber,ridge,linearreg,detr,boostr]:
    method.fit(X_train,y_train)
    print(str(method),method.score(X_train,y_train))
    
    
## findung the best depth for method
for i in range(1,200,5):
    detr = DecisionTreeRegressor(max_depth=i)
    detr.fit(X_train, y_train)
    yhattrain = detr.predict(X_train)
    yhattest = detr.predict(X_test)
    mae_train = mean_absolute_error(y_train, yhattrain)
    mae_test = mean_absolute_error(y_test, yhattest)
    R_squre_train = r2_score(y_train, yhattrain)
    R_squre_test = r2_score(y_test, yhattest)
    print("max_depth:",i , "R_squre_train:", R_squre_train,"R_squre_test:",R_squre_test )
## fit model
detr = DecisionTreeRegressor(max_depth=16)
detr.fit(X_train, y_train)
yhattrain = detr.predict(X_train)
yhattest = detr.predict(X_test)
mae_train = mean_absolute_error(y_train, yhattrain)
mae_test = mean_absolute_error(y_test, yhattest)
R_squre_train = r2_score(y_train, yhattrain)
R_squre_test = r2_score(y_test, yhattest)

test["market_share_predict"] = detr.predict(test)







