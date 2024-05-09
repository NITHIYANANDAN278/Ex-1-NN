<H3> NAME:N.Nithiyanandan</H3> 
<H3> REG NO:212222230099</H3>
<H3>EX. NO.1</H3>
<H3>29/02/2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df
df.isnull().sum()
df.duplicated()
df.describe()
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:
### DATASET:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/501a37bf-23a9-494a-a405-338c2dd7795c)
### DROPPING THE UNWANTED DATASET:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/ea2af879-9249-48cf-95ac-9e81c97111ca)
### CHECKING NULL VALUES:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/049d838f-369e-4797-9343-110941640828)
### CHECKING FOR DUPLICATION:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/78d5e594-2fef-42b5-8f15-a94ad364912b)
### DESCRIBING THE DATASET:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/8d224a54-ea26-4d53-84d6-75e8aecf1cdb)
### SCALING THE DATASET:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/4d990b2a-d948-485f-828f-e76cc6d0a9d6)
### X FEATURES:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/6134cc36-30af-4210-9fca-3307177c9a05)
### Y FEATURES:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/f1f9d7dc-73a8-42f0-b000-43fe0713c330)
### SPLITTING THE TRAINING AND TESTING DATASET:
![image](https://github.com/NITHIYANANDAN278/Ex-1-NN/assets/121784636/ea18beb2-8cf0-4ebd-b229-36e037da149c)










## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


