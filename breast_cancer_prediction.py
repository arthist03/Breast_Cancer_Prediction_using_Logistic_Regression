# Import Lib
import pandas as pd
import seaborn as sb
import os

#Kaggle_Id and Key
os.environ['KAGGLE_USERNAME'] = 'YOUR_USERNAME'
os.environ['KAGGLE_KEY'] = 'KAGGLE_API_KEY'

#Downloading_data
! kaggle datasets download -d uciml/breast-cancer-wisconsin-data

#unzipping data
! unzip /content/breast-cancer-wisconsin-data.zip

#Reading Data
df = pd.read_csv('/content/data.csv')

#Displaying Data
df.head()

#finding No. of Rows and Columns
df.shape

#Finding Null values
df.isna().sum()

#Deleting null values
df.dropna(axis=1, inplace=True)

#Checking, is it is deleted ir not?
df.shape

#finding no. of contents
df['diagnosis'].value_counts()

#Finding Data types
df.dtypes

#Labeling the data from (M,B) to (1,0)
from sklearn.preprocessing import LabelEncoder as LE
labelencoder = LE()
df.iloc[:,1] = labelencoder.fit_transform(df.iloc[:,1].values)

# Displaying all datas
df

#splitting data into independent and dependent variables
X = df.iloc[:,2:].values #Independent values
Y = df.iloc[:,1].values #Dependent Values

# splitting data into testing data and training data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25)

# Make all values into 0 and 1 for the standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#printing data
X_train

#Building Model ("We are using binary Classification or say Logistic classification")
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

#Using trained data => testing X-test data
predictions = classifier.predict(X_test)

# Visualising the error usnig Confusion_metrix lib and Y test data
from sklearn.metrics import confusion_matrix
import seaborn as sb
cm = confusion_matrix(Y_test, predictions)
print(cm)
sb.heatmap(cm, annot=True)

#Checking Acuracy data

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

# Comparing which data is not predicted well
print(Y_test) #Actual value
print(predictions) # trained Value
#Where 1 = M(Malignant) and 0 = B(Benign)