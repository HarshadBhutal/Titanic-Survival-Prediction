import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("Titanic_dataset.csv")


df.drop(["Cabin","Name","Ticket"],axis=1,inplace=True)

sns.countplot(x="Survived",data=df)
plt.show()

# print(df.isnull().sum()) // Displays how many missing values each column has.
 
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"]=LabelEncoder().fit_transform(df["Sex"])
df=pd.get_dummies(df,columns=["Embarked"],drop_first=True)
df=pd.get_dummies(df,columns=["AgeGroup"],drop_first=True)

x=df.drop(columns=["Survived"])
y=df["Survived"]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=43)

scaler=StandardScaler()
x_train_norm=scaler.fit_transform(x_train)
x_test_norm=scaler.transform(x_test)

model=LogisticRegression(max_iter=100)
model.fit(x_train_norm,y_train)

y_pred=model.predict(x_test_norm)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("roc auc ",roc_auc_score(y_test,y_pred))

