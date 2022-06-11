
import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix

df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())

X = df[['variance','skewness','curtosis','entropy']] 
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



LR = LogisticRegression()
LR.fit(X_train,y_train)

y_prediction = LR.predict(X_test)

predicted_values = []

for i in y_prediction:
  if i == 0:
    predicted_values.append("Authorized")
  else:
    predicted_values.append("Forged")

actual_values = []
for i in y_test:
  if i == 0:
    actual_values.append("Authorized")
  else:
    actual_values.append("Forged")


labels = ["Forged", "Authorized"]

cm = confusion_matrix(actual_values, predicted_values)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()