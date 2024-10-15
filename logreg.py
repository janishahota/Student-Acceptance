import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import  seaborn as sns

data = pd.read_csv('/Users/hota/Downloads/StudentData.csv')

X = data[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]
y = data['Chance of Admit ']

print(X.shape)
print(y.shape)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 8))
correlation_matrix = data.iloc[:,2:].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='magma')
plt.title('Correlation Heatmap')
plt.show()

print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

print(model.predict([[320,104,1,2,2.5,9,1]]))
