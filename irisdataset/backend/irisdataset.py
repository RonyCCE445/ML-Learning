import pandas as pd
import pickle
from sklearn.datasets import load_iris
from  sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,precision_score,recall_score,f1_score

#load Dataset
data= load_iris()
df= pd.DataFrame(data.data, columns=data.feature_names)
df['target']= data.target

#train_test_split
x= df.drop('target', axis=1)
y= df['target']
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2,random_state=42)

#training the model
model= RandomForestClassifier()
model.fit(x_train,y_train)

#evaluation

y_pred= model.predict(x_test)
conf_matrix= confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("Accuracy:", accuracy_score(y_test,y_pred))

# Saving the model as .pkl

with open("iris_model.pkl", "wb") as f:
    pickle.dump((model, data.target_names),f)

print("model saved as iris_model.pkl")