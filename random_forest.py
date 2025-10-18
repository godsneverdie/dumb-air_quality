from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
f=r'drop_col_combined.csv'
d=pd.read_csv(f,engine='python')
features=d.drop('label',axis=1)
labels=d['label']
train_feat,test_feat, train_lab,test_lab=train_test_split(
    features,
    labels,
    test_size=0.1,
    stratify=labels,
    random_state=42
)
rf=RandomForestClassifier(n_estimators=400,random_state=42,class_weight='balanced')
rf.fit(train_feat,train_lab)
pred=rf.predict(test_feat)
print(f"Accuracy-------->{accuracy_score(test_lab,pred)}")
c =confusion_matrix(test_lab,pred)
disp=ConfusionMatrixDisplay(confusion_matrix=c)
disp.plot()
plt.show()
