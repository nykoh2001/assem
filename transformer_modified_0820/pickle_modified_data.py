
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

X = []
y = []

path = "../dataset/modified"

for i in range(5, 41, 5):
    percentage_thres = i * 0.4
    for j in range(int(percentage_thres) + 1):
        file = open(f"{path}/{i}_{j}.txt", "r")
        lines = file.readlines()[:120000]
        for l in lines:
            X.append(l.rstrip())
            y.append(l.rstrip())
        file.close()

X_train_total, X_test_total, y_train_total, y_test_total = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=34
)

print(np.array(X).shape)

with open("X_train_m_total.pickle","wb") as fw:
    pickle.dump(X_train_total, fw)

with open("X_test_m_total.pickle","wb") as fw:
    pickle.dump(X_test_total, fw)
    
with open("y_train_m_total.pickle","wb") as fw:
    pickle.dump(y_train_total, fw)

with open("y_test_m_total.pickle","wb") as fw:
    pickle.dump(y_test_total, fw)