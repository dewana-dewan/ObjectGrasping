import numpy as np
from sklearn.svm import SVC

X = np.array([[2, 1], [6, 2], [5, 3], [3, 0], [5, 4], [1, 1]])
y = np.array([0, 1, 1, 0, 1, 0])

classifier_conf = SVC(kernel='poly',C = 10, gamma = 'auto', probability=True)
classifier_conf.fit(X, y)
pro1 = classifier_conf.predict_proba(np.array([[1, 3]]))
pro2 = classifier_conf.predict_proba(np.array([[4,2]]))

print (pro1);
print (pro2);
