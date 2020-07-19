import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
x=np.array([[0,0],[1,2],[1/2,3/2],[3/2,1/2]])
y = np.array([-1,1,1,1])
clf = SVC(kernel = 'linear')
clf.fit(x, y)
print(clf)
print(clf.support_vectors_)
print(clf.support_)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')

plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

print(clf.decision_function(x))