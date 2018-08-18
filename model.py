import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import model_from_json
from keras.utils import np_utils
import statistics
from sklearn.model_selection import GridSearchCV


x = np.load('x_input.npy')
y = np.load('y.npy')
train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)


model1 = RandomForestClassifier(random_state=14)
model2 = LogisticRegression()
model3 = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model4 = model_from_json(loaded_model_json)
# load weights into new model
model4.load_weights("model.h5")
print("Loaded model")

model5 = SVC()
model6 = neighbors.KNeighborsClassifier(n_neighbors=41, n_jobs=1)


parameter_space = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}
grid = GridSearchCV(model1, parameter_space)
grid.fit(train_x, train_y)
model2.fit(train_x,train_y)
model3.fit(train_x,train_y)
model5.fit(train_x,train_y)
model6.fit(train_x,train_y)


pred1 = grid.predict(test_x)
pred2 = model2.predict(test_x)
pred3 = model3.predict(test_x)
pred5 = model5.predict(test_x)
pred6 = model6.predict(test_x)


predictions = model4.predict(test_x)
rounded = [round(x[1]) for x in predictions]
pred4 = np.array([])
for i in range(0,len(rounded)):
	pred4 = np.append(pred4,rounded[i])


final_pred = np.array([])
for i in range(0,len(test_x)):
    final_pred = np.append(final_pred, statistics.mode([pred1[i],pred3[i],pred4[i],pred5[i],pred6[i]]))


 

correct = final_pred == test_y
accuracy = (np.sum(correct) / len(correct))*100
print(accuracy)






