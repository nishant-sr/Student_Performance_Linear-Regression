import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pplot
from matplotlib import style

# save csv file as dataframe
data = pd.read_csv("student-mat.csv", sep=";")

# filter the dataframe to contain observations for only the attributes of our choice
data = data[["G1", "G2", "G3", "studytime", "goout", "absences","traveltime"]]

# label is the final grade
predict = "G3"

# x is the training dataframe so excludes the label
x = np.array(data.drop(predict, 1))
print(x)
# y is the dataframe containing just the label
y = np.array(data[predict])

# splitting dataframe into testing(10% of data) and training data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# linear regression model
linear = linear_model.LinearRegression()
# finds linear regression (best line of fit) for training data x and y
linear.fit(x_train, y_train)
# value representing the accuracy of model
accuracy = linear.score(x_test, y_test)
# print(accuracy)

# prints multivariate coefficients and linear intercept
print("Coeffecients : \n" , linear.coef_)
print("Intercept : \n" , linear.intercept_)

# predict using test data
test_predictions = linear.predict(x_test)
trained_predictions = linear.predict(x_train)

# for x in range(len(test_predictions)):
#     print(test_predictions[x],x_test[x],y_test[x])

# visualizing results
x_var = "traveltime"
style.use("ggplot")
pplot.scatter(data[x_var],data["G3"])
pplot.title("Relation between Students' Commute Time and Final Grade")
pplot.xlabel("Travel Time to school (hours)")
pplot.ylabel("Final grade (#/20)")
pplot.show()