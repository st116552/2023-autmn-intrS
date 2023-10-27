import numpy
import pandas
import matplotlib.pyplot
import seaborn
from sklearn import tree
import csv

dataf = pandas.read_csv("dataFile (2).csv")
datafXOnly = pandas.read_csv("dataFileXOnly.csv")
#print(dataf.head())
#print(dataf.index.tolist())
#print(dataf.columns.tolist())
x = dataf.iloc[:, 0:1]
y = dataf.iloc[:, 1:2]
newX = datafXOnly.iloc[:, 0:1]

model = tree.DecisionTreeRegressor(max_depth=10)
model.fit(x, y)


print(model.score(x, y))
#print(tree.plot_tree(model))
matplotlib.pyplot.scatter(x, y, color="cornflowerblue")

newY = model.predict(newX)

matplotlib.pyplot.scatter(newX, newY, color="yellowgreen")
matplotlib.pyplot.show()
