import numpy
import pandas
import matplotlib.pyplot
from sklearn import tree
from sklearn.model_selection import train_test_split


def mse(y, y_predicted):
    return numpy.square(y - y_predicted) / len(y)


def mae(y, y_predicted):
    return numpy.absolute(y - y_predicted) / len(y)


def count_metrics(y, y_predicted):
    return [mse(y, y_predicted), mae(y, y_predicted)]


def draw_graph(x_arrays, y_arrays, names):
    line_colors = ['b', 'g', 'r', 'c', 'm', 'y']
    i = 0
    while i < len(x_arrays) and i < len(y_arrays) and i < len(names) and i < len(line_colors):
        line_width = 1 + i
        matplotlib.pyplot.plot(x_arrays[i], y_arrays[i], label=names[i], color=line_colors[i], linewidth=line_width)
        i += 1
    matplotlib.pyplot.xlabel("y, m")
    matplotlib.pyplot.ylabel('v, m/s')
    matplotlib.pyplot.grid()
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


def calc_decision_tree_regression(exp_data_file_name, data_file_name):
    data_f = pandas.read_csv(data_file_name)
    exp_data_f = pandas.read_csv(exp_data_file_name)
    data_f.head()
    exp_data_f.head()
    x = data_f['y']
    y = data_f['V']
    x_exp = exp_data_f['y']
    y_exp = exp_data_f['V']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = tree.DecisionTreeRegressor()
    model.fit(x_train, y_train)

    y_train_predicted = model.predict(x_train)
    y_test_predicted = model.predict(x_test)
    y_exp_predicted = model.predict(x_exp)

    metrics_train = count_metrics(y_train, y_train_predicted)
    metrics_test = count_metrics(y_test, y_test_predicted)
    metrics_exp = count_metrics(y_exp, y_exp_predicted)

    print('mse ans mae')
    print('train', metrics_train)
    print('test', metrics_test)
    print('exp', metrics_exp)
    draw_graph([x_train, x_test, x_exp], [y_train, y_test_predicted, y_exp], ['Dataset', 'Predicts', 'Experiment'])



#calc_decision_tree_regression('couetteRe1200.csv', 'dataRe1200.csv')
calc_decision_tree_regression('poiseuilleDataRe1200.csv', 'resultsRe1200.csv')

