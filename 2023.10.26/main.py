import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split


def mse(y, y_predicted):
    return numpy.mean(numpy.square(y - y_predicted))


def mae(y, y_predicted):
    return numpy.mean(numpy.absolute(y - y_predicted))


def count_metrics(y, y_predicted):
    return {'MSE': mse(y, y_predicted),
            'MAE': mae(y, y_predicted)}


def draw_graph(x_arrays, y_arrays, names):
    line_colors = ['b', 'y', 'c', 'g', 'r']
    # i = 0

    for i in range(len(x_arrays)):
        plt.plot(x_arrays[i], y_arrays[i], 'o', label=names[i], color=line_colors[i], linewidth=0.5)
    plt.xlabel("y, m")
    plt.ylabel('v, m/s')
    plt.grid()
    plt.legend()
    plt.show()


def calc_decision_tree_regression(exp_data_file_name, data_file_name):
    data_f = pandas.read_csv(data_file_name)
    exp_data_f = pandas.read_csv(exp_data_file_name)
    data_f.head()
    exp_data_f.head()
    x = data_f['y'].to_numpy().reshape((-1, 1))
    y = data_f['V']
    x_exp = exp_data_f['y'].to_numpy().reshape((-1, 1))
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
    print(f'Train: MSE = {metrics_train["MSE"]}, MAE = {metrics_train["MAE"]}')
    print(f'Test: MSE = {metrics_test["MSE"]}, MAE = {metrics_test["MAE"]}')
    print(f'Experiment: MSE = {metrics_exp["MSE"]}, MAE = {metrics_exp["MAE"]}')

    draw_graph([x_train, x_test, x_exp], [y_train, y_test_predicted, y_exp],
               ['Dataset', 'Predicts', 'Experiment'])


calc_decision_tree_regression('dataRe1200.csv', 'couetteRe1200.csv')
calc_decision_tree_regression('poiseuilleDataRe1200.csv', 'resultsRe1200.csv')
