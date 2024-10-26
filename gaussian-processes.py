from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import mean_squared_error
import pandas as pd

def main():
    # Load datasets
    train_data = pd.read_csv('updated_medical_train.csv')
    test_data = pd.read_csv('updated_medical_test.csv')

    X_train = train_data.drop(columns=['Medical Cost'])
    X_train = X_train.replace({True: 1, False: 0})
    y_train = train_data['Medical Cost']

    X_test = test_data.drop(columns=['Medical Cost'])
    X_test = X_test.replace({True: 1, False: 0})
    y_test = test_data['Medical Cost']
    mean, var = model(X_train, y_train, X_test, y_test)
    test_data = test_data.assign(Mean = mean)
    test_data = test_data.assign(Uncertainty=var)
    test_data.to_csv("gaussian_processes_medical_test.csv", index=False)
    

def model(X_train, Y_train, x_test, y_test):
    X_train = X_train.to_numpy()
    x_test = x_test.to_numpy()
    Y_train = Y_train.to_numpy()

    # RBF kernel
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

    # Create the Gaussian Process Regressor model
    gp = GaussianProcessRegressor(kernel=kernel, alpha = 1e-2)

    gp.fit(X_train, Y_train)

    # make predictions
    y_pred, sigma = gp.predict(x_test, return_std=True)

    mse = mean_squared_error(y_test, y_pred)
    print(mse)

    return y_pred, sigma



if __name__ == '__main__':
  main()
