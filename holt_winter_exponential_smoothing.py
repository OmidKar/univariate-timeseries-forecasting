# grid search ets models for monthly car sales
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from numpy import array
import itertools

class HoltWinterExponentialSmoothing:
    
    def __init__(self, data, n_test, seasonal=[0, 6, 12], parallel=True):
        self.data = data
        self.n_test = n_test
        self.seasonal = seasonal
        self.parallel = parallel    

    # one-step Holt Winter’s Exponential Smoothing forecast
    def exp_smoothing_forecast(self, history, config):
        t,d,s,p,b,r = config
        # define model
        history = array(history)
        model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
        # fit model
        model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
        # make one step forecast
        yhat = model_fit.predict(len(history), len(history))
        return yhat[0]

    # root mean squared error or rmse
    def measure_rmse(self, actual, predicted):
        return sqrt(mean_squared_error(actual, predicted))

    # split a univariate dataset into train/test sets
    def train_test_split(self):
        return self.data[:-self.n_test], self.data[-self.n_test:]

    # walk-forward validation for univariate data
    def walk_forward_validation(self, cfg):
        predictions = list()
        # split dataset
        train, test = self.train_test_split()
        # seed history with training dataset
        history = [x for x in train]
        #print(history)
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = self.exp_smoothing_forecast(history, cfg)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = self.measure_rmse(test, predictions)
        return error

    # score a model, return None on failure
    def score_model(self, cfg, debug=False):
        result = None
        # convert config to a key
        key = str(cfg)
        # show all warnings and fail on exception if debugging
        if debug:
            result = self.walk_forward_validation(cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = self.walk_forward_validation(cfg)
                    
            except:
                error = None
        # check for an interesting result
        if result is not None:
            print(' > Model[%s] %.3f' % (key, result))
        return (key, result)

    # grid search configs
    def grid_search(self, cfg_list):
        scores = None
        if self.parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(self.score_model)(cfg) for cfg in cfg_list)
            scores = executor(tasks)
        else:
            scores = [self.score_model(scfg) for cfg in cfg_list]
        # remove empty results
        scores = [r for r in scores if r[1] != None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores

    # create a set of exponential smoothing configs to try
    def exp_smoothing_configs(self):
        models = list()
        # define config lists
        t_params, d_params, s_params = ['add', 'mul', None], [True, False], ['add', 'mul', None]
        p_params, b_params, r_params = self.seasonal, [True, False], [True, False]
        # create config instances
        orders = [t_params, d_params, s_params, p_params, b_params, r_params]
        models = list(itertools.product(*orders))
        return models


    def run(self):
        # model configs
        cfg_list = self.exp_smoothing_configs()
        # grid search
        scores = self.grid_search(cfg_list)
        # list top 3 configs
        for cfg, error in scores[:3]:
            print(cfg, error)        
    
if __name__ == '__main__':   
    series = read_csv('monthly_sales.csv', header=0, index_col=0)
    data = series.values

    n_test = 12
    obj = HoltWinterExponentialSmoothing(data, n_test)
    obj.run()