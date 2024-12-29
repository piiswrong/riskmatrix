# fit_para = {
#     "num_boost_round": 20000,
#     "early_stopping_rounds": 2000,
#     "verbose_eval":  log_period
# }
# xgb_para = {
#     "tree_method": "hist",
#     "device": "cuda:0",
#     "nthread": os.cpu_count(),
#     "objective": "reg:squarederror",
# #   "objective": "reg:pseudohubererror",
#     "max_depth": 7,
# #   "max_depth": 4,
#     "subsample": 0.8,
# #   "subsample": 0.5,
#     "colsample_bytree": 0.1,
# #   "colsample_bytree": 0.5,
# #   "min_child_weight": 0.5,
# #   "min_child_weight": 200,
#     "reg_alpha": 0.98,
#     "reg_lambda": 0.98,
#     "eval_metric": "rmse",
#     "seed": int(time.time()),
#     "num_parallel_tree": 7,
#     "learning_rate": 1E-2,
#     "verbosity": 1
# }

import xgboost as xgb
from sklearn.metrics import r2_score
import numpy as np
import time
import joblib

class xgb_model:
    def __init__(self, x_train = None, y_train = None,
                    x_val = None, y_val = None,
                    xgb_para = {}, **fit_para):
        self.pkg = xgb
        self.dtrain = xgb.DMatrix(x_train, label = y_train)
        self.dval = xgb.DMatrix(x_val, label = y_val)
        self.para = xgb_para
        self.fit_para = fit_para
        self.fit_para["evals"] = [
            (self.dtrain, "train"), (self.dval, "val")
        ]
        self.fit_para["custom_metric"] = lambda x, y: (
            "mr2", -r2_score(y.get_label(), x)
        )
    def train(self):
        start = time.time()
        self.model = xgb.train(self.para, self.dtrain, **self.fit_para)
        print("Time: %.0fs" % (time.time() - start))
        num_iter = (
            len(self.model.get_dump()) // self.para["num_parallel_tree"]
            if "num_parallel_tree" in self.para
            else len(self.model.get_dump())
        )
        print(f"Total Iterations: {num_iter}")
        print(
            "Best Iteration: %d"
            % (self.model.best_iteration
                if hasattr(self.model, "best_iteration") else num_iter)
        )
        return self
    def _predict(self, data):
        predict = par(
            self.model.predict,
            iteration_range = (0, self.model.best_iteration + 1)
        ) if hasattr(self.model, "best_iteration") else self.model.predict
        return F(data, xgb.DMatrix, predict)
    @property
    def FI(self):
        return self.model.get_fscore()

    def save(self, name):
        joblib.dump(self.model, name)
        return self
    def load(self, name):
        self.model = joblib.load(name)
        return self
    def plotFI(self, **para):
        self.pkg.plot_importance(self.model, **para)
        return self
    def plotTree(self, num = 1, **para):
        if self.pkg is xgb:
            self.pkg.plot_tree(self.model, num_trees = num, **para)
        else:
            self.pkg.plot_tree(self.model, **para)
        return self
    def predict(self, data, num_iter = None):
        if num_iter is None:
            return self._predict(data)
        else:
            pos, output = 0, []
            while pos < data.shape[0]:
                output.append(self._predict(data[pos : pos + num_iter]))
                pos += num_iter
            return np.concatenate(output)