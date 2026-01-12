from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score

def _fit_and_score(estimator, X, y, params, cv, scoring):
    model = estimator.set_params(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "params": params,
        "mean_score": scores.mean(),
        "std_score": scores.std()
    }

class ParallelExecutor:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def run(self, estimator, X, y, param_list, cv, scoring):
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_and_score)(estimator, X, y, params, cv, scoring)
            for params in param_list
        )
        return results
