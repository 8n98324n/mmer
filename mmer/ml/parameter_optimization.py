import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

class AutoOptimize:
    def __init__(self):
        pass

    # n_principal_component_list = [int(x) for x in np.linspace(3, 15, num=13)]
    def model_preparation(self, method):
        n_parameter_grid_search_iteration = 100
        parameter_search_fold = 5
        my_random_state = 42
        n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=3)]  # default=100
        max_depth = [int(x) for x in np.linspace(2, 12, num=6)]
        max_depth.append(None)
        min_samples_split = [int(x) for x in np.linspace(2, 22, num=6)]
        min_samples_leaf = [int(x) for x in np.linspace(2, 12, num=6)]
        bootstrap = [True, False]
        svc_grid = {'C': [x for x in np.linspace(0.1, 10, num=100)],
                    'gamma': [x for x in np.linspace(0.001, 0.1, num=100)],
                    'kernel': ['rbf', 'poly', 'sigmoid']}

        random_grid = {'n_estimators': n_estimators,  # 'max_features': max_features,
                       'max_depth': max_depth, 'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

        if method == "rf":
            forest_model = RandomForestClassifier(random_state=my_random_state)
            rf_random = RandomizedSearchCV(estimator=forest_model, param_distributions=random_grid,
                                           n_iter=n_parameter_grid_search_iteration, cv=parameter_search_fold,
                                           verbose=2,
                                           random_state=my_random_state)
            return rf_random
        elif method == "svm":
            svc_model = svm.SVC(random_state=my_random_state)
            svc_random = RandomizedSearchCV(estimator=svc_model, param_distributions=svc_grid,
                                            n_iter=n_parameter_grid_search_iteration, cv=parameter_search_fold,
                                            verbose=2,
                                            random_state=my_random_state)
            return svc_random
        else:
            raise "Error! The value of param_dict['model'] should be in ['rf', 'svm'] "
        # return rf_random, svc_random