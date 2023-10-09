import pandas as pd
import numpy as np

from sklearn import svm, metrics
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

import researchtoolbox.ml.supervised_learning as svl
from researchtoolbox.utility.os import *

import shap


"""
This module contains classes and methods for transforming a BVP signal in a BPM signal.
"""

class cross_validation_setting_model:
    
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    def __init__(self, n_fold=10, repeat_times=1, add_principal_component_analysis=False, principal_component_ratio=0.85):
        self.add_principal_component_analysis = add_principal_component_analysis
        self.principal_component_ratio = principal_component_ratio
        self.repeat_times = repeat_times
        self.n_fold = n_fold


class cross_validation:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """

    def __init__(self, param_dict):
        self.all_test_x = []
        self.remember = []
        self.predict_y_result = []
        self.total_score = []
        self.total_F1 = []
        self.total_accuracy = []
        self.shap_values_all = []
        self.predict_y_prob = []

        self.model = param_dict.get("model", None)
        self.features = param_dict.get("features", None)
        self.output_file_path = param_dict.get("output_file_path", None)
        # self.save_path_bar = param_dict["save_path_bar"]
        # self.save_path_dot = param_dict["save_path_dot"]
        self.select_important_feature = param_dict.get("select_important_feature", None)
        self.parameter_grid_search = param_dict.get("parameter_grid_search", None)


    def do_cross_validation(self, cv_settings:cross_validation_setting_model,
                            data_model:svl.supervised_learning_data_model, model_random):
        # add_principal_component_analysis = (len(cv_settings.principal_component_ratio)>0)
        #
        # if add_principal_component_analysis:
        #     n_principal_component_list = cv_settings.principal_component_list
        # else:
        #     n_principal_component_list = [0]
        #
        # for index_pca in range(len(n_principal_component_list)):

        if cv_settings.add_principal_component_analysis:  # 进行主成分分析法
            # n_principal_component = n_principal_component_list[index_pca]

            pca = PCA(n_components=cv_settings.principal_component_ratio)
            x_transformed = pca.fit_transform(data_model.x)
            print("Principal Component Number=" + str(x_transformed.shape[1]))

            y = data_model.Y
            x = pd.DataFrame(data=x_transformed)
        else:  # 不进行主成分分析
            y = data_model.Y
            x = data_model.x

        ind = 0
        for repeat_index in range(cv_settings.repeat_times):
            if cv_settings.repeat_times > 1:
                print("Repeat:" + str(repeat_index + 1))
            if cv_settings.n_fold:
                sk = KFold(n_splits=cv_settings.n_fold, random_state=repeat_index, shuffle=True)
            else:
                sk = LeaveOneOut()

            n_cross_validation = 0

            for train_index, val_index in sk.split(x, y):
                n_cross_validation = n_cross_validation + 1
                self.run_each_cross_validation(repeat_index, n_cross_validation,x,y,train_index, val_index,
                                               data_model.remember_no, model_random, cv_settings.n_fold)
                ind += 1

        # 打印平均准确率、平均f1分数
        print("\tavg accuracy:", sum(self.total_accuracy) / len(self.total_accuracy))
        if len(self.total_F1) > 0:
            print("\tavg f1:", sum(self.total_F1) / len(self.total_F1))
            print()

        # 保存预测结果
        temp = np.append(self.predict_y_prob, self.predict_y_result[:, np.newaxis], axis=1)
        df_predict = pd.DataFrame(data=np.append(temp, self.true_y[:, np.newaxis], axis=1),)
                                # columns=["Predict_type_1", "Predict_type_2", "Predict_result", "Truth", '1', '2'])
        df_predict.loc[:, "No-step"] = self.remember
        df_predict.to_csv(self.output_file_path, index=False)

        # do shap analysis
        self.all_test_x = pd.DataFrame(self.all_test_x, columns=self.features)
        shap.initjs()

        # shap.summary_plot(self.shap_values_all, features=self.all_test_x, feature_names=self.features, plot_type="bar", show=True,
        #                 save=True, save_path=self.save_path_bar)

        # shap.summary_plot(self.shap_values_all, features=self.all_test_x, feature_names=self.features, plot_type="dot",
        #                   show=True,
        #                   save=True, save_path=self.save_path_dot)

        shap.summary_plot(self.shap_values_all, features=self.all_test_x, feature_names=self.features, plot_type="bar")
        shap.summary_plot(self.shap_values_all, features=self.all_test_x, feature_names=self.features, plot_type="dot")
        return self.total_accuracy, self.total_F1


    def run_each_cross_validation(self, repeat_index, n_cross_validation, x, y, train_index, val_index, remember_no, model_random, n_fold):
        train_x, test_x = x.iloc[train_index, :], x.iloc[val_index, :]
        train_y, test_y = y.iloc[train_index, :], y.iloc[val_index, :]
        if self.select_important_feature:
            if self.model == "svm":
                sel = SelectFromModel(svm.SVC)
            elif self.model == "rf":
                sel = SelectFromModel(RandomForestClassifier)
            else:
                raise "Error! The value of param_dict['model'] should be in ['rf', 'svm'] "

            sel.fit(train_x, train_y)
            selected_feat = train_x.columns[(sel.get_support())]
            train_x = train_x[selected_feat]
            test_x = test_x[selected_feat]

        if self.parameter_grid_search:
            model_random.fit(train_x, train_y)
            print(model_random.best_params_)
            f = open(os.path.join(Path().check_path_or_create("./best parameters"), "best parameters_{}_{}_{}.txt".format(self.model, repeat_index, n_cross_validation)), "a+", encoding="utf-8")
            f.write(str(model_random.best_params_)+"\n")
            f.close()
            if self.model == "rf":
                model = RandomForestClassifier(bootstrap=model_random.best_params_["bootstrap"],
                                                max_depth=model_random.best_params_["max_depth"],
                                                min_samples_leaf=model_random.best_params_["min_samples_leaf"],
                                                min_samples_split=model_random.best_params_["min_samples_split"],
                                                n_estimators=model_random.best_params_["n_estimators"],
                                                random_state=n_cross_validation)
            elif self.model == "svm":
                model = svm.SVC(C=model_random.best_params_["C"],
                                gamma=model_random.best_params_["gamma"],
                                kernel=model_random.best_params_["kernel"],
                                probability=True,
                                random_state=n_cross_validation
                                )
            else:
                raise "Error! The value of param_dict['model'] should be in ['rf', 'svm'] "
        else:
            if self.model == "rf":
                model = RandomForestClassifier(random_state=n_cross_validation)
            elif self.model == "svm":
                model = svm.SVC(random_state=n_cross_validation, probability=True)
            else:
                raise "Error! The value of param_dict['model'] should be in ['rf', 'svm'] "

        model.fit(train_x, train_y.values.ravel())
        prediction = model.predict(test_x)
        accuracy = metrics.accuracy_score(test_y, prediction)

        if n_cross_validation == 1:
            self.predict_y_prob = model.predict_proba(test_x)
            self.predict_y_result = prediction
            self.true_y = test_y["Step-new"]
            self.remember = remember_no.iloc[val_index, :]
        else:
            self.predict_y_prob = np.append(self.predict_y_prob, model.predict_proba(test_x), axis=0)
            self.predict_y_result = np.append(self.predict_y_result, prediction, axis=0)
            self.true_y = np.append(self.true_y, test_y["Step-new"], axis=0)
            self.remember = pd.concat([self.remember, remember_no.iloc[val_index, :]], axis=0, ignore_index=True)

        # do shap analysis
        explainer = shap.KernelExplainer(model.predict_proba, train_x, link="logit")
        shap_values = explainer.shap_values(test_x, nsamples=1000)
        if n_cross_validation == 1:
            self.all_test_x = test_x
            self.shap_values_all = shap_values[0]
        else:
            self.shap_values_all = np.append(self.shap_values_all, shap_values[0], axis=0)
            self.all_test_x = np.append(self.all_test_x, test_x, axis=0)

        # print(accuracy)
        self.total_accuracy.append(accuracy)
        self.total_score.append(model.score(test_x, test_y))

        if n_fold:
            current_f1 = metrics.f1_score(test_y, prediction, average='macro')  # pos_label=21
            self.total_F1.append(current_f1)
