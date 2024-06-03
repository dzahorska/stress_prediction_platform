import os
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter('ignore', category=ConvergenceWarning)


def classification_methods(model_type):
    model = None
    params = None
    if model_type == 0:
        model = LogisticRegression(random_state=42)
        params = {'penalty': ['l1', 'l2', None],
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                  }
    elif model_type == 1:
        model = KNeighborsClassifier()
        params = {'n_neighbors': [3, 5, 7, 9],
                  'weights': ['uniform', 'distance']}
    elif model_type == 2:
        model = SVC(probability=True, random_state=42)
        params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'gamma': [0.001, 0.01, 0.1, 1],
                  'kernel': ['linear', 'rbf']}
    elif model_type == 3:
        model = RandomForestClassifier(random_state=42)
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    elif model_type == 4:
        model = GradientBoostingClassifier(random_state=42)
        params = {
            'n_estimators': [50, 100, 150, 200, 250, 300],
            'learning_rate': [0.01, 0.1, 0.5],
            'max_depth': [3, 5, 7, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_type == 5:
        model = DecisionTreeClassifier(random_state=42)
        params = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 3, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 5, 10, 15, 20],
            'max_features': ['sqrt', 'log2']
        }
    elif model_type == 6:
        model = MLPClassifier(random_state=42)
        params = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (20, 20), (20,)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'max_iter': [1100, 1200, 1500, 2000, 2500, 3000, 5000],
            'solver': ['adam', 'lbfgs', 'sgd']
        }
    elif model_type == 7:
        model = LinearSVC(random_state=42)
        params = {
            'loss': ['hinge', 'squared_hinge'],
            'C': [1e-5, 1e-4, 0.1, 1, 10],
            'dual': [True],
            'max_iter': [1100, 1200, 1500, 2000]
        }
    elif model_type == 8:
        model = GaussianProcessClassifier(kernel=1.0 * RBF(1.0), random_state=42)
        params = {
            'max_iter_predict': [50, 100, 150, 200],
            'n_restarts_optimizer': [0, 1, 2]
        }
    elif model_type == 9:
        model = AdaBoostClassifier(random_state=42)
        params = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.5]
        }
    elif model_type == 10:
        model = QuadraticDiscriminantAnalysis()
        params = {
            'reg_param': [0.00001, 0.0001, 0.001,0.01, 0.1],
            'store_covariance': [True, False],
            'tol': [0.0001, 0.001, 0.01, 0.1]
        }
    elif model_type == 11:
        model = XGBClassifier()
        params = {'max_depth': range(2, 15),
                  'n_estimators': range(60, 220, 20),
                  'learning_rate': [0.1, 0.01, 0.001, 0.05],
                  "min_child_weight": [1, 2, 4, 5, 10],
                  'subsample': [0.5, 0.7, 1]
                  }
    return model, params


def data_import(data_type, column_type, test_size=0.15):
    scaler = StandardScaler()
    database = pd.read_excel(os.path.join(f'{data_type}test_data.xlsx'))
    y = database[column_type]
    x_train, x_test, y_train, y_test = train_test_split(database, y, test_size=test_size, random_state=42, stratify=y)

    x_train = get_x_data(column_type, x_train, data_type)
    x_test = get_x_data(column_type, x_test, data_type)

    selected_columns = x_train.columns

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test, selected_columns, database


def make_model(model_type, column_type, data, enable_hyperparameter_tuning, test_size, parameters):

    x_train, x_test, y_train, y_test, selected_columns, database = data_import(data, column_type, test_size)

    if enable_hyperparameter_tuning:
        model, _ = classification_methods(model_type)
        model.set_params(**parameters)
        model.fit(x_train, y_train)
        best_model = model
    else:
        model, params = classification_methods(model_type)
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        best_model.fit(x_train, y_train)

    return best_model, x_train, x_test, y_train, y_test, selected_columns, database


def metrics_evaluation(model, x, y):
    predictions = model.predict(x)
    accuracy_score = metrics.accuracy_score(y, predictions)
    precision_score = metrics.precision_score(y, predictions, average='macro', zero_division=0)
    recall_score = metrics.recall_score(y, predictions, average='macro', zero_division=0)
    f1_score_score = metrics.f1_score(y, predictions, average='macro', zero_division=0)

    _, _, roc_auc, _, _ = detailed_metrics(model, x, y)
    tn, fp, fn, tp = metrics.confusion_matrix(y, predictions).ravel()
    specificity = tn / (tn + fp)
    mcc = metrics.matthews_corrcoef(y, predictions)

    return accuracy_score, precision_score, recall_score, f1_score_score, roc_auc, specificity, mcc


def detailed_metrics(model, x, y):
    if hasattr(model, 'predict_proba'):
        decision_scores = model.predict_proba(x)[:, 1]
    else:
        decision_scores = model.decision_function(x)

    fpr, tpr, _ = metrics.roc_curve(y, decision_scores)
    roc_auc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(y, decision_scores)
    return fpr, tpr, roc_auc, precision, recall


def roc_curve_plot(algorithms_data, column_t):
    plt.figure(figsize=(16, 12))
    for algorithm_name, array_roc in algorithms_data:
        fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test = array_roc
        plt.plot(fpr_train, tpr_train, label=f'{algorithm_name} (Train - AUC = {roc_auc_train:.2f})')

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Different Algorithms (Train) {column_t}')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(16, 12))
    for algorithm_name, array_roc in algorithms_data:
        fpr_train, tpr_train, roc_auc_train, fpr_test, tpr_test, roc_auc_test = array_roc
        plt.plot(fpr_test, tpr_test, label=f'{algorithm_name} (Test - AUC = {roc_auc_test:.2f})')

    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for Different Algorithms (Test) {column_t}')
    plt.legend(loc='lower right')
    plt.show()


def feature_extraction(best_model, x_train, selected_features,algorithm, col, plot_type, user_index=None):
    explainer = shap.KernelExplainer(best_model.predict, x_train)
    shap_values = explainer.shap_values(x_train)
    fig = None
    if not isinstance(selected_features, list):
        selected_features = list(selected_features)

    if plot_type == 'summary':
        plt.title(f'SHAP Summary Plot for {algorithm} - {col}')
        shap.summary_plot(shap_values, x_train, feature_names=selected_features)
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(12, 4)
        ax.tick_params(axis='both', which='major', labelsize=10)
    elif plot_type == 'bar':
        plt.title(f'SHAP Bar Plot for {algorithm} - {col}')
        shap.summary_plot(shap_values, x_train, plot_type='bar', feature_names=selected_features)
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(12, 4)
        ax.tick_params(axis='both', which='major', labelsize=10)
    elif plot_type == 'waterfall':
        plt.title(f'SHAP Bar Plot for {algorithm} - {col}')
        explainer_waterfall = shap.Explainer(best_model.predict, x_train, feature_names=selected_features)
        shap_values_waterfall = explainer_waterfall(x_train)
        shap.plots.waterfall(shap_values_waterfall[user_index])
        fig = plt.gcf()
        ax = plt.gca()
        fig.set_size_inches(20, 10)
        ax.tick_params(axis='both', which='major', labelsize=10)

    return fig


def compare_models(model_runs):
    results = []
    for model_name, model, x_eval, y_eval, selected_columns,_ in model_runs:
        accuracy_score, precision_score, recall_score, f1_score, roc_auc, sp, mcc = metrics_evaluation(model, x_eval, y_eval)
        results.append((model_name, round(accuracy_score, 3), round(precision_score, 3), round(recall_score, 3), round(f1_score, 3), round(roc_auc, 3), round(sp, 3), round(mcc, 3)))
    return pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity', 'MCC'])


def single_prediction(model_runs, ind):
    results = []
    pred_prob = 0
    df_data = None
    indicator = None
    for model_name, model, _, _, selected_columns, database in model_runs:
        results = []
        temp_data = database.loc[:, selected_columns]  # Use .loc to ensure the DataFrame structure is intact
        sample = temp_data.iloc[[ind]]
        prediction = model.predict(sample)
        if hasattr(model, 'predict_proba'):
            if prediction[0] == 1:
                pred_prob = model.predict_proba(sample)[:, 1][0]
                indicator = 'Under stress'
            elif prediction[0] == 0:
                pred_prob = model.predict_proba(sample)[:, 0][0]
                indicator = 'Not under stress'
        else:
            decision_score = model.decision_function(sample)[0]
            probabilities_class_1 = 1 / (1 + np.exp(-decision_score))
            probabilities_class_0 = 1 - pred_prob
            if prediction[0] == 1:
                pred_prob = probabilities_class_1
                indicator = 'Under stress'
            elif prediction[0] == 0:
                pred_prob = probabilities_class_0
                indicator = 'Not under stress'

        rounded_prob = round(float(pred_prob), 2)
        results.append((ind, model_name, indicator, rounded_prob))
        df_data = sample.transpose()
        df_data.reset_index(inplace=True)
        df_data.columns = ['Feature', 'Value']
    df = pd.DataFrame(results, columns=['Patient id', 'Model', 'Prediction', 'Probability'])
    return df, df_data


def confusion_matrix_plot(cm, column_type, data_type):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {data_type} set {column_type}')
    plt.show()


def get_x_data(column_type, data, type):
    cognitive_columns = {
        'HR_class': ['current_education_status_bachelor', 'smokes_conventional', 'health_status_good',
                     'monit_is_succesful', 'Robinson'],
        'SBP_class': ['SBP', 'Me(SBP)', 'SBP1/Me(SBP)'],
        'DBP_class': ['DBP', 'DBP1/Me(DBP)', 'phys_labor'],
        'Additive_class': ['Me(HR)', 'smokes_e-cigarettes', 'sleep_mode_satisfaction_mosdef_no',
                         'alcohol_status_rarely', 'gender_of_pat', 'SBP1/Me(SBP)', 'DBP', 'age_of_pat',
                         'sleep_mode_satisfaction_rather_yes', 'diagnosis_healthy'],
        'Multiplicity_class': ['BP_avg', 'Me(DBP)', 'DBP1/Me(DBP)'],
        'stress_class': ['health_status_excellent', 'health_status_unsatisfactory', 'DBP1/Me(DBP)',
                         'food_frequency_vary', 'DBP', 'Me(DBP)', 'is_alert', 'sleep_mode_satisfaction_rather_yes']
    }

    physical_columns = {
        'HR_class': ['HR', 'alcohol_status_rarely', 'current_education_status_highschool', 'bmi_normal',
                     'out_of_plan', 'sleep_mode_satisfaction_rather_yes', 'diagnosis_masked_hypertension',
                     'Me(HR)', 'food_quality_50%', 'current_marital_status_widow', 'sleep_mode_satisfaction_mosdef_yes'],
        'SBP_class': ['SBP', 'has_chronic_diseases', 'SBP1/Me(SBP)', 'height', 'working_under_stress'],
        'DBP_class': ['DBP', 'DBP1/Me(DBP)', 'Me(SBP)'],
        'Additive_class': ['SBP', 'DBP1/Me(DBP)', 'HR', 'monit_is_succesful', 'food_frequency_only_1_time',
                           'food_quality_100%', 'Robinson', 'food_quality_70%', 'before_midnight'],
        'Multiplicity_class': ['SBP', 'DBP1/Me(DBP)', 'news_status_regularly', 'smokes_conventional'],
        'stress_class': ['SBP', 'DBP1/Me(DBP)', 'Me(HR)']

    }

    if type == 'cog':
        return data[cognitive_columns[column_type]]
    elif type == 'phys':
        return data[physical_columns[column_type]]
    else:
        return "Not found"
