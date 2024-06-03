import io

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve
from functions import make_model, feature_extraction, \
    metrics_evaluation, detailed_metrics, classification_methods, \
    compare_models, data_import, single_prediction  # Assuming these are correctly defined in functions.py
from sklearn.exceptions import ConvergenceWarning
import warnings
import os
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors

warnings.filterwarnings("ignore")

warnings.simplefilter('ignore', category=ConvergenceWarning)
st.set_page_config(page_title="ML Classifier Dashboard", layout="wide")

# Define a custom theme
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }
    .reportview-container {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #A8B5E0;
    }
    h1 {
        color: #03396c;
    }
    .stButton>button {
        color: #000000;
        border-radius: 10px;
        border: 2px solid #A8B5E0;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

st.title('Advanced ML Classifier Dashboard')

model_dict = {
    'Logistic Regression': 0, 'K-Nearest Neighbors': 1, 'Support Vector Classifier': 2,
    'Random Forest': 3, 'Gradient Boosting': 4, 'Decision Tree': 5, 'MLPClassifier': 6,
    'LinearSVC': 7, 'GaussianProcess': 8, 'AdaBoostClassifier': 9,
    'QuadraticDiscriminantAnalysis': 10, 'XGBoost': 11
}

feature_data_types = {
    'current_education_status_bachelor': bool, 'smokes_conventional': bool, 'health_status_good': bool,
    'monit_is_succesful': bool, 'Robinson': float, 'SBP': int, 'Me(SBP)': int, 'SBP1/Me(SBP)': float,
    'DBP': int, 'DBP1/Me(DBP)': float, 'phys_labor': bool, 'Me(HR)': int, 'smokes_e-cigarettes': bool,
    'sleep_mode_satisfaction_mosdef_no': bool, 'alcohol_status_rarely': bool, 'gender_of_pat': bool,
    'age_of_pat': int, 'sleep_mode_satisfaction_rather_yes': bool, 'diagnosis_healthy': bool,
    'BP_avg': float, 'Me(DBP)': int, 'health_status_excellent': bool, 'health_status_unsatisfactory': bool,
    'food_frequency_vary': bool, 'is_alert': bool, 'HR': int, 'current_education_status_highschool': bool,
    'bmi_normal': bool, 'out_of_plan': bool, 'diagnosis_masked_hypertension': bool, 'food_quality_50%': bool,
    'current_marital_status_widow': bool, 'sleep_mode_satisfaction_mosdef_yes': bool, 'has_chronic_diseases': bool,
    'height': int, 'working_under_stress': bool, 'food_frequency_only_1_time': bool, 'food_quality_100%': bool,
    'food_quality_70%': bool, 'before_midnight': bool, 'news_status_regularly': bool
}


def load_data(data_type):
    database = pd.read_excel(os.path.join(f'{data_type}test_data.xlsx'))
    return database


if 'model_runs' not in st.session_state:
    st.session_state.model_runs = []

shap_plot_type = None
# Sidebar configuration
data_type = st.sidebar.selectbox('Select Data Type', ['cog', 'phys'], index=0)
column_type = st.sidebar.selectbox('Select Target Class',
                                   ['HR_class', 'SBP_class', 'DBP_class', 'stress_class', 'Additive_class',
                                    'Multiplicity_class'], index=0)
algorithm_index = st.sidebar.radio("Choose an Algorithm",
                                   ['Logistic Regression', 'K-Nearest Neighbors', 'Support Vector Classifier',
                                    'Random Forest', 'Gradient Boosting', 'Decision Tree', 'MLPClassifier', 'LinearSVC',
                                    'GaussianProcess', 'AdaBoostClassifier', 'QuadraticDiscriminantAnalysis',
                                    'XGBoost'], index=2)

model_id = model_dict[algorithm_index]
model, params = classification_methods(model_id)

set_type = st.sidebar.selectbox("Choose Set Type", ['Train', 'Test'], index=1)
test_size = st.sidebar.slider('Test Set Size', min_value=0.1, max_value=0.5, value=0.15, step=0.05)
shap_plot = st.sidebar.checkbox('Enable SHAP value importance plot', False)
if shap_plot:
    shap_plot_type = st.sidebar.selectbox('Select SHAP Plot Type', ['Summary Plot', 'Bar Plot', 'Waterfall Plot'])

if shap_plot_type == 'Waterfall Plot':
    x_train, _, _, _, _, _ = data_import(data_type, column_type, test_size)
    user_index = st.sidebar.slider('Select instance index', min_value=0, max_value=len(x_train) - 1, step=1)
enable_hyperparameter_tuning = st.sidebar.checkbox('Enable Hyperparameter Tuning', False)

if enable_hyperparameter_tuning:
    for param, values in params.items():
        selected_value = st.sidebar.selectbox(f"Select {param}", values)
        params[param] = selected_value

show_confusion_matrix = st.sidebar.checkbox("Show Confusion Matrix", False)
show_classification_results = st.sidebar.checkbox("Show Classification Results", False)
show_detailed_metrics = st.sidebar.checkbox("Show Detailed Metrics", False)
correlation_matrix = st.sidebar.checkbox("Show Correlation Matrix", False)
if correlation_matrix:
    corr_type = st.sidebar.radio("Select Correlation Type", ["Pearson", "Spearman", "Kendall"])
user_input = st.sidebar.checkbox('Enter Your Input', False)
# save_prediction = st.sidebar.checkbox('Show and Save Prediction for Specific Patient', False)
# if save_prediction:
#     _, _, _, _, _, database = data_import(data_type, column_type, test_size)
#     number = st.sidebar.number_input('Enter Patient Number', min_value=0, max_value=len(database) - 1, step=1, key='patient_number')


if len(st.session_state.model_runs) > 1:
    compare_model = st.sidebar.checkbox("Compare Models", disabled=False)
else:
    compare_model = st.sidebar.checkbox("Compare Models", disabled=True)

if shap_plot:
    save_prediction = st.sidebar.checkbox('Show and Save Prediction for Specific Patient', disabled=False)

else:
    save_prediction = st.sidebar.checkbox('Show and Save Prediction for Specific Patient', disabled=True)

if save_prediction:
    if (shap_plot_type == 'Summary Plot') or (shap_plot_type == 'Bar Plot'):
        _, _, _, _, _, database = data_import(data_type, column_type, test_size)
        number = st.sidebar.number_input('Enter Patient Number', min_value=0, max_value=len(database) - 1, step=1,
                                         key='patient_number')


run_model = st.sidebar.button('Run Model')

with st.sidebar.expander("Save to PDF Options", expanded=False):
    save_confusion_matrix = st.checkbox("Save Confusion Matrix", False)
    save_classification_results = st.checkbox("Save Classification Results", False)
    save_detailed_metrics = st.checkbox("Save Detailed Metrics", False)
    save_correlation_matrix = st.checkbox("Save Correlation Matrix", False)
    save_comparing_table = st.checkbox("Save Comparing Table", False)
    # save_shap_plot = st.sidebar.checkbox("Save Shap Plot", False)

# Button to generate PDF
generate_pdf = st.sidebar.button('Generate PDF')


def save_figure_to_session_state(fig, key):
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    st.session_state[key] = buf


def save_results_to_pdf(components, filename="results.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    margin = 50
    current_height = height - margin

    def check_add_new_page(required_height):
        nonlocal current_height
        if current_height < required_height:
            c.showPage()
            current_height = height - margin

    for component in components:
        if isinstance(component, str):
            # Handle text
            check_add_new_page(20)
            c.drawString(100, current_height, component)
            current_height -= 20
        elif isinstance(component, io.BytesIO):
            # Handle figures saved as byte streams
            component.seek(0)  # Go to the beginning of the BytesIO buffer
            img = ImageReader(component)
            img_width, img_height = img.getSize()
            aspect_ratio = img_width / img_height
            new_width = width - 2 * margin
            new_height = new_width / aspect_ratio
            center_x = (width - new_width) / 2
            check_add_new_page(new_height + 10)  # Ensure space for image and some padding
            c.drawImage(img, center_x, current_height - new_height, width=new_width, height=new_height,
                        preserveAspectRatio=True, mask='auto')
            current_height -= (new_height + 10)
        elif isinstance(component, pd.DataFrame):
            # Handle DataFrames
            check_add_new_page(50)
            data = [component.columns.tolist()] + component.values.tolist()

            # Create a Table object
            table = Table(data)

            # Add style to the table
            style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ])
            table.setStyle(style)

            # Draw the table on the canvas
            table.wrapOn(c, width - 2 * margin, height - 2 * margin)
            table_width, table_height = table.wrap(0, 0)
            x = (width - table_width) / 2
            y = current_height - table_height
            table.drawOn(c, x, y)
            current_height -= (table_height + 10)

    c.save()


if run_model:

    if enable_hyperparameter_tuning:
        best_model, x_train, x_test, y_train, y_test, selected_columns, database = make_model(
                model_dict[algorithm_index], column_type, data_type, True, test_size, params)
    else:
        best_model, x_train, x_test, y_train, y_test, selected_columns, database = make_model(
            model_dict[algorithm_index], column_type, data_type, False, test_size, params)

    st.session_state.model_run = True

    if set_type == 'Train':
        x_eval, y_eval = x_train, y_train
    else:
        x_eval, y_eval = x_test, y_test

    st.session_state.model_runs.append((algorithm_index, best_model, x_eval, y_eval, selected_columns, database))

    if shap_plot_type == 'Waterfall Plot':
        st.header(f'SHAP {shap_plot_type}')
        fig = feature_extraction(best_model, x_train, selected_columns, algorithm_index, column_type,
                                 plot_type=shap_plot_type.lower().replace(" ", "").replace("plot", ""),
                                 user_index=user_index)
        st.pyplot(fig)
        save_figure_to_session_state(fig, 'SHAP plot')
    elif (shap_plot_type == 'Summary Plot') or (shap_plot_type == 'Bar Plot'):
        st.header(f'SHAP {shap_plot_type}')
        fig = feature_extraction(best_model, x_train, selected_columns, algorithm_index, column_type,
                                 plot_type=shap_plot_type.lower().replace(" ", "").replace("plot", ""),
                                 user_index=None)
        st.pyplot(fig)
        save_figure_to_session_state(fig, 'SHAP plot')

    # Confusion Matrix
    if show_confusion_matrix:
        st.header(f'Confusion Matrix for {set_type} set')
        if set_type == 'Train':
            cm = confusion_matrix(y_train, best_model.predict(x_train))
        else:
            cm = confusion_matrix(y_test, best_model.predict(x_test))
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
        save_figure_to_session_state(fig, 'confusion_matrix')

    # Classification Results
    if show_classification_results:
        st.header(f'Classification Results for {set_type} set')
        if set_type == 'Train':
            accuracy_score, precision_score, recall_score, f1_score_score, roc_auc, specificity, mcc = metrics_evaluation(best_model, x_train,
                                                                                               y_train)
        else:
            accuracy_score, precision_score, recall_score, f1_score_score, roc_auc, specificity, mcc = metrics_evaluation(best_model, x_test,
                                                                                               y_test)
        st.metric(label="Accuracy", value=f"{accuracy_score:.3f}")
        st.metric(label="Precision", value=f"{precision_score:.3f}")
        st.metric(label="Recall", value=f"{recall_score:.3f}")
        st.metric(label="F1 Score", value=f"{f1_score_score:.3f}")
        st.metric(label="ROC AUC Score", value=f"{roc_auc:.3f}")
        st.metric(label="Specificity", value=f"{specificity:.3f}")
        st.metric(label="Matthews Correlation Coefficient", value=f"{mcc:.3f}")

        results = {
            "Accuracy": f"{accuracy_score:.3f}",
            "Precision": f"{precision_score:.3f}",
            "Recall": f"{recall_score:.3f}",
            "F1 Score": f"{f1_score_score:.3f}",
            "ROC AUC": f"{roc_auc:.3f}",
            "Specificity": f"{specificity:.3f}",
            "MCC": f"{mcc:.3f}"
        }
        df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
        st.session_state['classification_results'] = df

    # ROC and Precision-Recall Curve
    if show_detailed_metrics:
        st.header(f'ROC and Precision-Recall Curve for {set_type} set')
        if set_type == 'Train':
            fpr, tpr, roc_auc, precision, recall = detailed_metrics(best_model, x_train, y_train)
        else:
            fpr, tpr, roc_auc, precision, recall = detailed_metrics(best_model, x_test, y_test)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax[0].set_xlabel('False Positive Rate')
        ax[0].set_ylabel('True Positive Rate')
        ax[0].set_title('Receiver Operating Characteristic')
        ax[0].legend(loc="lower right")

        ax[1].plot(recall, precision, color='green', lw=2)
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('Precision')
        ax[1].set_title('Precision-Recall Curve')

        st.pyplot(fig)
        save_figure_to_session_state(fig, 'roc_pr_curves')

    if correlation_matrix:
        st.header('Correlation Matrix')
        data = load_data(data_type)
        data = data[selected_columns]
        corr_matrix = data.corr(method=corr_type.lower())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        st.pyplot(fig)
        save_figure_to_session_state(fig, 'correlation_matrix')

    if compare_model:
        st.header('Model Comparison')
        model_comparison_results = compare_models(st.session_state.model_runs)
        st.table(model_comparison_results)

        st.session_state['model_comparison_results'] = model_comparison_results

    if save_prediction and shap_plot:
        components = []
        prediction_results = None
        st.header('Patient Prediction and Saving')
        if (shap_plot_type == 'Summary Plot') or (shap_plot_type == 'Bar Plot'):
            prediction_results, data_based = single_prediction(st.session_state.model_runs, number)
        elif shap_plot_type == 'Waterfall Plot':
            prediction_results, data_based = single_prediction(st.session_state.model_runs, user_index)

        st.session_state['patient_prediction'] = prediction_results
        st.session_state['data_based'] = data_based
        st.table(st.session_state['patient_prediction'])

        components.append("SHAP value importance plot:")
        components.append(st.session_state['SHAP plot'])
        components.append(" ")
        components.append("Used Data for Patient Prediction:")
        components.append(st.session_state['data_based'])
        components.append(" ")
        components.append("Patient prediction:")
        components.append(st.session_state['patient_prediction'])
        save_results_to_pdf(components, 'patient_prediction.pdf')

if user_input:
    best_model, x_train, x_test, y_train, y_test, selected_columns, database = make_model(
        model_dict[algorithm_index], column_type, data_type, False, test_size, params)
    st.header("User Input for Prediction")

    with st.form(key='user_input_form') as form:
        inputs = {}
        for feature in selected_columns:
            expected_type = feature_data_types.get(feature, str)
            label = f"Enter {feature}"
            if expected_type == bool:
                inputs[feature] = st.radio(label, [True, False], key=feature)
            elif expected_type == int:
                inputs[feature] = st.number_input(label, step=1, key=feature)
            elif expected_type == float:
                inputs[feature] = st.number_input(label, step=0.01, key=feature)
            else:
                inputs[feature] = st.text_input(label, key=feature)

        submitted = st.form_submit_button(label='Submit')

    if submitted:
        validated_data = []
        for feature in selected_columns:
            value = inputs[feature]
            try:
                if feature_data_types[feature] == bool:
                    validated_data.append(value)
                elif feature_data_types[feature] == int:
                    validated_data.append(int(value))
                elif feature_data_types[feature] == float:
                    validated_data.append(float(value))
                else:
                    validated_data.append(value)  # Default fallback, though not expected
            except ValueError as e:
                st.error(f"Invalid input for {feature}. Please enter a valid {feature_data_types[feature].__name__}.")

        input_data = np.array([validated_data])  # Correctly structured as a 2D array
        input_df = pd.DataFrame(input_data, columns=selected_columns)

        try:
            prediction = best_model.predict(input_df)
            if prediction[0] == 1:
                st.success(f"Prediction: {prediction[0]} -  you have a __sharp increase__ in indicators, you might be "
                           f"__under stress__")
            else:
                st.error(f"Prediction: {prediction[0]} -  your indicators are __normal__, you are __not under "
                         f"stress__")

        except Exception as e:
            st.error(f"An error occurred: {e}")


if not run_model and not user_input:
    st.markdown("""
    ### Welcome to the Advanced ML Classifier Dashboard for Stress Prediction
    This interactive dashboard lets you configure, run, and analyze various machine learning classification models.
    Use the sidebar to customize your analysis settings and press 'Run Model' to generate results.
    """, unsafe_allow_html=True)
    with st.expander("Click here for the __Guide to use the program__", expanded=False):
        st.markdown("""
        ### Guide to Using the Advanced ML Classifier Dashboard
        - **Choose your data and target class**: Select the type of data and the specific class you are interested in.
        - **Select an algorithm**: Choose from a variety of algorithms available.
        - **Select a set type**: Choose set type to perform algorithm results evaluation.
        - **Select SHAP plot type**: Choose SHAP plot type to perform algorithm results evaluation.
        - **Enable Hyperparameter Tuning**: Choose your own hyperparameters for classification algorithm.
        - **View additional results**: Optionally, toggle the visibility of confusion matrices, classification results, 
        detailed metrics, correlation matrix.
        - **Enter your own data**: If you want to predict stress state based on input values, choose the option.
        - **Run the model**: Execute the model to see outputs including SHAP plots and performance metrics.
        - **Compare models**: Compare the models performance after running program at least two times.
        - **See the prediction for specific patient and save results to pdf**: You can see the prediction results for 
        specific patient by entering the number and save results along with SHAP plot.
        - **Save to PDF**: Choose the desired options to save into the PDF file. 
        """, unsafe_allow_html=True)

if generate_pdf:
    components_to_save = []

    # Check if the user wants to save the confusion matrix
    if save_confusion_matrix and 'confusion_matrix' in st.session_state:
        components_to_save.append("Confusion Matrix:")
        components_to_save.append(st.session_state['confusion_matrix'])

    # Check if the user wants to save classification results
    if save_classification_results and 'classification_results' in st.session_state:
        components_to_save.append(" ")
        components_to_save.append("Classification Results:")
        components_to_save.append(st.session_state['classification_results'])

    # Check for detailed metrics (ROC, Precision-Recall Curves)
    if save_detailed_metrics and 'roc_pr_curves' in st.session_state:
        components_to_save.append(" ")
        components_to_save.append("ROC AUC Curve and Precision-Recall curve:")
        components_to_save.append(st.session_state['roc_pr_curves'])

    # Check for correlation matrix
    if save_correlation_matrix and 'correlation_matrix' in st.session_state:
        components_to_save.append(" ")
        components_to_save.append("Correlation Matrix:")
        components_to_save.append(st.session_state['correlation_matrix'])

    # Check for model comparison table
    if save_comparing_table and 'model_comparison_results' in st.session_state:
        components_to_save.append(" ")
        components_to_save.append("Model Comparison Results:")
        components_to_save.append(st.session_state['model_comparison_results'])

    if not components_to_save:
        st.sidebar.error("No components to save!")
    else:
        # Generate the PDF with the selected components
        save_results_to_pdf(components_to_save, filename="analysis_results.pdf")
        st.sidebar.success("PDF generated successfully!")
