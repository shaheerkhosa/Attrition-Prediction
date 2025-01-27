from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from pathlib import Path
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import json

variable_names = [
        "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
        "DistanceFromHome", "Education", "EducationField",
        "EnvironmentSatisfaction", "Gender", "HourlyRate",
        "JobInvolvement", "JobLevel", "JobRole", "JobSatisfaction",
        "MaritalStatus", "MonthlyIncome", "MonthlyRate",
        "NumCompaniesWorked", "OverTime", "PercentSalaryHike",
        "PerformanceRating", "RelationshipSatisfaction",
        "StockOptionLevel", "TotalWorkingYears",
        "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
        "YearsInCurrentRole", "YearsSinceLastPromotion",
        "YearsWithCurrManager"
    ]

# Utility functions
def jsonify_input_from_string(input_string):
    values = input_string.strip().split(',')
    if len(values) != len(variable_names):
        raise ValueError(f"Expected {len(variable_names)} values, got {len(values)}")
    input_dict = {var: value for var, value in zip(variable_names, values)}
    input_dict.pop("Attrition", None)
    return json.dumps(input_dict, indent=4)

def load_dataset(filepath):
    return pd.read_csv(filepath)

def balance_data(data, target_col='Attrition'):
    features = data.drop(columns=[target_col])
    target = data[target_col]

    encoders = {}
    for col in features.select_dtypes(include='object').columns:
        encoder = LabelEncoder()
        features[col] = encoder.fit_transform(features[col])
        encoders[col] = encoder

    if target.dtypes == 'object':
        encoder = LabelEncoder()
        target = encoder.fit_transform(target)
        encoders[target_col] = encoder

    oversampler = SMOTE(sampling_strategy=0.85, random_state=42)
    features, target = oversampler.fit_resample(features, target)
    return features, target, encoders

def normalize_and_standardize_features(x_train, x_test, numerical_features):
    scaler = StandardScaler()
    x_train[numerical_features] = scaler.fit_transform(x_train[numerical_features])
    x_test[numerical_features] = scaler.transform(x_test[numerical_features])
    return x_train, x_test, scaler

def predict_user_input(user_input, model, encoders, scaler, numerical_features, categorical_features, use_json=True):
    try:
        if use_json:
            input_df = pd.DataFrame([json.loads(user_input)])
        else:
            input_df = pd.DataFrame([user_input])

        for col, encoder in encoders.items():
            if col in categorical_features:
                input_df[col] = input_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else None)
                if input_df[col].isnull().any():
                    raise ValueError(f"Unknown category in '{col}': {user_input[col]}")

        input_df[numerical_features] = scaler.transform(input_df[numerical_features])
        prediction = model.predict(input_df)
        return "Yes" if prediction[0] == 1 else "No"

    except Exception as e:
        return f"Error processing input: {str(e)}"

def display_evaluation_metrics(model, x_test, y_test):
    st.write("### Model Evaluation Metrics")
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write(f"**Precision:** {report['weighted avg']['precision']:.2f}")
    st.write(f"**Recall:** {report['weighted avg']['recall']:.2f}")
    st.write(f"**F1-Score:** {report['weighted avg']['f1-score']:.2f}")
    st.write(f"**Accuracy:** {report['accuracy']:.2f}")

    cm = confusion_matrix(y_test, y_pred)
    st.table(pd.DataFrame(cm, columns=["Predicted No", "Predicted Yes"], index=["Actual No", "Actual Yes"]))

    fig, ax = plt.subplots()
    mat = ax.matshow(cm, cmap="coolwarm")
    fig.colorbar(mat)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center', color="black")
    st.pyplot(fig)

    y_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    plt.legend(loc="lower right")
    st.pyplot(fig)

def plot_attrition_visualizations(data, title):
    plt.figure(figsize=(17, 6))
    plt.subplot(1, 2, 1)
    attrition_rate = data["Attrition"].value_counts()
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette=["#1d7874", "#8B0000"])
    plt.title(f"{title} - Employee Attrition Counts", fontweight="black", size=20, pad=20)
    for i, v in enumerate(attrition_rate.values):
        plt.text(i, v, v, ha="center", fontweight='black', fontsize=18)
    
    plt.subplot(1, 2, 2)
    plt.pie(
        attrition_rate,
        labels=["No", "Yes"],
        autopct="%.2f%%",
        textprops={"fontweight": "black", "size": 15},
        colors=["#1d7874", "#AC1F29"],
        explode=[0, 0.1],
        startangle=90,
    )
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    plt.title(f"{title} - Employee Attrition Rate", fontweight="black", size=20, pad=10)

    st.pyplot(plt)
    plt.close()
  
model_options = {
        "Random Forest": RandomForestClassifier(max_depth=4, random_state=0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Neural Network": MLPClassifier(random_state=42, max_iter=1000),
        "CatBoost": CatBoostClassifier(random_state=42, verbose=0),
    }
  
def save_models(x_train, y_train):
    
    if not (Path('./models').exists()):
        try:
            Path('./models').mkdir()
            print(f"Directory '{Path('./models')}' created successfully.")
        except FileExistsError:
            print(f"Directory '{Path('./models')}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{Path('./models')}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    for model_name, model_instance in model_options.items():
        model_path = Path(f'./models/{model_name}.pkl')
        if model_path.exists():
            print(f"Model already saved: {model_name}")
        else:
            print(f"Training and saving: {model_name}")
            model_instance.fit(x_train, y_train)
            joblib.dump(model_instance, model_path)
            print(f"Model saved: {model_name} at {model_path}")
            
    return model_options

def count_files_in_folder(folder_path):
    try:
        folder = Path(folder_path)
        # Use the `.iterdir()` method to list all items and filter files
        files = [item for item in folder.iterdir() if item.is_file()]
        return len(files)
    except FileNotFoundError:
        return "The folder does not exist."
    except Exception as e:
        return f"An error occurred: {e}"

def main():
    
    dataset_path = Path(__file__).parent / "IBM-HR-Analytics-Employee-Attrition-and-Performance-Revised.csv"
    data = load_dataset(dataset_path)
    
    target_col = 'Attrition'
    categorical_features = [col for col in data.select_dtypes(include='object').columns if col != target_col]
    numerical_features = [col for col in data.select_dtypes(include='number').columns if col != target_col]
    
    features, target, encoders = balance_data(data, target_col=target_col)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=42)
    x_train, x_test, scaler = normalize_and_standardize_features(x_train, x_test, numerical_features)
    
    models_path = './models'
    no_models_saved = count_files_in_folder(models_path)
    
    if no_models_saved != len(model_options):
        print("saving models")
        save_models(x_train, y_train)
    
    
    st.title("Employee Attrition Prediction")
    
    section = st.sidebar.radio("Choose Section:", ["How It Works", "Attributes", "Prediction"])

    if section == "How It Works":
        st.subheader("How It Works")
        
        #visualizations before preprocessing
        st.write("### Visualization: Before Preprocessing")
        plot_attrition_visualizations(data, "Before Preprocessing")

        #preprocess data
        balanced_data = pd.concat([features, pd.Series(target, name='Attrition')], axis=1)
        
        #visualizations after preprocessing
        st.write("### Visualization: After Preprocessing")
        plot_attrition_visualizations(balanced_data, "After Preprocessing")

        st.write("### Dataset Overview")
        st.write("Below is a preview of the dataset before preprocessing:")
        st.dataframe(data.head())

        st.write("""
        ### Preprocessing Steps
        1. **Handling Class Imbalance**: The target column ('Attrition') was highly imbalanced. We used SMOTE (Synthetic Minority Oversampling Technique) to generate synthetic samples, achieving a balance of 85% in the minority class.
        2. **Label Encoding**: Categorical variables were converted into numeric codes using Label Encoding.
        3. **Feature Scaling**: Numerical features were standardized to improve model performance.
        """)

        features, target, _ = balance_data(data, target_col=target_col)
        st.write("### After Preprocessing")
        processed_data = pd.concat([features, pd.Series(target, name=target_col)], axis=1)
        st.dataframe(processed_data.head())

    elif section == "Prediction":
        st.subheader("Prediction")

        selected_model_name = st.selectbox("Select a Model:", list(model_options.keys()))

        selected_model = joblib.load(f'./{models_path}/{selected_model_name}.pkl')
        
        st.write(f"### Using {selected_model_name}")
        display_evaluation_metrics(selected_model, x_test, y_test)

        input_mode = st.radio("Select Input Mode:", ["GUI", "JSON"])
        
        if input_mode == "GUI":
            st.subheader("Enter Employee Data using GUI")
            user_input_gui = {}
            
            for col in variable_names:
                if col in categorical_features:
                    options = encoders[col].classes_
                    user_input_gui[col] = st.selectbox(f"Select {col}:", options)
                if col in numerical_features:
                    min_val = int(data[col].min())
                    max_val = int(data[col].max())
                    user_input_gui[col] = st.slider(f"Select {col}:", min_val, max_val, step=1)

            if st.button("Predict with GUI"):
                prediction = predict_user_input(user_input_gui, selected_model, encoders, scaler, numerical_features, categorical_features, use_json=False)
                st.markdown(f"**Predicted Attrition: {prediction}**")

        else:
            st.subheader("Enter Employee Data using JSON/String")
            option = st.radio("Input as:", ["String", "JSON"])
            
            if option == "String":
                user_input_string = st.text_area("Enter comma-separated values:")
                if st.button("Generate and Predict"):
                    try:
                        user_input_json = jsonify_input_from_string(user_input_string)
                        st.subheader("Generated JSON:")
                        st.code(user_input_json, language="json")
                        prediction = predict_user_input(user_input_json, selected_model, encoders, scaler, numerical_features, categorical_features)
                        st.markdown(f"**Predicted Attrition: {prediction}**")
                    except ValueError as e:
                        st.error(f"Error: {e}")
            else:
                user_input_json = st.text_area("Enter JSON here:")
                if st.button("Predict with JSON"):
                    prediction = predict_user_input(user_input_json, selected_model, encoders, scaler, numerical_features, categorical_features)
                    st.markdown(f"**Predicted Attrition: {prediction}**")
    elif section == "Attributes":
        st.subheader("Attribute Insights")
        st.write("### Explore Attributes in the Raw Dataset")
        oldData = load_dataset(dataset_path)
        
        attribute = st.selectbox("Select an Attribute to Visualize:", categorical_features + numerical_features)

        if attribute:
            st.write(f"### Distribution of {attribute} by Attrition")
            fig, ax = plt.subplots(figsize=(10, 5))

            if attribute in categorical_features:
                sns.countplot(data=oldData, x=attribute, hue=target_col, ax=ax, palette="coolwarm")
                ax.set_title(f"{attribute} Distribution by Attrition (Raw Dataset)")
            else:
                sns.kdeplot(data=oldData, x=attribute, hue=target_col, ax=ax, fill=True, common_norm=False, palette="coolwarm")
                ax.set_title(f"{attribute} Density by Attrition (Raw Dataset)")

            st.pyplot(fig)


if __name__ == "__main__":
    main()
