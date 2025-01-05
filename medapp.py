import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset with st.cache_data (for caching data)
@st.cache_data
def load_data():
    data = pd.read_csv("E:\\job related docs\\Feyann LAB\\MedvisionAIapp\\heart_disease_uci.csv")  # Replace with the actual dataset path
    return data

data = load_data()

# Handle Missing Values
# Fill missing numerical values with mean (for simplicity)
data['trestbps'] = data['trestbps'].fillna(data['trestbps'].mean())
data['chol'] = data['chol'].fillna(data['chol'].mean())
data['thalch'] = data['thalch'].fillna(data['thalch'].mean())
data['oldpeak'] = data['oldpeak'].fillna(data['oldpeak'].mean())
data['ca'] = data['ca'].fillna(data['ca'].mean())

# Handle categorical columns with mode (most frequent value)
data['fbs'] = data['fbs'].fillna(data['fbs'].mode()[0])
data['restecg'] = data['restecg'].fillna(data['restecg'].mode()[0])
data['exang'] = data['exang'].fillna(data['exang'].mode()[0])
data['slope'] = data['slope'].fillna(data['slope'].mode()[0])
data['thal'] = data['thal'].fillna(data['thal'].mode()[0])

# Title and dataset overview
st.title("MedVisionAI: Heart Disease Prediction")
st.write("### Dataset Overview")
st.write(data.head())

# Data description
if st.checkbox("Show dataset info"):
    st.write(data.info())

if st.checkbox("Show summary statistics"):
    st.write(data.describe())

# Data visualization
st.write("### Data Visualization")
feature = st.selectbox("Select a feature to visualize", data.columns)
fig, ax = plt.subplots()
sns.histplot(data[feature], kde=True, ax=ax)
st.pyplot(fig)

# Data Preprocessing
X = data.drop(columns=["num"], axis=1)  # "num" is the target column based on your dataset
y = data["num"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(pd.get_dummies(X, drop_first=True))  # Handle categorical features with one-hot encoding
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model training
if st.button("Train Model"):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Store the trained model in session state using st.cache_resource
    st.session_state.model = model  # Save model to session state

    st.write(f"### Model Accuracy: {acc:.2f}")
    st.write("#### Classification Report")
    st.text(classification_report(y_test, y_pred))

# Store feature names after preprocessing during training
feature_names = pd.get_dummies(X, drop_first=True).columns

# User prediction
st.write("### Predict on New Data")
new_data = {}
for col in X.columns:
    value = st.text_input(f"Enter {col}", value="")
    new_data[col] = value

if st.button("Predict Heart Disease"):
    # Check if model is trained
    if 'model' not in st.session_state:
        st.error("Please train the model first!")
    else:
        new_df = pd.DataFrame([new_data])
        
        # Ensure proper data types for numeric columns
        for col in ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        
        # Perform one-hot encoding and align columns
        new_encoded = pd.get_dummies(new_df, drop_first=True)
        aligned_new_encoded = new_encoded.reindex(columns=feature_names, fill_value=0)
        
        # Scale the data
        new_scaled = scaler.transform(aligned_new_encoded)
        
        # Make prediction
        prediction = st.session_state.model.predict(new_scaled)
        st.write("Prediction:", "Heart Disease" if prediction[0] > 0 else "No Heart Disease")
