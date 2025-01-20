import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st
import plotly.graph_objects as go

# Load the dataset
try:
    df = pd.read_csv('heart (1).csv')

    # Preprocess the data
    X = df.drop('target', axis=1)
    y = df['target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Model evaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Add custom CSS
    st.markdown("""
        <style>
            body {
                background-color: #f7f8fc;
            }
            .main {
                background: #ffffff;
                border-radius: 10px;
                padding: 20px;
            }
            h1, h2, h3, h4 {
                color: #4c4c4c;
            }
            .stButton>button {
                background: linear-gradient(to right, #6a11cb, #2575fc);
                border: none;
                color: white;
                padding: 8px 16px;
                border-radius: 5px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background: linear-gradient(to right, #2575fc, #6a11cb);
            }
        </style>
    """, unsafe_allow_html=True)

    # Streamlit app
    st.title("💖 Heart Disease Prediction Dashboard")

    # Display model evaluation metrics
    st.subheader("Model Evaluation")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Accuracy", value=f"{accuracy:.2%}")
    with col2:
        st.text("Confusion Matrix:")
        st.text(cm)

    # Display the classification report as a table
    st.subheader("Classification Report")
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(2)  # Round values to 2 decimal places
    st.dataframe(
        report_df.style.applymap(
            lambda val: "background-color: #cce5ff;" if isinstance(val, float) and val >= 0.85 else
                        "background-color: #ffcccb;" if isinstance(val, float) else ""
        ).format("{:.2f}")
    )

    # Initialize session state to handle reset
    if "reset" not in st.session_state:
        st.session_state.reset = False
        st.session_state.age = 0
        st.session_state.sex = ""
        st.session_state.cp = ""
        st.session_state.trestbps = 0
        st.session_state.chol = 0
        st.session_state.fbs = ""
        st.session_state.restecg = ""
        st.session_state.thalach = 0
        st.session_state.exang = ""
        st.session_state.oldpeak = 0.0
        st.session_state.slope = ""
        st.session_state.ca = 0
        st.session_state.thal = ""

    # Function to reset input fields
    def reset_fields():
        st.session_state.reset = True

    # Get user input
    st.subheader("Patient Information")
    st.markdown("### Fill in the following details to predict heart disease risk:")

    age = st.number_input("🧓 Age", min_value=0, max_value=120, value=st.session_state.age, key='age')
    sex = st.selectbox("🚻 Gender", ["", "Male", "Female"], key='sex')
    cp = st.selectbox("💔 Chest Pain Type", ["", "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], key='cp')
    trestbps = st.number_input("📉 Resting Blood Pressure", min_value=0, max_value=300, value=st.session_state.trestbps, key='trestbps')
    chol = st.number_input("🩸 Cholesterol", min_value=0, max_value=600, value=st.session_state.chol, key='chol')
    fbs = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", ["", "True", "False"], key='fbs')
    restecg = st.selectbox("📊 Resting Electrocardiogram", ["", "Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"], key='restecg')
    thalach = st.number_input("🏃 Maximum Heart Rate", min_value=0, max_value=220, value=st.session_state.thalach, key='thalach')
    exang = st.selectbox("💪 Exercise-Induced Angina", ["", "True", "False"], key='exang')
    oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=st.session_state.oldpeak, key='oldpeak')
    slope = st.selectbox("📈 Slope of ST Segment", ["", "Upsloping", "Flat", "Downsloping"], key='slope')
    ca = st.number_input("🩻 Major Vessels", min_value=0, max_value=4, value=st.session_state.ca, key='ca')
    thal = st.selectbox("🧬 Thalassemia", ["", "Normal", "Fixed Defect", "Reversible Defect"], key='thal')

    # Convert user input to numerical values
    user_input = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [1 if cp == "Typical Angina" else 2 if cp == "Atypical Angina" else 3 if cp == "Non-Anginal Pain" else 4],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "True" else 0],
        "restecg": [1 if restecg == "Normal" else 2 if restecg == "ST-T Wave Abnormality" else 3],
        "thalach": [thalach],
        "exang": [1 if exang == "True" else 0],
        "oldpeak": [oldpeak],
        "slope": [1 if slope == "Upsloping" else 2 if slope == "Flat" else 3],
        "ca": [ca],
        "thal": [1 if thal == "Normal" else 2 if thal == "Fixed Defect" else 3]
    })

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Make a prediction
    if st.button("🔍 Predict"):
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)[0][1]  # Probability of having heart disease

        # Display the prediction
        st.markdown("### Prediction:")
        if prediction[0] == 1:
            st.success("You are likely to have heart disease. 🩺")
        else:
            st.success("You are unlikely to have heart disease. 🎉")

        # Display probability using a gauge chart
        st.markdown("### Heart Disease Risk Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba * 100,
            title={'text': "Heart Disease Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 20], 'color': "lightgreen"},
                    {'range': [20, 40], 'color': "green"},
                    {'range': [40, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"}]
            }
        ))
        st.plotly_chart(fig)

    # Reset button
    if st.button("🔄 Reset"):
        reset_fields()
        st.experimental_rerun()

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error(f"Exception type: {type(e).__name__}")
