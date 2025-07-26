import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# --- Cache Model and Encoders for Performance ---
@st.cache_resource
def load_resources():
    try:
        model_pipeline = joblib.load("best_model_pipeline.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        income_encoder = joblib.load("income_encoder.pkl")
        return model_pipeline, label_encoders, income_encoder
    except FileNotFoundError:
        st.error("Error: Model or encoders not found. Please ensure 'best_model_pipeline.pkl', 'label_encoders.pkl', and 'income_encoder.pkl' are in the same directory as the app. Run the Jupyter notebook cells to train and save them.")
        st.stop()

model_pipeline, label_encoders, income_encoder = load_resources()

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Classification App 💼",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling (Final Version with All Color Effects Background & White Bordered Text) ---
st.markdown("""
<style>
    /* Global Text Styling with White Color and Black Border (Outline) */
    body, .stApp, .stMarkdown, .stText, p, span, li {
        color: white !important; /* Force white text */
        text-shadow:
            -1px -1px 0 #000,
             1px -1px 0 #000,
            -1px  1px 0 #000,
             1px  1px 0 #000; /* Black outline using text-shadow */
    }

    /* Override for input fields and specific elements where white text is not desired */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stSelectbox .css-1dp5eyl.e1g8pov61, /* Target the actual selected value in selectbox */
    .stDateInput input,
    .stTimeInput input {
        color: #343A40 !important; /* Keep input values dark for readability */
        text-shadow: none !important; /* Remove text-shadow from input values */
    }

    /* Specific overrides for elements that should not have the border/white text */
    .stAlert {
        color: inherit !important; /* Alerts should use their default text color */
        text-shadow: none !important; /* Remove text-shadow from alerts */
    }
    .stAlert.info {
        color: #004085 !important; /* Info alert specific text color */
    }
    .stAlert.success {
        color: #155724 !important; /* Success alert specific text color */
    }
    .stAlert.warning {
        color: #856404 !important; /* Warning alert specific text color */
    }
    .stAlert.error {
        color: #721c24 !important; /* Error alert specific text color */
    }

    /* Prediction result box text - specific styling for highlight */
    .prediction-result {
        background-color: #D4EDDA; /* Light green for success */
        border-left: 8px solid #28A745; /* Thicker green border */
        padding: 3em 2em; /* Increased padding for more space */
        margin-top: 2em;
        border-radius: 12px;
        font-weight: bold;
        text-align: center;
        box-shadow: 0px 8px 20px rgba(40,167,69,0.2); /* More prominent shadow */
        animation: fadeIn 1s ease-out;

        /* Flexbox for vertical stacking and centering content */
        display: flex;
        flex-direction: column; /* Stack items vertically */
        align-items: center;    /* Center horizontally */
        justify-content: center; /* Center vertically */
    }

    .prediction-result .prediction-label {
        font-size: 1.8em; /* Size for "Predicted Income Class:" */
        font-weight: 600; /* Slightly less bold than the value */
        margin-bottom: 0.5em; /* Space between label and value */
        color: #155724 !important; /* Dark green text for readability */
        text-shadow: none !important; /* No text shadow for clarity */
    }

    .prediction-result .prediction-value {
        font-size: 3.5em; /* Significantly larger for the actual predicted value */
        font-weight: 900; /* Extra bold */
        letter-spacing: 2px; /* Add some spacing to the value */
        color: #28A745 !important; /* A slightly brighter green for the value */
        text-shadow: none !important; /* No text shadow for clarity */
    }


    /* Buttons text - Streamlit buttons already handle their own text color */
    .stButton>button {
        color: white !important; /* Ensure button text is white */
        text-shadow: none !important; /* No text shadow on button text */
    }

    /* Overall App Background and Font */
    .stApp {
        /* All Color Effects Background Texture with Dynamic Shifts */
        background:
            radial-gradient(at 20% 80%, #ff6f6130, transparent 60%),  /* Soft Coral */
            radial-gradient(at 80% 20%, #4CAF5030, transparent 60%),  /* Leaf Green */
            radial-gradient(at 50% 50%, #2196F330, transparent 60%),  /* Bright Blue */
            radial-gradient(at 0% 100%, #9C27B030, transparent 60%),  /* Deep Purple */
            radial-gradient(at 100% 0%, #FFC10730, transparent 60%),  /* Amber Yellow */
            linear-gradient(10deg, #E91E6320, transparent 70%),     /* Pink */
            linear-gradient(160deg, #00BCD420, transparent 70%);     /* Cyan */

        background-size: 400% 400%; /* Even larger for more dramatic movement */
        background-position: 0% 0%; /* Initial position */
        background-blend-mode: soft-light; /* A good blend for complex color layering. Experiment with 'overlay', 'screen', 'hard-light', 'difference' */

        animation: allColorBackgroundShift 90s ease infinite alternate; /* Slower, smoother, infinite shift */

        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Keyframe Animation for the All-Color Background */
    @keyframes allColorBackgroundShift {
        0% { background-position: 0% 0%; }
        20% { background-position: 100% 0%; }
        40% { background-position: 50% 100%; }
        60% { background-position: 0% 100%; }
        80% { background-position: 100% 50%; }
        100% { background-position: 0% 0%; }
    }


    /* Headers */
    h1 {
        /* Headers already included in global, but you can override specific font size/weight here if needed */
        text-shadow:
            -2px -2px 0 #000,
             2px -2px 0 #000,
            -2px  2px 0 #000,
             2px  2px 0 #000,
            -2px  0px 0 #000, /* Add more shadows for thicker border on headings */
             2px  0px 0 #000,
             0px -2px 0 #000,
             0px  2px 0 #000;
    }
    h2 {
        text-shadow:
            -1.5px -1.5px 0 #000,
             1.5px -1.5px 0 #000,
            -1.5px  1.5px 0 #000,
             1.5px  1.5px 0 #000;
    }
    h3 {
         text-shadow:
            -1px -1px 0 #000,
             1px -1px 0 #000,
            -1px  1px 0 #000,
             1px  1px 0 #000;
    }


    /* Labels for Input Fields */
    .stTextInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        color: white !important; /* Ensured labels are white */
        text-shadow:
            -1px -1px 0 #000,
             1px -1px 0 #000,
            -1px  1px 0 #000,
             1px  1px 0 #000; /* Black outline */
    }

    /* Streamlit Containers / Cards */
    .stContainer {
        background-color: rgba(255, 255, 255, 0.9); /* Slightly transparent white for contrast against busy background */
        border-radius: 12px;
        padding: 2.5em;
        margin-bottom: 2em;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); /* More prominent shadow for cards */
        border: 1px solid #E9ECEF; /* Subtle border */
    }

    /* Sidebar Customization */
    .st-emotion-cache-1ldf5b-container { /* This targets the sidebar container */
        background-color: rgba(240, 242, 246, 0.95); /* Slightly transparent sidebar */
        padding: 2em 1.5em;
        border-right: 1px solid #DEE2E6;
        box-shadow: 2px 0px 8px rgba(0,0,0,0.05);
    }
    /* Specific text within sidebar that might need adjustment */
    .st-emotion-cache-1ldf5b-container .st-emotion-cache-10q7pck, /* Sidebar Title */
    .st-emotion-cache-1ldf5b-container .stMarkdown p, /* Sidebar general markdown text */
    .st-emotion-cache-1ldf5b-container .stInfo, /* Sidebar st.info box */
    .st-emotion-cache-1ldf5b-container label {
        color: #343A40 !important; /* Keep sidebar text dark for readability against its lighter background */
        text-shadow: none !important; /* Remove text shadow in sidebar */
    }


    /* Adjust padding for the main content block */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
     @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }

</style>
""", unsafe_allow_html=True)

# --- Preprocessing Function ---
def preprocess_input(df_raw, encoders):
    df = df_raw.copy()

    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            mapping_dict = {cls: idx for idx, cls in enumerate(encoder.classes_)}
            df[col] = df[col].map(mapping_dict)
            df[col] = df[col].fillna(0).astype(int)
        else:
            st.warning(f"Column '{col}' expected for encoding but not found in input data. Skipping encoding for this column.")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            st.error(f"Error converting column '{col}' to numeric after preprocessing. Check data types.")
            raise ValueError(f"Non-numeric values found in column '{col}' after preprocessing.")
    return df

# --- Main Title and Description ---
st.title("💸 Employee Salary Classifier")
st.markdown("""
<p style='font-size:1.3em; text-align: center; color: white;'>
    Uncover insights into income levels: Predict whether an employee's annual income is
    <span style='font-weight:bold; color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>&gt;50K USD</span> or
    <span style='font-weight:bold; color: white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000;'>&le;50K USD</span>
    based on their demographic and employment attributes.
</p>
""", unsafe_allow_html=True)

st.write("---")

# --- Input Section (Single Prediction) ---
st.header("👤 Single Employee Prediction")
st.markdown("Enter the details of an employee below to predict their salary class.")

with st.container(border=False):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📝 Personal & Work Info")
        age = st.slider("Age", 17, 90, 35, help="Age of the employee (years)")
        workclass_options = sorted(label_encoders['workclass'].classes_)
        workclass = st.selectbox("Workclass", workclass_options, help="Type of employer (e.g., Private, Self-emp-not-inc)")
        gender_options = sorted(label_encoders['gender'].classes_)
        gender = st.selectbox("Gender", gender_options, help="Employee's gender (Male/Female)")
        race_options = sorted(label_encoders['race'].classes_)
        race = st.selectbox("Race", race_options, help="Employee's racial background")

    with col2:
        st.markdown("### 🎓 Education & Marital Status")
        education_to_num = {
            "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
            "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
            "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
        }
        education_display_options = sorted(education_to_num.keys(), key=lambda x: education_to_num[x])
        selected_education_str = st.selectbox("Education Level", education_display_options, index=education_display_options.index("Bachelors"), help="Highest level of education achieved")
        educational_num = education_to_num[selected_education_str]

        marital_status_options = sorted(label_encoders['marital-status'].classes_)
        marital_status = st.selectbox("Marital Status", marital_status_options, help="Employee's marital status")
        relationship_options = sorted(label_encoders['relationship'].classes_)
        relationship = st.selectbox("Relationship", relationship_options, help="Relationship status (e.g., Husband, Not-in-family)")

    with col3:
        st.markdown("### 💰 Financial & Other Details")
        occupation_options = sorted(label_encoders['occupation'].classes_)
        occupation = st.selectbox("Occupation", occupation_options, help="Employee's occupation")
        hours_per_week = st.slider("Hours per Week", 1, 99, 40, help="Average hours worked per week")
        capital_gain = st.number_input("Capital Gain", min_value=0, max_value=100000, value=0, step=100, help="Capital gains from investments (USD)")
        capital_loss = st.number_input("Capital Loss", min_value=0, max_value=100000, value=0, step=100, help="Capital losses (USD)")
        fnlwgt = st.number_input("Fnlwgt (Final Weight)", min_value=10000, max_value=1000000, value=200000, step=1000, help="Population weight assigned by census (statistical value)")
        native_country_options = sorted(label_encoders['native-country'].classes_)
        native_country = st.selectbox("Native Country", native_country_options, index=native_country_options.index('United-States'), help="Country of origin")

    st.markdown("---")
    st.markdown("#### Review Input Data:")
    expected_columns_order = [
        'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status', 'occupation',
        'relationship', 'race', 'gender', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country'
    ]

    input_data_dict = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    input_df_raw = pd.DataFrame([input_data_dict])
    st.dataframe(input_df_raw)

    if st.button("🚀 Predict Salary Class for Single Employee"):
        with st.spinner('Calculating prediction...'):
            time.sleep(1)
            try:
                processed_input_df = preprocess_input(input_df_raw.copy(), label_encoders)
                processed_input_df = processed_input_df[expected_columns_order]
                prediction_numeric = model_pipeline.predict(processed_input_df)[0]
                prediction_class = income_encoder.inverse_transform([prediction_numeric])[0]

                st.markdown(f"""
                <div class="prediction-result">
                    <div class="prediction-label">✨ Predicted Income Class:</div>
                    <div class="prediction-value">{prediction_class}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)

st.write("---")

# --- Batch Prediction Section ---
st.header("📁 Batch Prediction (Upload CSV)")
st.markdown("Upload a CSV file with multiple employee entries to get predictions for all of them.")
st.info("The CSV file must contain the following columns: " + ", ".join([f"`{col}`" for col in expected_columns_order]) + ". Ensure column names match exactly.")

with st.container(border=False):
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            batch_data_raw = pd.read_csv(uploaded_file)
            st.markdown("### Uploaded Data Preview:")
            st.dataframe(batch_data_raw.head())

            st.write("Processing batch predictions...")
            with st.spinner('Making predictions for your batch data... This may take a moment.'):
                time.sleep(1.5)

                batch_processed_df = preprocess_input(batch_data_raw.copy(), label_encoders)
                batch_processed_df = batch_processed_df[expected_columns_order]

                batch_predictions_numeric = model_pipeline.predict(batch_processed_df)
                batch_predictions_class = income_encoder.inverse_transform(batch_predictions_numeric)

                batch_data_raw['Predicted_Income_Class'] = batch_predictions_class

            st.success("✅ Batch predictions complete!")
            st.markdown("### Predicted Results (First 5 Rows):")
            st.dataframe(batch_data_raw.head())

            csv = batch_data_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Full Predictions CSV",
                data=csv,
                file_name="predicted_employee_salaries.csv",
                mime="text/csv",
                help="Download the original CSV with an added 'Predicted_Income_Class' column."
            )
        except Exception as e:
            st.error(f"An error occurred during batch prediction: {e}")
            st.exception(e)

st.write("---")
st.markdown("""
    <div style="text-align: center; color: white; font-size: 0.9em; margin-top: 2em;">
        Developed with ❤️ for IBM SkillsBuild Project | Version 1.5
    </div>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.header("💡 About This App")
    st.info("""
    This application predicts an employee's annual income level
    (whether it's `>50K` or `≤50K` USD) based on various
    demographic and employment features.

    It uses a pre-trained machine learning model (Random Forest Classifier)
    and scikit-learn preprocessing pipelines saved from a Jupyter Notebook.
    """)
    st.subheader("How to Use:")
    st.markdown("""
    1.  **Single Prediction:** Fill out the form with employee details and click 'Predict'.
    2.  **Batch Prediction:** Upload a CSV file with your data. Ensure column names match
        the input fields (e.g., `age`, `workclass`, `capital-gain`, etc.).
    """)
    st.subheader("Data Source:")
    st.markdown("This model is trained on a modified version of the [Adult Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult).")
    st.markdown("[GitHub Repository](https://github.com/your-repo-link-here) (Replace with your actual link)")