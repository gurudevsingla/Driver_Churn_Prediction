import streamlit as st
import joblib
import numpy as np

# Force the app to use light theme
st.set_page_config(page_title="Driver Churn Prediction", page_icon="🚗", layout="centered",
                   initial_sidebar_state="expanded")

# Load the scaler and model
scaler = joblib.load('models/scaler.pkl')  # Correct the path if needed
rf_model = joblib.load('models/rf_model.pkl')  # Correct the path if needed

# Light Theme Styling
background_color = "#f0f4f8"  # Light background
text_color = "#333333"  # Dark text for readability
button_color = "#4CAF50"
button_hover_color = "#45a049"
churn_high_color = "#f44336"
churn_low_color = "#4CAF50"
form_background_color = "#ffffff"  # White form background
form_text_color = "#333333"  # Dark text inside form

# Apply CSS for light theme
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {background_color};
            color: {text_color};
        }}
        .main {{
            background-color: {form_background_color};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .stButton>button {{
            background-color: {button_color};
            color: {text_color};
            border-radius: 5px;
            padding: 12px 24px;
            font-size: 18px;
            transition: background-color 0.3s;
        }}
        .stButton>button:hover {{
            background-color: {button_hover_color};
        }}
        .stSelectbox, .stNumberInput {{
            font-size: 16px;
            padding: 10px;
            background-color: {form_background_color};
            border-radius: 5px;
        }}
        .churn-high {{
            color: {churn_high_color}; /* Red for high churn risk */
            font-size: 28px;
            font-weight: bold;
        }}
        .churn-low {{
            color: {churn_low_color}; /* Green for low churn risk */
            font-size: 28px;
            font-weight: bold;
        }}
        .stTitle {{
            color: {text_color};
        }}
        .stMarkdown {{
            font-size: 18px;
            color: {form_text_color};
        }}
        .stForm {{
            background-color: {form_background_color};
            color: {form_text_color};
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }}
        .stSelectbox div {{
            color: {form_text_color};
        }}
    </style>
""", unsafe_allow_html=True)

# UI for input fields
st.title("🚗 Driver Churn Prediction")
st.markdown("Please fill out the details below to predict the churn probability for a driver.")

# Create an interactive form for the user input
with st.form("driver_form", clear_on_submit=False):
    # Collecting inputs with better UI
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (in years)", min_value=18, max_value=100, step=1,
                              help="Enter the age of the driver.")
        income = st.number_input("Income (in rupees)", min_value=1000, max_value=1000000, step=1000,
                                 help="Enter the income of the driver in rupees.")
        total_business_value = st.number_input("Total Business Value (in rupees)", min_value=1000, max_value=1000000,
                                               step=1000, help="Enter the total business value in rupees.")
        joining_designation = st.selectbox("Joining Designation", [1, 2, 3, 4, 5],
                                           help="Select the joining designation of the driver.")
        last_quarterly_rating = st.selectbox("Last Quarterly Rating", [1, 2, 3, 4, 5],
                                             help="Select the last quarterly rating of the driver.")

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"], help="Select the gender of the driver.")
        education = st.selectbox("Education Level", ["10+", "12+", "Graduate"],
                                 help="Select the education level of the driver.")
        grade = st.selectbox("Grade", [1, 2, 3, 4, 5], help="Select the grade of the driver.")
        quarterly_rating_increased = st.selectbox("Quarterly Rating Increased", ["Yes", "No"],
                                                  help="Did the quarterly rating increase?")
        salary_increased = st.selectbox("Salary Increased", ["Yes", "No"], help="Did the driver's salary increase?")

    # Submit button
    submit_button = st.form_submit_button("Predict Churn Probability")

# Preprocessing the inputs when the button is clicked
if submit_button:
    # Map user input to the required format
    gender = 0 if gender == "Male" else 1
    education_mapping = {"10+": 0, "12+": 1, "Graduate": 2}
    education = education_mapping[education]
    quarterly_rating_increased = 1 if quarterly_rating_increased == "Yes" else 0
    salary_increased = 1 if salary_increased == "Yes" else 0

    # Create a numpy array with the user input data
    user_input = np.array([[
        age,
        gender,
        education,
        income,
        joining_designation,
        grade,
        total_business_value,
        last_quarterly_rating,
        quarterly_rating_increased,
        salary_increased
    ]])

    # Transform the input data using the loaded scaler
    user_input_scaled = scaler.transform(user_input)

    # Predict churn probability
    churn_prob = rf_model.predict_proba(user_input_scaled)[:, 1]  # Assuming class 1 is churn

    # Display the result in an attractive way
    st.markdown("### Churn Probability")
    st.write(f"🟢 **Churn Probability**: {churn_prob[0]:.4f}")

    # Add a visual indicator based on churn probability
    if churn_prob[0] > 0.5:
        st.markdown('<div class="churn-high">⚠️ High Churn Risk!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="churn-low">✔️ Low Churn Risk!</div>', unsafe_allow_html=True)
