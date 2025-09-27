# app.py
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from ml_trainer import train_classification_model
from llm_integrator import generate_insights

# Load environment variables from .env file at the very start
load_dotenv()

# Configure the page to use a wide layout
st.set_page_config(layout="wide")

# Main application title
st.title("Auto ML & Insights Generator")

# File uploader component
uploaded_file = st.file_uploader("Upload your CSV data", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        
        st.success("CSV file loaded successfully.")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        st.subheader("Dataset Information")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")

        # --- Model Training Section ---
        st.subheader("1. Train a Predictive Model")
        target_variable = st.selectbox(
            "Select your target variable (the column to predict)", 
            df.columns
        )

        if st.button("Train Classification Model"):
            with st.spinner("Training the model... This may take a moment."):
                try:
                    report, confusion_matrix_fig = train_classification_model(df, target_variable)
                    st.session_state.report = report
                    st.session_state.target_variable = target_variable
                    
                    st.success("Model trained successfully!")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Classification Report")
                        st.json(report)
                    with col2:
                        st.subheader("Confusion Matrix")
                        st.pyplot(confusion_matrix_fig)
                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")

# --- Insight Generation Section ---
# This section will only appear after a model has been successfully trained
if 'report' in st.session_state:
    st.subheader("2. Generate Automated Insights (Powered by AI)")
    if st.button("Generate Business Insights"):
        with st.spinner("AI is analyzing the results and drafting a report..."):
            # Retrieve all necessary data from the session state
            df = st.session_state.df
            target_variable = st.session_state.target_variable
            report = st.session_state.report
            
            feature_columns = [col for col in df.columns if col != target_variable]
            insights = generate_insights(report, feature_columns, target_variable)
            st.markdown(insights)