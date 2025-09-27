# llm_integrator.py
import google.generativeai as genai
import os
import json
import streamlit as st # Upewnij się, że ten import jest na górze

def generate_insights(report: dict, column_names: list, target_column: str):
    """
    Generates business insights from model results using the Gemini 1.5 Pro model.
    """
    try:
        # This logic checks for the key in Streamlit's secrets first (for cloud deployment),
        # then falls back to .env for local development.
        if 'google_genai' in st.secrets:
            api_key = st.secrets["google_genai"]["api_key"]
        else:
            api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            raise ValueError("Google AI API key not found. Please set it in Streamlit secrets or your .env file.")
        
        genai.configure(api_key=api_key)

    except Exception as e:
        return f"### Configuration Error\nAn error occurred while configuring the API key: {e}"

    prompt = f"""
    You are a data science expert. Your task is to interpret the results of a machine learning model and present the findings in an accessible, business-friendly manner.

    Context:
    - A classification model was built to predict the column named '{target_column}'.
    - The features used for prediction were: {', '.join(column_names)}.

    Model Results (classification report in JSON format):
    {json.dumps(report, indent=2)}

    Your Task:
    1. In simple terms, explain what the key metrics `accuracy`, `precision`, and `recall` mean in the context of this problem.
    2. Based on these metrics, evaluate whether the model is good. Is it ready for business use?
    3. Provide two concrete, actionable business insights or recommendations that can be derived from the model's performance. Focus on how a company could use this model to make decisions.

    Present your response in elegant Markdown format, using headers and lists.
    """
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"### An Error Occurred\nAn error occurred while communicating with the Google Generative AI API: {e}"