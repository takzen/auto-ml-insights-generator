# llm_integrator.py
import google.generativeai as genai
import os
import json

def generate_insights(report: dict, column_names: list, target_column: str):
    """
    Generates business insights from model results using the Gemini 1.5 Pro model.
    """
    try:
        # Configure the API key from environment variables for security
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "### Configuration Error\n**Google AI API key not found.** Please set the `GOOGLE_API_KEY` in your `.env` file."
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"### API Key Configuration Error\nDetails: {e}"

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
        # Use the latest powerful model available through the API
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"### An Error Occurred\nAn error occurred while communicating with the Google Generative AI API: {e}"