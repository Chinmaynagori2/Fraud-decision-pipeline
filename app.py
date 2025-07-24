import streamlit as st
import pdfplumber
import pandas as pd
import sqlite3
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from pipeline import pipe
import json
from dotenv import load_dotenv


load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Define the fields we want to extract
FIELDS = ['Customer_Age', 'Claim_Amount', 'Claim_History', 'Policy_Type',
          'Incident_Severity', 'Claim_Frequency', 'Claim_Description',
          'Marital Status', 'Occupation', 'Income Level', 'Education Level',
          'Behavioral Data', 'Purchase History', 'Policy Start Date',
          'Policy Renewal Date', 'Coverage Amount', 'Premium Amount',
          'Deductible', 'Policy Type', 'Preferred Communication Channel',
          'Driving Record', 'Life Events']

from pydantic import BaseModel
from typing import Optional

class ClaimFields(BaseModel):
    Customer_Age: Optional[int]
    Claim_Amount: Optional[int]
    Claim_History: Optional[str]
    Policy_Type: Optional[str]
    Incident_Severity: Optional[str]
    Claim_Frequency: Optional[int]
    Claim_Description: Optional[str]
    Marital_Status: Optional[str]
    Occupation: Optional[str]
    Income_Level: Optional[int]
    Education_Level: Optional[str]
    Behavioral_Data: Optional[str]
    Purchase_History: Optional[str]
    Policy_Start_Date: Optional[str]
    Policy_Renewal_Date: Optional[str]
    Coverage_Amount: Optional[int]
    Premium_Amount: Optional[int]
    Deductible: Optional[int]
    Policy_Type_2: Optional[str]
    Preferred_Communication_Channel: Optional[str]
    Driving_Record: Optional[str]
    Life_Events: Optional[str]


# --- Functions ---

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    return full_text

def extract_fields_with_llm(text, fields):
    prompt = f"""
You are an AI assistant. Extract the following fields from the PDF content provided below.

Fields to extract:
{', '.join(fields)}

Return the results in JSON format. Use null if a field is missing.

PDF Content:
\"\"\"
{text}
\"\"\"
"""
    structured_llm = llm.with_structured_output(ClaimFields)
    try:
        claim = structured_llm.invoke(prompt)
        return claim.dict()
    except Exception as e:
        st.error(f"Could not parse response from Gemini: {e}")
        return None

def validate_dataframe(df):
    valid = True
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if (df[col] < 0).any() or df[col].isnull().any():
                st.warning(f"Invalid numeric data in: {col}")
                valid = False
        elif df[col].dtype == 'object':
            if df[col].isnull().any() or (df[col] == '').any():
                st.warning(f"Invalid text data in: {col}")
                valid = False
    return valid

def build_summary_prompt(record: dict,
                         top_pos: list[tuple[str,float]],
                         top_neg: list[tuple[str,float]]
                        ) -> str:
    # record: dict of field:value
    pos_lines = "\n".join([f"- **{k}**: contribution = {v:.2f}" for k,v in top_pos])
    neg_lines = "\n".join([f"- **{k}**: contribution = {v:.2f}" for k,v in top_neg])

    return f"""
        You are a fraud-detection analyst assistant.  
        Below is a customer claim profile and the six most impactful features (three pushing toward fraud, two away from fraud).  
        Generate a ONE-PAGE summary for a human reviewer—include:  
        1. A very brief description of the case (who, what).  
        2. The key drivers toward fraud.  
        3. Any reassuring factors.  
        4. A final “recommendation” (e.g. “Likely fraudulent—escalate to Investigation”).  

        **Profile:**  
        {json.dumps(record, indent=2)}  

        **Top positive fraud contributions:**  
        {pos_lines}

        **Top negative fraud contributions:**  
        {neg_lines}

        Write in clear bullet points or short paragraphs.
        """


# --- Streamlit App ---

st.title("PDF Fraud Detection App")

uploaded_file = st.file_uploader("Upload a digital PDF", type=["pdf"])

# Database setup
conn = sqlite3.connect('INSTRUCTOR.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS data_table (
    Customer_Age INTEGER,
    Claim_Amount INTEGER,
    Claim_History TEXT,
    Policy_Type TEXT,
    Incident_Severity TEXT,
    Claim_Frequency INTEGER,
    Claim_Description TEXT,
    Marital_Status TEXT,
    Occupation TEXT,
    Income_Level INTEGER,
    Education_Level TEXT,
    Behavioral_Data TEXT,
    Purchase_History TEXT,
    Policy_Start_Date TEXT,
    Policy_Renewal_Date TEXT,
    Coverage_Amount INTEGER,
    Premium_Amount INTEGER,
    Deductible INTEGER,
    Policy_Type_2 TEXT,
    Preferred_Communication_Channel TEXT,
    Driving_Record TEXT,
    Life_Events TEXT
)
''')

if uploaded_file is not None:
    with st.spinner(" Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Extracting fields using Gemini..."):
        extracted_data = extract_fields_with_llm(pdf_text, FIELDS)

    if extracted_data:
        df = pd.DataFrame([extracted_data])

        st.subheader(" Extracted Fields")
        st.dataframe(df)


        if validate_dataframe(df):
            prob, top_pos, top_neg, feat_contrib_sorted = pipe(df)
            st.subheader("Prediction Result")
            if prob > 0.5:
                st.error("Fraud detected")
            elif prob < 0.2:
                st.success("No fraud detected")
                df.to_sql('data_table', conn, if_exists='append', index=False)
                st.info("Data saved to database")
            else:
                st.warning("Human review needed")
                st.write("Fraud probability:", prob)
                record = df.iloc[0].to_dict()
                summary_prompt = build_summary_prompt(record, top_pos, top_neg)
                summary_resp = llm.invoke(summary_prompt).content
                with open("output_summary/summary.txt", "w", encoding="utf-8") as f:
                    f.write(summary_resp.text)
                st.info("Summary saved")
