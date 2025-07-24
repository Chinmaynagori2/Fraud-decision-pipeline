import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from pipeline import pipe
import re
import sqlite3

columns = ['Customer_Age', 'Claim_Amount', 'Claim_History', 'Policy_Type',
           'Incident_Severity', 'Claim_Frequency',
           'Claim_Description', 'Marital Status', 'Occupation', 'Income Level',
           'Education Level', 'Behavioral Data', 'Purchase History',
           'Policy Start Date', 'Policy Renewal Date', 'Coverage Amount',
           'Premium Amount', 'Deductible', 'Policy Type',
           'Preferred Communication Channel', 'Driving Record', 'Life Events']

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def parse_pdf_to_dict(pdf_text, columns):
    data_dict = {col: [] for col in columns}
    lines = pdf_text.split("\n")
    for line in lines:
        for col in columns:
            match = re.search(rf"{col}:\s*(.*)", line, re.IGNORECASE)
            if match:
                data_dict[col].append(match.group(1).strip())
    for col in columns:
        if len(data_dict[col]) == 0:
            data_dict[col].append(None)
    return data_dict

st.title("Fraud detection")

uploaded_file = st.file_uploader("Upload the PDF", type=["pdf"])

conn = sqlite3.connect('INSTRUCTOR.db')
cursor_obj = conn.cursor()
cursor_obj.execute('''
    CREATE TABLE IF NOT EXISTS data_table(
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
    pdf_text = extract_text_from_pdf(uploaded_file)
    data_dict = parse_pdf_to_dict(pdf_text, columns)
    df = pd.DataFrame(data_dict)

    valid = True
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if (df[col] < 0).any() or df[col].isnull().any():
                st.write(f"Invalid data in column: {col}")
                valid = False
                break
        elif df[col].dtype == 'object':
            if df[col].isnull().any() or (df[col] == '').any():
                st.write(f"Invalid data in column: {col}")
                valid = False
                break

    if valid:
        prob = pipe(df)
        if prob > 0.5:
            st.write("Fraud detected")
        elif prob < 0.2:
            st.write("No fraud detected")
            df.to_sql('data_table', conn, if_exists='append', index=False)
            st.write("Data saved in database")
            
        else:
            st.write("Human review needed")
            st.write("Probability of fraud:", prob)
