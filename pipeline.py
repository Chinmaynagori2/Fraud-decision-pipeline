# Load the pre-trained classifier model
import pickle
import pandas as pd
with open('rf_classifier.pkl', 'rb') as model_file:
    rf_classifier = pickle.load(model_file)


def pipe(df_eda):
    df_eda['Purchase History'] = pd.to_datetime(df_eda['Purchase History'], format='mixed',dayfirst=True)
    df_eda['Policy Start Date'] = pd.to_datetime(df_eda['Policy Start Date'], format='mixed',dayfirst=True)
    df_eda['Policy Renewal Date'] = pd.to_datetime(df_eda['Policy Renewal Date'], format='mixed',dayfirst=True)
    df_eda.drop_duplicates(inplace=False)
    to_drop = ['Claim_ID', 'Gender', 'Location', 'Interactions with Customer Service', 'Insurance Products Owned', 'Customer Preferences']
    df_eda.drop(to_drop, axis=1, inplace=True)
    from datetime import datetime

    "Purchase History	Policy Start Date	Policy Renewal Date"
    current_date = datetime.now()
    df_eda['purchase age'] = (current_date - df_eda['Purchase History']).dt.days/365.25
    df_eda['policy age'] = (df_eda['Policy Renewal Date'] - df_eda['Policy Start Date']).dt.days/365.25
    df_eda.drop(columns=['Purchase History', 'Policy Start Date', 'Policy Renewal Date'], inplace=True)
    df_eda = df_eda.dropna(axis=0)
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()

    object_columns = df_eda.select_dtypes(include=['object']).columns
    object_columns = object_columns.drop('Claim_Description')

    for col in object_columns:
        df_eda[col] = label_encoder.fit_transform(df_eda[col])
    def preprocess_text(text):
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    df_eda['Claim_Description'] = df_eda['Claim_Description'].apply(preprocess_text)
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.tokenize import word_tokenize
    import nltk

    nltk.download('punkt')

    df_eda['Claim_Description_Tokens'] = df_eda['Claim_Description'].apply(word_tokenize)

    vectorizer = CountVectorizer(max_features=500)
    claim_description_one_hot = vectorizer.fit_transform(df_eda['Claim_Description']).toarray()

    one_hot_columns = vectorizer.get_feature_names_out()
    claim_description_one_hot_df = pd.DataFrame(claim_description_one_hot, columns=one_hot_columns)

    df_eda = pd.concat([df_eda.drop(columns=['Claim_Description']), claim_description_one_hot_df], axis=1)
    df_eda.dropna(inplace=True)
    from sklearn.preprocessing import MinMaxScaler
    import re

    numeric_columns = df_eda.select_dtypes(include=['float64', 'int64']).columns
    df_eda[numeric_columns] = MinMaxScaler().fit_transform(df_eda[numeric_columns])

    # df_eda.head()

    y_pred_proba = rf_classifier.predict_proba(df_eda)
    return y_pred_proba[0][0]
