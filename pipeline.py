# Load the pre-trained classifier model
import pickle
import pandas as pd
with open('best_model.pkl', 'rb') as model_file:
    svm = pickle.load(model_file)


def pipe(df_eda):
    df_eda['Purchase History'] = pd.to_datetime(df_eda['Purchase History'], format='mixed',dayfirst=True)
    df_eda['Policy Start Date'] = pd.to_datetime(df_eda['Policy Start Date'], format='mixed',dayfirst=True)
    df_eda['Policy Renewal Date'] = pd.to_datetime(df_eda['Policy Renewal Date'], format='mixed',dayfirst=True)
    df_eda.drop_duplicates(inplace=False)
    to_drop = ['Claim_ID', 'Gender', 'Location', 'Interactions with Customer Service', 'Insurance Products Owned', 'Customer Preferences']
    df_eda.drop(to_drop, axis=1, inplace=True)
    from datetime import datetime

    # "Purchase History	Policy Start Date	Policy Renewal Date"
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
    
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate sentence embeddings
    embeddings = model.encode(df_eda['Claim_Description'].tolist(), show_progress_bar=True)

    import numpy as np
    embed_df = pd.DataFrame(embeddings, columns=[f'sbert_{i}' for i in range(embeddings.shape[1])])

    # Concatenate with original features
    df_eda = pd.concat([df_eda.reset_index(drop=True), embed_df], axis=1)


    from sklearn.preprocessing import MinMaxScaler
    df_eda.dropna(inplace=True)
    numeric_columns = df_eda.select_dtypes(include=['float64', 'int64']).columns
    df_eda[numeric_columns] = MinMaxScaler().fit_transform(df_eda[numeric_columns])

    # df_eda.head()
    # 2. Extract weights and bias
    w = svm.coef_.flatten()      # shape (n_features,)
    b = svm.intercept_[0]

    # 3. Get the raw feature vector for our single example
    x = df_eda.iloc[0].values.astype(float)  # ensure numeric dtype

    # 4. Compute contributions
    contribs = w * x

    # 5. Pair with feature names, sort by descending contribution
    feat_contrib = list(zip(df_eda.columns, contribs))
    feat_contrib_sorted = sorted(feat_contrib, key=lambda x: x[1], reverse=True)

    # 6. Take top 3 positive and top 2 negative for context
    top_pos = feat_contrib_sorted[:3]
    top_neg = feat_contrib_sorted[-2:]



    y_pred_proba = svm.predict_proba(df_eda)

    return y_pred_proba[0][0], top_pos, top_neg, feat_contrib_sorted
