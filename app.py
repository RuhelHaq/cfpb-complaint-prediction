import streamlit as st
import pandas as pd
import xgboost as xgb
import anthropic
import joblib
import json

# Paths to your saved artifacts
ARTIFACT_DIR = '/home/ubuntu/data/nb4_artifacts'
DATA_DIR = '/home/ubuntu/data'

# Load the trained model
model = xgb.XGBClassifier()
model.load_model(f"{ARTIFACT_DIR}/model_enriched_v2.json")

# Load the target encoder (from NB3)
target_encoder = joblib.load(f"{DATA_DIR}/model_artifacts/target_encoder.pkl")

# Load column lists and the extraction tool schema
with open(f"{ARTIFACT_DIR}/enriched_cols.json") as f:
    enriched_cols = json.load(f)

with open(f"{ARTIFACT_DIR}/baseline_cols.json") as f:
    baseline_cols = json.load(f)

with open(f"{ARTIFACT_DIR}/llm_categorical_cols.json") as f:
    llm_categorical_cols = json.load(f)

with open(f"{ARTIFACT_DIR}/extraction_tool.json") as f:
    extraction_tool = json.load(f)

# Claude client
client = anthropic.Anthropic()

# Build the input form

st.set_page_config(page_title = "CFPB Complaint Investigation Predictor")

st.markdown("""
    <style>
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Will This Complaint Get Investigated?")
st.write("Paste a consumer complaint narrative and fill in a few details to see a live prediction.")

narrative = st.text_area("Complaint narrative", max_chars=2000, height=200)

col1 , col2 = st.columns(2)

with col1:
    product = st.text_input("Product (e.g. Credit Reporting)",
        help = "The type of financial product involved, e.g. 'Credit reporting', 'Mortgage', 'Credit card'")

    issue = st.text_input("Issue",
                           placeholder = "e.g. Incorrect information on your report",
                           help = "The general category of problem, as used in the CFPB complaint database")

    sub_issue = st.text_input("Sub-issue",
                           placeholder="e.g. Information belongs to someone else",
                           help="A more specific category under Issue")

    company = st.text_input("Company",
                            placeholder="e.g. Experian, Equifax, Bank of America")


with col2:
    state = st.text_input("State (2 Letter code)",
                           placeholder="e.g. NY, CA, TX")
    has_tag = st.checkbox("Has tag")
    timely_response = st.checkbox("Timely response", value = True)

submitted = st.button("Predict")

# Call claude to extract the 5 LLM features
if submitted:
    if not narrative.strip():
        st.error("Please paste a complaint narrative before predicting")
    else:
        with st.spinner("Reading the complaint...."):
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                tools=[extraction_tool],
                tool_choice={"type": "tool", "name": "extract_complaint_features"},
                messages=[
                    {"role": "user", "content": f"Extract structured features from this CFPB complaint narrative:\n\n{narrative}"}
                ]
            )
            llm_features = response.content[0].input

        st.subheader("What Claude extracted")
        st.json(llm_features)

        # Build a single row dataframe combining structured inputs + LLM features
        row = {
            'product_clean': product,
            'Issue' : issue,
            'Sub-issue' : sub_issue,
            'Company' : company,
            'State' : state,
            'has_tag' : int(has_tag),
            'has_narrative' : 1,
            'timely_response' : int(timely_response),

        }
        row.update(llm_features)

        input_df = pd.DataFrame([row])

        # Target-encode the baseline categorical columns (same encoder from NB3)
        rename_map = {'Sub-issue' : 'Sub-Issue'}
        input_for_encoding = input_df[baseline_cols].rename(columns=rename_map)
        encoded_baseline = target_encoder.transform(input_for_encoding)
        encoded_baseline = encoded_baseline.rename(columns={'Sub-Issue': 'Sub-issue'})
        input_df[baseline_cols] = encoded_baseline.values

        # One-hot encode the LLM categorical fields
        input_df_encoded = pd.get_dummies(input_df, columns=llm_categorical_cols)

        # Align columns to match exactly what the model was trained on
        # (adds any missing dummy columns as 0, drops anything extra, fixes order)
        input_df_final = input_df_encoded.reindex(columns=enriched_cols, fill_value=0)

        # Predict
        probability = model.predict_proba(input_df_final)[:, 1][0]

        st.subheader("Prediction")
        st.metric("Probability of investigation", f"{probability:.1%}")