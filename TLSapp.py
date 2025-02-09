import streamlit as st
import pickle
import json
import pandas as pd
import numpy as np
import spacy
import pytesseract
import cv2
import re
import fitz  # PyMuPDF
from transformers import pipeline
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tempfile
import os

# -----------------------------
# Load Models and Initialize
# -----------------------------

# Load SpaCy model for NLP
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    st.error(f"Error loading spaCy model: {e}")

# Initialize transformers pipelines for zero-shot classification
try:
    classifier = pipeline("zero-shot-classification")
    fraud_detector = pipeline("zero-shot-classification")
except Exception as e:
    st.error(f"Error initializing transformer pipelines: {e}")

# Set Tesseract command path (adjust for your system, if needed)
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # For Linux/Mac; for Windows use full path like r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Predefined list of potentially fraudulent terms
fraud_keywords = [
    "undisclosed payment", "false representation", "forgery", "unilateral termination",
    "penalty for early termination", "hidden fees", "exclusive rights", "secret payment terms"
]

# -----------------------------
# Define Utility Functions
# -----------------------------

def extract_text_from_image(image_file):
    """Extract text from scanned agreement using OCR."""
    image = Image.open(image_file)
    image = image.convert('RGB')  # Ensure correct format for OCR
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF agreement using PyMuPDF with a temporary file."""
    text = ""
    # Write the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.getbuffer())
        temp_pdf_path = temp_pdf.name

    # Open the temporary PDF file with PyMuPDF
    doc = fitz.open(temp_pdf_path)
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()  # Close the document to release the file

    # Remove the temporary file
    os.remove(temp_pdf_path)
    return text

def categorize_agreement(text):
    """Categorize agreement type using NLP."""
    categories = ["NDA", "SLA", "Vendor Agreement", "Licensing Deal", "Contract"]
    result = classifier(text, categories)
    return result['labels'][0]

def detect_fraud_in_contract(text):
    """Detect potential fraud in contract text using AI and keyword matching."""
    fraud_indicators = []
    # Check for suspicious keywords
    for keyword in fraud_keywords:
        if keyword.lower() in text.lower():
            fraud_indicators.append(f"Suspicious Keyword: {keyword}")
    # Use zero-shot classification to detect fraud context
    categories = ["fraud", "legitimate"]
    result = fraud_detector(text, candidate_labels=categories)
    if result['labels'][0] == "fraud" and result['scores'][0] > 0.5:
        fraud_indicators.append("Suspicious clause detected by AI: fraud")
    if not fraud_indicators:
        return "No fraud detected."
    return fraud_indicators

def extract_cash_flows(text):
    """Extract cash flow details using regex patterns."""
    amounts = re.findall(r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?|\d+ USD|\d+ dollars', text, re.IGNORECASE)
    dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b', text)
    
    if not amounts or not dates:
        st.warning("⚠️ Warning: No cash flow details detected in the contract.")
    
    return list(zip(amounts, dates))

def financial_impact_analysis(cash_flows):
    """Estimate potential financial impact from cash flows."""
    total_cash_flow = 0
    for amount, _ in cash_flows:
        try:
            amt = float(amount.replace("$", "").replace(",", ""))
            total_cash_flow += amt
        except Exception as e:
            continue
    return total_cash_flow

def risk_assessment(text):
    """Analyze agreement for potential legal and financial risks."""
    risk_keywords = ["penalty", "liquidated damages", "breach", "non-compliance", "audit"]
    risks = [word for word in risk_keywords if word in text.lower()]
    return risks

def track_renewal_dates(text):
    """Extract and track renewal or expiration dates from contracts."""
    renewal_dates = re.findall(r'(renewal|expiration|end)\s+date.*?(\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    if not renewal_dates:
        renewal_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
        renewal_dates = [("Default Renewal Date", renewal_date)]
    return renewal_dates

def scenario_analysis(cash_flows, discount_rate=0.05):
    """Perform financial scenario analysis using NPV calculations."""
    npv = 0
    for i, (amount, _) in enumerate(cash_flows, 1):
        try:
            amt = float(amount.replace("$", "").replace(",", ""))
            npv += amt / ((1 + discount_rate) ** i)
        except Exception as e:
            continue
    return npv

def visualize_contract_data(cash_flows, categories):
    """Generate visualizations of contract cash flows and categories."""
    try:
        cash_flow_values = [float(amount.replace("$", "").replace(",", "")) for amount, _ in cash_flows]
    except Exception as e:
        st.error("Error processing cash flow amounts for visualization.")
        cash_flow_values = []

    categories_count = {category: categories.count(category) for category in set(categories)}
    
    # Plot contract categories distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(categories_count.keys()), y=list(categories_count.values()), ax=ax1)
    ax1.set_title('Contract Categories Distribution')
    ax1.set_xlabel('Contract Categories')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

    # Plot cash flows over time if available
    if cash_flow_values:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.lineplot(x=list(range(1, len(cash_flow_values) + 1)), y=cash_flow_values, ax=ax2)
        ax2.set_title('Cash Flows Over Time')
        ax2.set_xlabel('Sequential Payment Number')
        ax2.set_ylabel('Cash Flow ($)')
        st.pyplot(fig2)

def calculate_risk_score(risks):
    """Calculate risk score based on identified risks."""
    risk_score = len(risks)  # Each risk identified adds 1 point
    return risk_score

def calculate_revenue_potential(cash_flows):
    """Calculate revenue potential based on total cash flows."""
    total_revenue = 0
    for amount, _ in cash_flows:
        try:
            amt = float(amount.replace("$", "").replace(",", ""))
            total_revenue += amt
        except Exception as e:
            continue
    revenue_score = min(total_revenue / 1000000 * 40, 40)  # Scale and cap at 40
    return revenue_score

def calculate_financial_viability(npv_value):
    """Calculate financial viability based on NPV."""
    if npv_value > 0:
        return 20
    else:
        return -20

def calculate_contract_intelligence_score(risk_score, revenue_potential, financial_viability):
    """Calculate the Contract Intelligence Score (CIS)."""
    risk_weight = 0.4
    revenue_weight = 0.4
    viability_weight = 0.2
    cis = 100 - (risk_weight * (risk_score * 10)) + (revenue_weight * revenue_potential) + (viability_weight * financial_viability)
    cis = max(0, min(cis, 100))
    return cis

# -----------------------------
# Load External Model and Version History
# -----------------------------

model_filename = "agreement_analysis_model.pkl"
version_history_filename = "version_history.json"

# Load the external model (if required)
try:
    with open(model_filename, "rb") as f:
        analysis_model = pickle.load(f)
except Exception as e:
    st.warning(f"Could not load external model from {model_filename}: {e}")
    analysis_model = None

# Load version history or create an empty dictionary
if os.path.exists(version_history_filename):
    with open(version_history_filename, "r") as f:
        version_history = json.load(f)
else:
    version_history = {}

# -----------------------------
# Streamlit App Layout
# -----------------------------

st.set_page_config(page_title="Contract Intelligence", layout="wide")
st.title("Contract Intelligence Web App")

# Custom CSS for better visualization
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        font-weight: 500;
        color: #2196F3;
    }
    .container {
        margin: 30px 0;
    }
    .card {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 20px;
        background-color: #FAFAFA;
    }
    .card-header {
        font-size: 18px;
        font-weight: 600;
    }
    .card-body {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a PDF or Image file", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file:
    st.markdown('<div class="container"><div class="card"><div class="card-header">Uploaded Document</div><div class="card-body">', unsafe_allow_html=True)
    st.write("File Name: ", uploaded_file.name)
    st.markdown('</div></div></div>', unsafe_allow_html=True)

    # Process the document based on its extension
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        agreement_text = extract_text_from_pdf(uploaded_file)
    elif file_extension in ["jpg", "jpeg", "png"]:
        agreement_text = extract_text_from_image(uploaded_file)
    else:
        st.error("Unsupported file type.")
        agreement_text = ""

    if agreement_text:
        # Agreement Category
        category = categorize_agreement(agreement_text)
        st.markdown(f"### Agreement Category: **{category}**")

        # Fraud Detection
        fraud_results = detect_fraud_in_contract(agreement_text)
        st.markdown("### Fraud Detection")
        if fraud_results == "No fraud detected.":
            st.success(fraud_results)
        else:
            st.warning("\n".join(fraud_results))

        # Cash Flow Extraction
        cash_flows = extract_cash_flows(agreement_text)
        st.markdown("### Extracted Cash Flows")
        if cash_flows:
            st.write(pd.DataFrame(cash_flows, columns=["Amount", "Date"]))
        else:
            st.info("No cash flow details found.")

        # Financial Impact Analysis
        financial_impact = financial_impact_analysis(cash_flows)
        st.markdown(f"### Financial Impact: **${financial_impact:.2f}**")

        # Risk Assessment
        risks = risk_assessment(agreement_text)
        st.markdown("### Risk Assessment")
        if risks:
            st.warning("\n".join(risks))
        else:
            st.success("No significant risks detected.")

        # Renewal Dates Tracking
        renewal_dates = track_renewal_dates(agreement_text)
        st.markdown("### Renewal Dates")
        st.write(renewal_dates)

        # Scenario Analysis (NPV)
        npv_value = scenario_analysis(cash_flows)
        st.markdown(f"### Net Present Value (NPV): **${npv_value:.2f}**")

        # Calculate Contract Intelligence Score (CIS)
        risk_score = calculate_risk_score(risks)
        revenue_potential = calculate_revenue_potential(cash_flows)
        financial_viability = calculate_financial_viability(npv_value)
        cis = calculate_contract_intelligence_score(risk_score, revenue_potential, financial_viability)
        st.markdown(f"### Contract Intelligence Score (CIS): **{cis:.2f}**")

        # Visualize Contract Data
        visualize_contract_data(cash_flows, [category] * len(cash_flows))

        # Update version history and save to JSON file
        version_history[uploaded_file.name] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(version_history_filename, "w") as f:
            json.dump(version_history, f)
        st.markdown("### Version History")
        st.write(version_history)

# About Section
st.markdown("""
    <div class="container">
        <div class="card">
            <div class="card-header">About the App</div>
            <div class="card-body">
                This web application analyzes uploaded agreements and provides insights related to fraud detection, financial impact, risks, renewal dates, and an overall Contract Intelligence Score (CIS).
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)