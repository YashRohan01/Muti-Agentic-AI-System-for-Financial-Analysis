from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vector_store import initialize_vector_store


from typing import Dict, List, Any
import pandas as pd
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
persist_directory = 'chroma_db'

def get_financial_metric(
    processed_data_df: pd.DataFrame, # Pass the DataFrame as an argument
    company_name : str,
    metric_name : str,
    start_date : str = None,
    end_date : str = None,
    form_type : str = None
) -> List[Dict[str,Any]]:
    """
    Retrieves financial data for a specific metric for a given company from a processed DataFrame.
    Filters by date range and form type if provided.
    """
    if processed_data_df.empty:
        # print("Error: Input DataFrame for get_financial_metric is empty.") # Suppress for cleaner output within generate_financial_docs
        return []

    df_filtered = processed_data_df[
        (processed_data_df['company_name'].str.lower() == company_name.lower()) &
        (processed_data_df['metric_name'].str.lower() == metric_name.lower())
    ].copy()

    if start_date:
        try:
            start_dt = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered['end_date'] >= start_dt]
        except ValueError:
            pass # print(f"Warning: Invalid start_date format '{start_date}'. Skipping filter.")

    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered['end_date'] <= end_dt]
        except ValueError:
            pass # print(f"Warning: Invalid end_date format '{end_date}'. Skipping filter.")

    if form_type:
        df_filtered = df_filtered[df_filtered['form_type'].str.lower() == form_type.lower()]

    df_filtered = df_filtered.sort_values(by='end_date').reset_index(drop=True)
    return df_filtered.to_dict(orient='records')

def calculate_yoy_growth(
    processed_data_df: pd.DataFrame,
    company_name: str,
    metric_name: str,
    period_type: str = 'yearly'
) -> List[Dict[str, Any]]: # Change return type to list of dicts
    """
    Calculates year-over-year (or quarter-over-quarter) growth for a given metric.
    Returns a list of dictionaries, each summarizing growth for a period.
    """
    data = get_financial_metric(processed_data_df, company_name, metric_name)
    if not data:
        return [] # Return empty list for no data

    df = pd.DataFrame(data)
    df['end_date'] = pd.to_datetime(df['end_date'])
    df['period'] = df['end_date'].dt.to_period('Q' if period_type == 'quarterly' else 'Y')

    df = df.sort_values(by=['period', 'filed_date'], ascending=[True, False]).drop_duplicates(subset=['period'], keep='first')
    df = df.sort_values(by='period').set_index('period')

    if len(df) < 2:
        return [] # Return empty list for not enough data

    growth_details = []
    for i in range(1, len(df)):
        current_period = str(df.index[i])
        previous_period = str(df.index[i-1])
        current_value = df['value'].iloc[i]
        previous_value = df['value'].iloc[i-1]

        detail = {
            "current_period": current_period,
            "previous_period": previous_period,
            "current_value": current_value,
            "previous_value": previous_value,
            "growth_percentage": None, # Initialize growth percentage
            "notes": ""
        }

        if pd.isna(current_value) or pd.isna(previous_value):
            detail["notes"] = "Missing value (NaN)."
        elif previous_value != 0:
            growth = ((current_value - previous_value) / previous_value) * 100
            detail["growth_percentage"] = growth
        else:
            detail["notes"] = "Zero previous value."
        growth_details.append(detail)

    return growth_details


CHUNK_SIZE = 4096
CHUNK_OVERLAP = 512
COMPANY_YEAR_PATTERN = r"(\w+)-(\d{4})\.pdf"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data/10K Reports")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
REPORT_PATH = os.path.join(BASE_DIR, "Reports")

def embedd_reports():
    print("=" * 50)
    print("📚 SEC 10-K Document Embedding System")
    print("=" * 50)
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        print("Please create the directory and add your company folders containing 10-K PDFs")
        return
    
    print(f"🔍 Checking for PDFs in company-specific folders under {DATA_DIR}...")
    pdf_files = []
    
    for company_folder in os.listdir(DATA_DIR):
        company_path = os.path.join(DATA_DIR, company_folder)
        if not os.path.isdir(company_path):
            print(f"⚠️ Skipping non-folder item: {company_folder}")
            continue
        
        company_pdfs = [
            os.path.join(company_folder, f)
            for f in os.listdir(company_path)
            if f.endswith(".pdf")
        ]
        pdf_files.extend(company_pdfs)
    
    if not pdf_files:
        print("❌ No PDF files found. Please add 10-K documents to the company-specific folders.")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF documents across {len(os.listdir(DATA_DIR))} company folders")
    print("🛠️ Starting embedding process...")
    
    initialize_vector_store()
    
    print("=" * 50)
    print("✅ Embedding completed successfully!")
    print("=" * 50)

if __name__=='__main__':
    embedd_reports()
    