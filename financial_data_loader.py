import requests
import json
import os
import re
import time # Import the time module for delays
import pandas as pd
from typing import Dict, Any, Optional, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


CHUNK_SIZE = 4096
CHUNK_OVERLAP = 512
COMPANY_YEAR_PATTERN = r"(\w+)-(\d{4})\.pdf"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data/10K Reports")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
REPORT_PATH = os.path.join(BASE_DIR, "Reports")

headers = {'User-Agent':"ashtertiary@gmail.com"}
folder_name = "Data/Data Frames"
os.makedirs(folder_name, exist_ok=True)
print(f"Ensured '{folder_name}' directory exists.")


def parse_sec_facts_json_to_dataframe(sec_json_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parses a typical SEC EDGAR company facts JSON response into a Pandas DataFrame.

    Args:
        sec_json_data (Dict[str, Any]): The raw JSON dictionary obtained from the SEC API.

    Returns:
        pd.DataFrame: A DataFrame containing normalized financial facts.
                      Returns an empty DataFrame if parsing fails or no data.
    """
    all_records: List[Dict[str, Any]] = []

    company_cik = sec_json_data.get('cik')
    company_name = sec_json_data.get('entityName')

    facts = sec_json_data.get('facts', {})
    us_gaap_facts = facts.get('us-gaap', {})

    for metric_name, metric_details in us_gaap_facts.items():
        # Get label and description for the metric, useful for context
        label = metric_details.get('label', metric_name)
        description = metric_details.get('description', '')

        if 'units' in metric_details:
            for unit_type, entries in metric_details['units'].items():
                for entry in entries:
                    record = {
                        'company_cik': company_cik,
                        'company_name': company_name,
                        'metric_name': metric_name,
                        'metric_label': label,
                        'metric_description': description,
                        'unit': unit_type,
                        'value': entry.get('val'),
                        'end_date': entry.get('end'), # The end date of the reporting period
                        'fiscal_year': entry.get('fy'),
                        'fiscal_period': entry.get('fp'),
                        'form_type': entry.get('form'), # e.g., '10-K', '10-Q'
                        'filed_date': entry.get('filed'), # The date the filing was made
                        'accession_number': entry.get('accn'), # Unique identifier for the filing
                        'period_frame': entry.get('frame') # e.g., 'CY2023Q4', 'CY2023'
                    }
                    all_records.append(record)

    df = pd.DataFrame(all_records)

    # Convert date columns to datetime objects for easier filtering and sorting
    if not df.empty:
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        df['filed_date'] = pd.to_datetime(df['filed_date'], errors='coerce')
        # Ensure value is numeric
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
 

    return df



def get_financial_data(ciks):
    for cik in ciks:
        padded_cik = str(cik).zfill(10) # Pad with leading zeros to 10 digits
        api_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{padded_cik}.json"

        print(f"\n--- Attempting to fetch data for CIK: {cik} ---")

        try:
            responses = requests.get(api_url, headers=headers)
            responses.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            print(f"Successfully fetched data for CIK = {cik}")

            data = responses.json()
            print(f"Converted response to JSON for CIK = {cik}")  
            df =  parse_sec_facts_json_to_dataframe(data)
            df.to_parquet(os.path.join(folder_name,f'{padded_cik}_financial_data.parquet'), index=False)

        except requests.exceptions.HTTPError as errh:
            print(f"HTTP Error for CIK {cik}: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Error Connecting for CIK {cik}: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout Error for CIK {cik}: {errt}")
        except requests.exceptions.RequestException as err:
            print(f"An unexpected error occurred during the request for CIK {cik}: {err}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response for CIK {cik}: {e}")
        except IOError as e:
            print(f"Error saving data to file for CIK {cik}: {e}")
        time.sleep(0.15)

def load_and_chunk_pdfs():
    """Load and split PDFs into chunks with metadata from company-specific directories"""
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n"]
    )
    
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"DATA_DIR '{DATA_DIR}' does not exist.")
    
    print(f"📂 Loading PDFs from company-specific folders in {DATA_DIR}...")
    for company_folder in os.listdir(DATA_DIR):
        company_path = os.path.join(DATA_DIR, company_folder)
        if not os.path.isdir(company_path):
            print(f"⚠️ Skipping non-folder item: {company_folder}")
            continue
        
        for filename in os.listdir(company_path):
            if filename.endswith(".pdf"):
                match = re.match(COMPANY_YEAR_PATTERN, filename)
                if not match:
                    print(f"⚠️ Skipping non-conforming file: {filename} in {company_folder}")
                    continue
                
                company, year = match.groups()
                print(f"🔍 Processing {company.capitalize()} {year} report from {company_folder}")
                file_path = os.path.join(company_path, filename)
                loader = PyPDFLoader(file_path)
                
                try:
                    pages = loader.load_and_split()
                except Exception as e:
                    print(f"❌ Error loading {filename} in {company_folder}: {e}")
                    continue
                
                for page in pages:
                    page.metadata.update({
                        "company": company.capitalize(),
                        "year": year,
                        "source": filename
                    })
                    documents.append(page)
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_documents([doc]))
    
    print(f"✅ Generated {len(chunks)} chunks from {len(documents)} pages")
    return chunks
