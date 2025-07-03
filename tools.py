from langchain_core.tools import tool
from typing import Any, List, Dict
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
import itertools
import pandas as pd
import os
import requests
import json
import time
import warnings
from financial_data_loader import parse_sec_facts_json_to_dataframe
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser, StrOutputParser

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def get_all_metrics_list():
    filepath = "./Data/all_metrics.txt"

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        metrics = [line.strip() for line in f if line.strip()]  # Read and clean each line
    return ", ".join(metrics)


def get_metric_name_for_ciks(ciks):
    """
    Reads metric files for given CIKs and returns a dictionary mapping each CIK to its corresponding metrics.

    Args:
        ciks (list): List of CIK strings.

    Returns:
        dict: Dictionary with CIKs as keys and corresponding metrics as comma-separated strings.
    """
    folder_path = "Data/company_metrics_lists"
    metrics_dict = {}

    for cik in ciks:
        # Generate the file name with leading zeros
        file_name = f"{'0' * (10 - len(cik))}{cik}_metrics.txt"
        file_path = os.path.join(folder_path, file_name)


        if os.path.exists(file_path):
            try:
                # Read the file and extract metrics
                with open(file_path, 'r', encoding='utf-8') as f:
                    metrics = [line.strip() for line in f if line.strip()]  # Clean up lines

                # Combine metrics into a comma-separated string and update the dictionary
                metrics_dict[cik] = ", ".join(metrics)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    return metrics_dict

@tool
def get_financial_values(
    processed_data_df: pd.DataFrame,
    metric_name: str,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, Any]:
    """
    Extracts the 'value' column for a specific metric within the date range from a pre-filtered DataFrame.

    Args:
        processed_data_df (pd.DataFrame): The pre-filtered DataFrame containing the financial data.
        metric_name (str): The name of the metric to filter.
        start_date (str, optional): Start date for filtering (inclusive).
        end_date (str, optional): End date for filtering (inclusive).

    Returns:
        List[Any]: List of values corresponding to the metric within the specified date range.
    """
    df_filtered = processed_data_df[
        processed_data_df['metric_name'].str.lower() == metric_name.lower()
    ]

    if start_date:
        start_dt = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered['end_date'] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered['end_date'] <= end_dt]

    return {
        "data": df_filtered['value'].tolist(), # List of numeric values
        "metric_name": metric_name, # The metric name from the input
        "start_date": start_date, # Include the input start_date
        "end_date": end_date # Include the input end_date
    }

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

def get_relevant_douments(query,k=10):
    """
    Get the top 'k' documents from the Chroma Vector DB using similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    persist_directory = 'chroma_db'
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    docs = vectordb.similarity_search(query, k=k)
    return docs

def get_all_company_ciks():
    """
    Extracts `cik_str` and `title` from a JSON file containing company data.
    
    Args:
        filepath (str): Path to the JSON file containing company data.
        
    Returns:
        List[dict]: A list of dictionaries with `cik_str` and `title`.
    """
    filepath='./Data/company_tickers.json'
    try:
        # Read the JSON file
        with open(filepath, 'r') as file:
            data = json.load(file)
        
        # Extract `cik_str` and `title`
        company_ciks = [
            {"cik_str": details["cik_str"], "title": details["title"]}
            for key, details in data.items()
        ]
        
        return company_ciks
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except KeyError as e:
        print(f"Error: Missing key in data - {e}")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON file")
        return []

def get_financial_data_from_sec(ciks):
    headers = {'User-Agent':"ashtertiary@gmail.com"}
    folder_name = "Data/Data Frames"
    os.makedirs(folder_name, exist_ok=True)
    print(f"Ensured '{folder_name}' directory exists.")
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

def get_metric_values_from_dataframe(
    processed_data_df: pd.DataFrame,
    metric_name: str,
    start_date: str = None,
    end_date: str = None
) -> List[Any]:
    """
    Extracts the 'value' column for a specific metric within the date range from a pre-filtered DataFrame.

    Args:
        processed_data_df (pd.DataFrame): The pre-filtered DataFrame containing the financial data.
        metric_name (str): The name of the metric to filter.
        start_date (str, optional): Start date for filtering (inclusive).
        end_date (str, optional): End date for filtering (inclusive).

    Returns:
        List[Any]: A flat list of values corresponding to the metric within the specified date range.
    """
    # Filter the DataFrame by metric_name
    df_filtered = processed_data_df[
        processed_data_df['metric_name'].str.lower() == metric_name.lower()
    ]

    # Filter by start_date, if provided
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df_filtered = df_filtered[df_filtered['end_date'] >= start_dt]

    # Filter by end_date, if provided
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_filtered[df_filtered['end_date'] <= end_dt]

    # Ensure 'value' column is flattened into a simple list
    return df_filtered['value'].values.tolist()

def get_metric_for_all_ciks(result, k=3):
    """
    Extracts the first k specified metrics with data available for all companies and organizes them in a dictionary format.

    Args:
        result: An object containing:
                - metrics: List of metrics to extract
                - ciks: List of CIKs corresponding to companies
                - companies: List of company names
                - start_date: Start date for filtering
                - end_date: End date for filtering
        k: Maximum number of metrics to return (default is 5)
        
    Returns:
        dict: A nested dictionary in the form:
              {'metric_name': {'company_name': data, 'company_name': data, ...}, ...}
    """
    import os
    import pandas as pd

    folder_name = 'Data/Data Frames'
    datas = {}
    metric_count = 0  # Counter for included metrics

    for metric in result.metrics:
        metric_data = {}  # To hold data for this specific metric
        data_available = False  # Flag to check if data is available for any company

        for cik, name in zip(result.ciks, result.companies):
            padded_cik = str(cik).zfill(10)
            parquet_path = os.path.join(folder_name, f'{padded_cik}_financial_data.parquet')

            try:
                # Read parquet file
                df = pd.read_parquet(parquet_path)
                
                # Get metric values
                data = get_metric_values_from_dataframe(df, metric, result.start_date, result.end_date)
                
                if data:  # Check if data is not empty
                    metric_data[name] = data
                    data_available = True  # Set flag if at least one company has data

            except FileNotFoundError:
                print(f"File not found: {parquet_path}")
            except Exception as e:
                print(f"Error processing {name} with CIK {cik}: {e}")

        # Add this metric's data to the main dictionary if data is available
        if data_available:
            datas[metric] = metric_data
            metric_count += 1

        # Stop if we have reached the desired number of metrics
        if metric_count >= k:
            break

    return datas

def optimize_holt_forecast(data, periods=3):
    """
    Optimizes forecasting by finding the best alpha and beta parameters for Holt's Linear Trend method.
    
    Args:
        data (list or np.ndarray): Time series data.
        periods (int): Number of future values to predict.
        
    Returns:
        dict: A dictionary containing:
            - forecast: List of forecasted values.
            - smoothing_level: Optimized smoothing level (alpha).
            - smoothing_trend: Optimized smoothing trend (beta).
    """
    data = np.array(data)
    best_score = float('inf')
    best_params = None

    # Grid search for optimal alpha and beta
    for alpha in np.arange(0.1, 1.0, 0.1):
        for beta in np.arange(0.1, 1.0, 0.1):
            try:
                model = Holt(data).fit(
                    smoothing_level=alpha, 
                    smoothing_trend=beta,
                    optimized=False
                )
                forecast = model.forecast(1)
                error = mean_squared_error([data[-1]], forecast)
                
                if error < best_score:
                    best_score = error
                    best_params = {'smoothing_level': alpha, 'smoothing_trend': beta}
            except Exception as e:
                continue

    # If no parameters are found, fall back to default
    if best_params is None:
        best_params = {'smoothing_level': 0.5, 'smoothing_trend': 0.5}
    
    # Forecast using the best parameters
    model = Holt(data).fit(**best_params, optimized=False)
    forecast = model.forecast(periods).tolist()
    return {
        'forecast': forecast,
        'smoothing_level': best_params['smoothing_level'],
        'smoothing_trend': best_params['smoothing_trend']
    }

def auto_arima_forecast(data, periods=3, max_p=3, max_d=2, max_q=3):
    """
    Automatically determines the best ARIMA order and forecasts future values.

    Args:
        data (list or np.ndarray): Time series data.
        periods (int): Number of future values to predict.
        max_p (int): Maximum number of AR terms to consider.
        max_d (int): Maximum number of differences to consider.
        max_q (int): Maximum number of MA terms to consider.

    Returns:
        dict: A dictionary containing:
            - forecast: List of forecasted values.
            - order: Best (p, d, q) parameters.
            - aic: Akaike Information Criterion for the fitted model.
            - bic: Bayesian Information Criterion for the fitted model.
    """
    warnings.filterwarnings("ignore", category=UserWarning, message="Non-invertible starting MA parameters found.")
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels.tsa.statespace.sarimax")
    data = np.array(data)
    
    best_aic = float("inf")
    best_bic = float("inf")
    best_order = None
    best_model = None
    
    # Iterate over all combinations of (p, d, q)
    for p, d, q in itertools.product(range(max_p + 1), range(max_d + 1), range(max_q + 1)):
        try:
            model = ARIMA(data, order=(p, d, q))
            fitted_model = model.fit()
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            # Update the best model if the current one is better
            if aic < best_aic:
                best_aic = aic
                best_bic = bic
                best_order = (p, d, q)
                best_model = fitted_model
        except Exception:
            continue

    if best_model is None:
        raise ValueError("Could not fit any ARIMA models. Please check your data.")
    
    # Forecast using the best model
    forecast = best_model.forecast(steps=periods).tolist()
    
    return {
        'forecast': forecast,
        'order': best_order,
        'aic': best_aic,
        'bic': best_bic
    }

def process_and_forecast(D, periods=3):
    """
    Applies the optimize_holt_forecast and auto_arima_forecast functions to each 'company' key in the dictionary.

    Args:
        D (dict): Input dictionary with structure:
                  {
                      'key': {'company': [values]},
                      'another_key': {'company': [values]},
                      ...
                  }
        periods (int): Number of periods to forecast.

    Returns:
        tuple: (
            full_results: {
                'holt': {metric: {company: {forecasted_values, smoothing_level, smoothing_trend}}},
                'arima': {metric: {company: {forecasted_values, order, aic, bic}}}
            },
            simplified_results: {
                'holt': {metric: {company: forecasted_values}},
                'arima': {metric: {company: forecasted_values}}
            }
        )
    """
    holt_result = {}
    arima_result = {}
    simplified = {'holt': {}, 'arima': {}}

    for metric, companies in D.items():
        holt_result[metric] = {}
        arima_result[metric] = {}
        simplified['holt'][metric] = {}
        simplified['arima'][metric] = {}

        for company, values in companies.items():
            if values:  # Ensure the list is not empty
                # Holt Forecast
                holt_forecast = optimize_holt_forecast(values, periods)
                holt_result[metric][company] = {
                    'forecasted_values': holt_forecast['forecast'],
                    'smoothing_level': holt_forecast['smoothing_level'],
                    'smoothing_trend': holt_forecast['smoothing_trend']
                }
                simplified['holt'][metric][company] = holt_forecast['forecast']

                # ARIMA Forecast
                arima_forecast_result = auto_arima_forecast(values, periods)
                arima_result[metric][company] = {
                    'forecasted_values': arima_forecast_result['forecast'],
                    'order': arima_forecast_result['order'],
                    'aic': arima_forecast_result['aic'],
                    'bic': arima_forecast_result['bic']
                }
                simplified['arima'][metric][company] = arima_forecast_result['forecast']

    full_results = {'holt': holt_result, 'arima': arima_result}
    return full_results, simplified

def execute_python_code(code_string: str):
    """
    Executes a given Python code string and captures stdout and stderr for debugging.

    Args:
        code_string (str): The Python code to execute. It is expected to handle its own outputs (e.g., saving files, writing markdown, etc.).
    """
    # Backup original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()

    # Redirect stdout and stderr
    sys.stdout = redirected_output
    sys.stderr = redirected_error

    try:
        # Execute the provided code string
        exec(code_string, {})  # Empty global dictionary for security

    except Exception as e:
        # Print exception encountered during execution
        print(f"Error during code execution: {e}")
        print("Generated Code Output (stdout):")
        print(redirected_output.getvalue())
        print("Generated Code Errors (stderr):")
        print(redirected_error.getvalue())

    finally:
        # Restore original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    # Print captured stdout and stderr for review
    output = redirected_output.getvalue()
    error = redirected_error.getvalue()
    if output:
        print("Execution Output (stdout):")
        print(output)
    if error:
        print("Execution Errors (stderr):")
        print(error)

def plot_and_save_forecasts(data, holt_data, arima_data, output_folder='Reports/Predictions'):
    """
    Generates and saves plots for each metric with historical, Holt, and ARIMA forecasted data for each company.

    Args:
        data (dict): Dictionary containing historical data in the format:
                     {'metric': {'company': [historical_values]}}
        holt_data (dict): Dictionary containing Holt forecasted data and parameters.
                     {'metric': {'company': {'forecasted_values': [...], 'smoothing_level': ..., 'smoothing_trend': ...}}}
        arima_data (dict): Dictionary containing ARIMA forecasted data and parameters.
                     {'metric': {'company': {'forecasted_values': [...], 'order': (p, d, q), 'aic': ..., 'bic': ...}}}
        output_folder (str): Directory to save the plots.

    Returns:
        list: List of filenames of the saved images.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    saved_images = []  # List to store the filenames of saved images
    
    for metric, companies in data.items():
        plt.figure(figsize=(12, 8))  # Create a new figure for each metric
        for company, historical_values in companies.items():
            # Holt data
            holt_forecast = holt_data.get(metric, {}).get(company, {})
            holt_values = holt_forecast.get('forecasted_values', [])
            holt_params = f"(α={holt_forecast.get('smoothing_level', 'N/A')}, β={holt_forecast.get('smoothing_trend', 'N/A')})"
            
            # ARIMA data
            arima_forecast = arima_data.get(metric, {}).get(company, {})
            arima_values = arima_forecast.get('forecasted_values', [])
            arima_order = arima_forecast.get('order', 'N/A')
            arima_aic = arima_forecast.get('aic', 'N/A')
            arima_bic = arima_forecast.get('bic', 'N/A')

            # Total lengths for plotting
            total_length_holt = len(historical_values) + len(holt_values)
            total_length_arima = len(historical_values) + len(arima_values)

            # Plot historical data
            plt.plot(range(len(historical_values)), historical_values, label=f"{company} (Historical)", marker='o')
            
            # Plot Holt forecasted data
            if holt_values:
                plt.plot(range(len(historical_values), total_length_holt), holt_values, label=f"{company} (Holt {holt_params})", linestyle='--', marker='x')

            # Plot ARIMA forecasted data
            if arima_values:
                plt.plot(range(len(historical_values), total_length_arima), arima_values, label=f"{company} (ARIMA {arima_order}, AIC={arima_aic:.2f}, BIC={arima_bic:.2f})", linestyle='-.', marker='s')
        
        # Add plot details
        plt.title(f"{metric} Forecast with Holt and ARIMA")
        plt.xlabel("Time")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        
        # Save the figure
        filename = f"{metric.replace(' ', '_')}_forecast_combined.png"
        output_path = os.path.join(output_folder, filename)
        plt.savefig(output_path)
        saved_images.append(filename)  # Add the filename to the list
        plt.close()  # Close the figure to avoid overlap
    
    print(f"Combined plots saved in the '{output_folder}' folder.")
    return saved_images  # Return the list of filenames

def execute_code(code_str: str):
    """
    Executes the given Python code iteratively until no error is encountered.

    Args:
        code_str (str): The initial Python code to execute.

    Returns:
        dict: A dictionary containing:
              - 'final_output': The final stdout after successful execution.
              - 'final_code': The corrected code that was successfully executed.
    """
    max_iterations = 5  # Prevent infinite loops with a maximum iteration limit
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"Attempt {iteration}: Executing code...")

        # Execute the code and capture output and errors
        result = execute_python_code(code_str)
        executed, err, output = result["executed"], result["error"], result["output"]

        # If no error, return the final result
        if executed and not err:
            print("Code executed successfully!")
            return {"final_output": output, "final_code": code_str}

        # Log the error and invoke LLM for correction
        print(f"Error detected: {err}. Verifying and correcting the code...")
        verified_code = verify_chain.invoke({"code": code_str, "error": err})


        # Update the code for the next iteration
        code_str = verified_code

    # If max iterations are reached
    raise RuntimeError("Max iterations reached. Unable to resolve the error.")

code_corrector_prompt = PromptTemplate(
    template="""
    You are an experienced coder with expertise in debugging Python code, especially in the following areas:
    - ReportLab PDF generation.
    - Matplotlib plotting.
    - Resolving basic syntax, logical, and formatting errors.

    **Your Task:**
    - Identify and correct errors in the provided code while preserving its functionality and purpose.
    - 
    - Ensure the code is **error-free**, **executable**, and adheres to Python best practices.

    **Common Errors and How to Resolve Them:**
    1. **Unfinished or Mismatched Literals:**
       - Unmatched brackets, braces, or parentheses (e.g., '(', '[', '{{').
         **Resolution:** Ensure that every opening literal has a corresponding closing literal in the correct location.
       - Unterminated string literals (e.g., a string missing closing quotation marks).
         **Resolution:** Close all open string literals with appropriate quotes.
       - Errors like "{{ was never closed" or "[ was never closed."
         **Resolution:** Identify and properly close unmatched literals at the mentioned line or in the surrounding code.

    2. **Syntax and Execution Errors:**
       - Issues with f-strings, such as unmatched `(` or `{{`.
         **Resolution:** Ensure that all placeholders in f-strings are properly enclosed and formatted.
       - JSON-related issues (e.g., `json.decoder.JSONDecodeError: Invalid \\escape`).
         **Resolution:** Correct improperly escaped characters, ensuring valid JSON syntax (e.g., use `\\` for backslashes or `"` for strings).
       - Errors like `Invalid \\escape: line 2 column 8665`.
         **Resolution:** Replace invalid escape sequences with valid alternatives.

    3. **Indentation Errors:**
       - Errors like "unexpected indent" or "indentation error."
         **Resolution:** Align all indents consistently using spaces or tabs (preferably spaces, following PEP 8 guidelines). Verify nesting levels.

    4. **Type-Related Errors:**
       - Errors like `TypeError: object of type 'float' has no len()`.
         **Resolution:** Check variable types before performing operations. For example:
           - If a float is passed where a sequence is expected, convert it to a list or handle it differently.

    5. **General Debugging:**
       - Spelling mistakes in variable or function names (e.g., `include_risk_analysi`).
         **Resolution:** Correct spelling errors based on context or user intent.
       - Undefined variable or function names (e.g., `name 'include_vis' is not defined`).
         **Resolution:** Define the missing variable or function with a reasonable default value (e.g., `include_vis = True/False`).

    **Requirements:**
    - Ensure functionality is preserved; do not change the intended behavior of the code.
    - Avoid unnecessary text, explanations, or comments unrelated to debugging.
    - Return only the corrected code.

    **Additional Guidelines:**
    - Include a success message indicating the file has been saved, such as:
      `"File saved successfully at: <file_path>"`
    - Double-check for errors introduced during debugging, ensuring there are no typos or logical mistakes.
    - Provide a complete and executable Python script.

    **Input Details:**
    CODE:
    {code}

    FOUND ERROR:
    {error}

    **If the input code contains errors, correct them and return the complete, fixed code.**
    **The line at which the error occurs is already mentioned in the input error**

    **Output:**
    - Return only the corrected Python code. No additional explanations or comments.
    """,
    input_variables=["code", "error"]
)


verify_chain = code_corrector_prompt | model | StrOutputParser()