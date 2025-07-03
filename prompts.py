from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from tools import get_all_metrics_list, get_all_company_ciks

from parsers import metric_parser,rag_parser,task_parser,query_parser,data_parser, report_parser, visualizer_parser,financial_analysis_parser,predictive_analysis_parser, narrative_analysis_parser, strategic_recommendation_parser,rag_report_parser,risk_analysis_parser,concise_answer_parser, rag_concise_parser

query_prompt = PromptTemplate(
    template="""
    You are an expert financial query classifier agent. Your task is to classify a user's query as either "simple" or "complex".
    You have a deep understading of financial metrics like AccountPayable, Liabilities and so on.

    - A query is considered **simple** if it can be answered by fetching a basic financial metric (e.g., liabilities, revenue) for a specific company and year without requiring additional analysis or insights. 
      Example: "What were the liabilities for Apple in the year 2018?", "How did the Net Profit of Meta vs Micrsofot change from 2022-2024."

    - A query is considered **complex** if it requires advanced operations such as data visualization, narrative analysis, trend analysis, or future predictions. 
      Example: "What does the trend of Apple's liabilities over the last 5 years indicate?", "Show me Meta's profit using a bar chart and what does the future look like"

    - Narrative analysis is done when the query needs to state reasons as to what the trend signifies.

    - Financial Metric Analysis is done when the query requires getting information about a data. Always return at least 3 metrics that match requirement the best in decreasing order.

    - Predictive Analysis is done when future trends are required about a metric.

    - Also tell from the query what all ciks (for companies) are required to answer the query.

    USER QUERY:
    {query}

    GIVEN COMPANIES INFORMATION (Name and CIK for lookup, assume these are the full names):
    {companies}

    Based on the above criteria, classify the query as either "simple" or "complex" and provide a brief explanation for your classification.


    {format_instructions}
    """,
    input_variables=['query'],
    partial_variables={"format_instructions": query_parser.get_format_instructions(),"companies":get_all_company_ciks()}
)

rag_prompt = PromptTemplate(
    template="""
    You are a RAG agent who excels in understanding financial reports. Your task is to summarize the documents 
    for the given user query.
    
    -You will recieve a List of relevant documents (in decreasing order of their similarity from the query) and your job is to summarize the documents such that they can be written in a
    financial journal of the highest standards.

    - You have to give content for the final report, alongwith suitable headings and information.
    - Make the report professional with good formatting and proper points and several headings.
    - The name of the file should be appropriately decided by you.

    USER QUERY:
    {query}

    Relevant Documents:
    {docs}

    - Summarize the documents with all the relevant information and professional wordings.

    Strictly follow the specified format:
    {{
    code: python code generating and saving the report.
    }}
    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**


    {format_instructions}
    """,
    input_variables=['query','docs'],
    partial_variables={"format_instructions": rag_parser.get_format_instructions()}
)

rag_report_writing_prompt = PromptTemplate(
    template="""
  You are an experienced financial reporter with expertise in generating pdf reports using python reportlab.

  - You will get the summary of the 10K report for the specified user query.

  - You have to generate a professional Finacial Report using reportlab in python

  - The report should have several headings, should focus more on bullet points rather than long paragraphs.

  - It should contain tables to demonstrate data bor better understanding.

  - It should also include a 'Table of Contents' section at the beginning of the report which is a table again.

  - Include a success message at the end of the python code (not the report).

  - Save the pdf file in the './Reports/' folder.

  REPORT SUMMARY:
  {summary}

  USER QUERY:
  {query}

  """,
  input_variables=["query","summary"]
)

concise_rag_prompt = PromptTemplate(
    template="""
    You are an experienced financial reporter with expertise in summarizing the information recieved form 10K documents.

    - Your job is to give bullet points from the recieved documents and 10K report summary and suxh that the usdr query gets andwered.

    GIVEN USER QUERY :
    {query}

    REQUIRED DOCUMENTS:
    {documents}

    REQUIRED 10K SUMMARY:
    {summary}

    Please adhere to this structure:
    {{
      points: List[str] (Each string is a bullet point)
    }}

    **DO NOT INCLUDE DANGLING PERIODS**

    eg :
      {{
        "points": [
          "Challenges in managing frequent product and service transitions to remain
          competitive, including timely development, market acceptance, and production ramp-up.",
        .
          "Risk of new technologies leading to lower revenues and profit margins.",
        .
          "Compliance risks related to laws and regulations concerning technology,
          intellectual property, digital platforms, machine learning, AI, internet,
          telecommunications, media, labor, and environmental issues.",
        .
          "Failures in IT systems, increasing cybersecurity threats (including ransomware),
          and network disruptions impacting online services, customer transactions, and
          manufacturing.",
        .
          "Risks associated with high profile and the value of confidential information.",
        .
          "These risks are expected to accelerate in frequency and sophistication, especially
          during diplomatic or armed conflict."
        ]
      }} is wrong.

      {{
  "points": [
    "Challenges in managing frequent product and service transitions to remain competitive, including timely development, market acceptance, and production ramp-up.",
    "Risk of new technologies leading to lower revenues and profit margins.",
    "Compliance risks related to laws and regulations concerning technology, intellectual property, digital platforms, machine learning, AI, internet, telecommunications, media, labor, and environmental issues.",
    "Failures in IT systems, increasing cybersecurity threats (including ransomware), and network disruptions impacting online services, customer transactions, and manufacturing.",
    "Risks associated with high profile and the value of confidential information.",
    "These risks are expected to accelerate in frequency and sophistication, especially during diplomatic or armed conflict."
  ]is right

    {format_instructions}
    """,
    input_variables=["query","documents","summary"],
    partial_variables={"format_instructions":rag_concise_parser.get_format_instructions()}
)

task_allocating_prompt = PromptTemplate(
    template="""
    You are a task classifier, whose job is to decide what all tasks does the given complex query need. 

    - Financial Metric Analysis is required when questions about financial metrics like AccountPayable, Liabilities, Foreign Payments, etc are required
     along with some sort of visulaization. This is more than just a basic RAG search, it reuqires describing the trends, implications that can be drawn from it 
     an deciding the overall impact on the company on the given metric.

    - Predictive Analysis is required when the user query asks about the future trend of the given metric values.
      A basic time series forecasting method that will be able to forecast the method is used. The implications of this trend, the meaning of this trend and 
      the impact of this trend will also be inlcuded. 
      (NOTE: The current year is 2025, June).

    - Narrative Analysis is required when the user query talks about reasons or implications that are not purely financial in nature. These informations can be retireved from 10K forms.
      eg. What were the risks flagged for apple in 2019. (purely narrative)
      eg. How does the Apple's approach to sustainability reporting align with industry best practices, and what measurable outcomes have been achieved in environmental, social, and governance (ESG) areas over the past three years? (purely narrative)

    - Hybrid Queries: These queries will require multiple tasks from Narrative Analysis, Predictive Analysis, Financial Analysis. 
      eg. What happened to Apple's profit in 2020 and why?: Requires Financial Analysis +  Narrative Analysis

    - You also need to give a list of all the CIKs that the user query is talking about so that it is possible to fetch required information.

    - Return the exact metric name matches that are needed
    eg: if the query requires the metric 'Revenue' then only return (These are in decreasing order of importance):
      'Reveune','RevenueFromContractWithCustomerExcludingAssessedTax','SalesRevenueNet','SalesRevenueServicesGross'.
      'OperatingIncomeLoss','NontradeReceivablesCurrent', etc should not be returned as they are not important or matching

    - Give at least 3 and at max 5 metrics, not more than that in any case.
    - Give prority to those metrics that match the query directly : eg: Liabilites should be mapped to Liabilities as it direclty matches. It should not be mapped to AccountsPayableAndAccruedLiabilities,AccountsPayableAndAccruedLiabilitiesCurrent etc

      USER_QUERY: 
      {query}

    - Give at least 3 and at max 10 metrics, not more than that in any case.
      METRICS:
      {metrics}
      - A Dictionary of cik and a comma separated text containing financial metrics for different ciks.
      - You should give the exact string of the metric required from the input {metrics}, no changes in spelling,spaces,case, anything, same exact string as in the document.
      - Eg: do not return 'Gross Profit', in the document it is referred as 'GrossProfit' and so on.
    
      GIVEN COMPANIES INFORMATION (Name and CIK for lookup, assume these are the full names):
      {companies}

    **NOTE**:
    - You need to return at least 10 metrics in decreasing order of importance.

      {format_instructions}
    """,
    input_variables=["query","metrics"],
    partial_variables={"format_instructions":task_parser.get_format_instructions(),"companies":get_all_company_ciks()}
)

financial_analysis_prompt = PromptTemplate(
    template="""
    You are an expert financial agent with expertise in understanding the given financial metric along with understanding the 
    significance of the trend being observed. The data you recieve is a Python Dictionary of Dictionary, containing metric name as keys 
    and a dictionary of comapny_name to datas as values. So it can have multiple metrcis along with multiple companies.

    - Your task is to give a detailed analysis as to
      - what each of the metrics means (a good definition of it), 
      - what the trend being observed for each metric is, 
      - whether the trend is good or bad respective to the metric,
    Do these for all the metrics in the input data.
    Each section should have professional wordings along good formatting and detailed explanation.

    Please adhere to this structure
    {{
           "financial_metric_report": List[str] "Detailed explanation of the metrics that will be written in the report",
           "require_visualization": bool "Infer from the user query whether visualization is needed. If nothing is mentioned 
              in the query, then decide whether it'll be better to explain the query with a visualizaion or is it not necessary.
              return as True or False."
           "require_summarization": bool "Infer from the user query whether analysis/summarization is needed. If nothing is mentioned 
              in the query, then decide whether it'll be better to explain the query with an explanation of the observed trends or is it not necessary.
              return as True or False."
    }}

    USER QUERY:
    {query}

    DATA:
    {data}

    **NOTE**:
    Visualization is required when:
    - the user query directly asks for it.
    - It is required to answer the query properly.

  Very Rarely is visualization not required though.

  **DO NOT INCLUDE MARKDOWN TEXTS LIKE **,## etc, in "financial_report" BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**
  Just return a basic string in "financial_report" and bool in the other 2.

    {format_instructions}

    """,
    input_variables=["query","data"],
    partial_variables={"format_instructions":financial_analysis_parser.get_format_instructions()}
)

data_visualization_prompt = PromptTemplate(
    template="""
    You are an expert Python programmer specialized in data visualization with Matplotlib and efficient in handling pandas DataFrames.
    Your task is to generate complete and runnable Python code to visualize the provided data using the most suitable data visualization technique.

    ### INSTRUCTIONS:
    1. **Code Requirements:**
       - The visualization MUST use `matplotlib.pyplot`.
       - You MUST generate a separate plot for each metric in the provided data.
       - Save each generated image in the folder: `Reports/Images/`.
       - Name the output file: `type_of_chart_title.png`. The title must be short, clear, and derived from the `metric_name` and `company_names`.
       - Ensure all necessary imports (e.g., `import matplotlib.pyplot as plt`) are at the top.
       - Include a legend for clarity in the visualization.

    2. **Data Structure:**
       - The `datas` dictionary has the following format:
        {{
          metric1_name: {{company1_name: values_list, company2_name: values_list, ...}}, 
          metric2_name: {{company1_name: values_list, company2_name: values_list, ...}},
          And so on.
        }}
       - Access the data using the provided loop to avoid errors:
         ```python
            for metric_name, company_data in datas.items():
              plt.figure(figsize=(12, 6))
              for company_name, values in company_data.items():
                # Directly use 'values' since it's already a list
                historical_values = values
                dates = pd.date_range(start=start_date, end=end_date, periods=len(historical_values))
         ```

    3. **Visualization Requirements:**
       - For each metric:
         - Generate **one visualization** showing all companies' data.
         - Adjust for cases where one company has more data points than others.
         - Choose the best visualization type based on the data or use the provided `visualization_request` ({visualization_request}). Options include:
           - Line Chart
           - Bar Graph
           - Scatter Plot
           - Stacked Bar Graph
           - Pie Chart
           - Doughnut Chart
         - Include axis labels, a title, and any other necessary annotations to make the chart professional and easy to interpret.

    4. **Output File Management:**
       - The Generated Code should Save the file with a name derived from:
         - Type of chart
         - Metric name
         - Company names
       - Example: `line_chart_revenue_apple_google.png`.

    
    5. **Error Handling:**
       - Handle cases where no data is available for a company or metric gracefully.

    ### INPUT DATA:
    - FULL NUMERIC DATA:
      {datas}
    - START DATE:
      {start_date}
    - END DATE:
      {end_date}
    - METRIC NAME:
      Inferred from the `datas` dictionary.

    ### OUTPUT:
    - A list of filenames for all generated visualizations.
    - Include a success message at the end of the response.

    {format_instructions}
    """,
    input_variables=["datas", "start_date", "end_date", "visualization_request"],
    partial_variables={"format_instructions": visualizer_parser.get_format_instructions()}
)

financial_performance_analysis_prompt = PromptTemplate(
    template="""
    You are an expert financial analyst and reporter with extensive knowledge and experience in analyzing financial data and trends.
    Your task is to provide a comprehensive and professional analysis of the financial performance based on the user's query and provided data.
    
    **Your Analysis Should Include**:
    - Identification of major trends and patterns in the data.
    - Highlight any notable discrepancies or anomalies in specific years.
    - Provide detailed observations about mean values, variations, and overall stability.
    - Analyze years with significant discrepancies and hypothesize potential reasons or external factors influencing these discrepancies.
    - Include a clear narrative that connects the observations with possible causes, such as market trends, company decisions, or economic factors.


    **USER QUERY**:
    {query}

    **DATA PROVIDED**:
    {datas}

    **NOTE**:
    - **Give at least 5 points for this section**
    - Ensure your analysis is structured and concise, with bullet points or numbered lists where appropriate.
    - Avoid unnecessary jargon; ensure the explanation is accessible yet professional.
    - If the data provided is insufficient or inconsistent, mention this and suggest possible next steps for further investigation.

    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**

    Please adhere to this structure
    {{
           "performance_report": List[str] "Detailed explanation of the metrics that will be written in the report",
    }}
    
  """,
  input_variables=["query", "datas"]
)

narrative_analysis_prompt = PromptTemplate(
    template="""
    You are a financial report Narrative Analyzer. Your task is to summarize the information you get from the provided list of documents.

    - You have to analyze the documents, see whether the trend has a positive or a negative impact if the same trend continues.

    **Impact Assessment**:
    - For each trend, determine future impact direction using this scale:
     - Strongly Positive (+2)
     - Moderately Positive (+1)
     - Neutral (0)
     - Moderately Negative (-1)
     - Strongly Negative (-2)
    - Provide quantitative evidence where available (e.g., "25% ncrease in R&D spend") 
    - Give insightful reasons and proper explanation for the summary and see whether the user query requires strategic recommendations or not  

    Strategic Recommendations wil be required if the query requires Guidance for future planning and execution.
    eg what were Apple's Assets for 2022 and how to manage them properly.

    Risk analysis will be required if the query asks for it.

    Please adhere to this structure
    {{
           "narrative_report": List[str] "Your summary here. Avoid using markdown, tables, or unescaped special characters.",
           "require_strategic_recommendation": bool "Infer from the user query whether a strategic recommendation 
            is required or not. It'll be required when an advice is being asked. This advice can be beneficial for  
            the future as it can guide the decision making. Return as True or False."
           "require_risk_analysis": bool "Decide whether the user query requires risk analysis or not."
    }}

    USER QUERY:
    {query}

    RELEVANT DOCUMENTS:
    {documents}

    **NOTE**:
    You need to check the report again after generating to ensure that the output is properly formatted such that no json error like Invalid \escape come.

    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**

    {format_instructions}
""",
input_variables=["query","documents"],
partial_variables={"format_instructions":narrative_analysis_parser.get_format_instructions()}
)

strategic_recommendation_prompt = PromptTemplate(
    template="""
    You are an experienced financial strategist with expertise in crafting actionable and forward-looking strategies based on financial insights and relevant documentation.

    **Your Task**:
    - Analyze the provided user query, relevant 10K documents, and the summarized content of these documents.
    - Frame well-structured and actionable strategies that align with the company's financial performance, goals, and market position.

    **USER QUERY**: 
    {query}

    **RELEVANT DOCUMENTS**:
    {documents}

    **SUMMARY OF THE 10K DOCUMENTS**:
    {summary}

    **YOUR OUTPUT SHOULD INCLUDE**:
    1. **Strategic Focus Areas**:
       - Identify key areas where strategies are required (e.g., revenue growth, cost reduction, market expansion, product development, etc.).
    
    2. **Recommended Strategies**:
       - Provide clear, concise, and actionable strategies tailored to the user query.
       - Each strategy should be practical and backed by insights from the 10K documents or other provided information.
    
    3. **Supporting Rationale**:
       - Briefly explain why each recommended strategy is appropriate.
       - Use data or insights from the provided summary to justify your recommendations.

    4. **Potential Risks and Mitigation**:
       - Identify potential risks associated with the recommended strategies.
       - Suggest risk mitigation approaches to ensure successful implementation.

    **NOTE**:
    - Structure your recommendations clearly using bullet points or numbered lists.
    - Ensure your tone is professional, concise, and actionable.
    - Avoid generic suggestions; ensure all recommendations are specific to the user query and provided documents as well as the company.

    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**

    Please adhere to this structure
    {{
           recommendation_report: List[str] = Field(description="It refers to the string generated by the Agent that will be in the final pdf")
    }}

  """,
  input_variables=["query", "documents", "summary"]
)

risk_analysis_prompt = PromptTemplate(
    template="""
    You are an expert and experienced financial risk analyst. Your task is to analyze the risks associated with the user query based on the provided relevant documents and the summary of the 10K documents.

    **Your Objective**:
    - Identify and evaluate key risks.
    - Provide actionable insights to mitigate or manage these risks.

    **USER QUERY**:
    {query}

    **RELEVANT DOCUMENTS**:
    {documents}

    **SUMMARY OF THE 10K DOCUMENTS**:
    {summary}

    **YOUR OUTPUT SHOULD INCLUDE**:
    1. **Key Risk Areas**:
       - Identify the primary risks associated with the user query.
       - Categorize risks (e.g., operational, financial, market-related, regulatory, etc.).

    2. **Risk Evaluation**:
       - Assess the likelihood and potential impact of each identified risk.
       - Use insights from the documents to substantiate your evaluation.

    3. **Risk Mitigation Strategies**:
       - Provide actionable steps to address each risk.
       - Include both short-term and long-term approaches.

    **NOTE**:
    - **Give at least 5 points**
    - Present your analysis in a structured format with headings, bullet points, or numbered lists for readability.
    - Ensure your tone is professional and analytical.
    - Avoid generic statements; your insights should be specific to the query and supported by the provided data.

    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**

    Please adhere to this structure
    {{
           risk_analysis_report: List[str] = Field(description="analyze the risk associate with the given user query to be included in the report.")
    }}

    """,
    input_variables=["query", "documents", "summary"],
)

prediction_analysis_prompt = PromptTemplate(
    template="""
    You are an expert financial agent with expertise in understanding the given financial metric along with understanding the 
    significance of the trend being observed. The data you recieve is a Python Dictionary of Dictionary, containing metric name as keys 
    and a dictionary of comapny_name to datas as values. So it can have multiple metrcis along with multiple companies.

    - Your task is to give a detailed analysis as to
      - what the trend being observed for each metric is, 
      - whether the trend is good or bad respective to the metric,
      - what statergies should the company use so that in future it performs better with respect to the metric.
    Do these for all the metrics in the input data.
    Each section should have professional wordings along good formatting and detailed explanation.

    FUTURE DATA:
    {future_data}

    **Strictly DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**

    Please adhere to this structure
    {{
           prediction_report : str = Field(description="Includes the future values of datas and explains the implications of it.")
    }}
    
    """,
    input_variables=["future_data"]

)

metric_prompt = PromptTemplate(
    template="""
        You are an expert financial analysis agent specializing in understanding financial statements and all the metrics contained within them.
        Your task is to analyze the user's query and extract **all relevant financial information** into a structured JSON format.

        User Query: {query}

        ---

        IMPORTANT INSTRUCTIONS:
        - **Your output MUST be a python Dictionary** that strictly adheres to the provided `Metric_Information` schema.
        - **Only use metrics from the GIVEN FINANCIAL METRICS list.** Do not invent or include any metrics not explicitly listed.
        - **Only use company names from the GIVEN COMPANIES INFORMATION list.** Map user mentions (e.g., 'Apple') to the full company name (e.g., 'Apple Inc.').
        - If the user's query implies a visualization (e.g., 'plot', 'show trends'), infer the `visualization_type` accordingly.
        - If dates or fiscal periods are implied (e.g., 'last 5 years', 'Q1 2023'), extract them accurately. For 'last X years', calculate the `start_date` relative to today (today is {current_date}).
        - If no specific metrics are mentioned but data retrieval/visualization is implied, suggest the 2 most commonly requested and relevant metrics (e.g., 'Revenue', 'NetIncomeLoss').
        - Provide a concise `explanation` for your choices in the final JSON output.
        - If a user query cannot be fully addressed by the available metrics or companies, still fill the JSON fields as best as possible and use the `explanation` to state limitations.

        ---

        GIVEN FINANCIAL METRICS:
        {financial_metrics}

        ---

        GIVEN COMPANIES INFORMATION (Name and CIK/Ticker for lookup, assume these are the full names):
        {companies}

        ---

        {format_instructions}
    """,
    input_variables=['query', 'financial_metrics', 'companies', 'current_date'], # Added current_date
    partial_variables={"format_instructions": metric_parser.get_format_instructions()}
)

code_corrector_prompt = PromptTemplate(
    template="""
    You are an experienced coder with expertise in debugging Python code, especially pertaining to:
    - ReportLab PDF generation.
    - Matplotlib plotting.
    - Resolving basic syntax and formatting errors.

    **Your Task:**
    - Identify and correct errors in the provided code while preserving its functionality and purpose.
    - Ensure the code is **executable** and **free of errors**.

    **Common Errors to Resolve:**
    1. **Unfinished or Mismatched Literals:**
       - Unmatched '(', '[', '{{', or any other opening literal.
       - Unterminated string literals or comments, such as "unterminated string literal (detected at line --)".
       The most common:
       - **Errors like '{{ was never closed' or '[ was never closed'. eg '{{' was never closed (<string>, line 201)**
    2. **Syntax and Execution Errors:**
       - Errors related to f-strings, such as unmatched '(' or '{{'.
       - f-string: unmatched '(' (<string>, line 38)
       - like : Error during code execution: f-string: expecting '}}' (<string>, line 43)
       - JSON-related issues, such as `json.decoder.JSONDecodeError: Invalid \escape`.
       - Issues like `Invalid \escape: line 2 column 8665`.
    3. **Indentation Errors:**
       - Unexpected or inconsistent indentation that causes execution failure.
    4. **Type-Related Errors:**
       - Errors like `TypeError: object of type 'float' has no len()`.
    5. **General Debugging:**
       - Correct spelling errors in variable or function names (e.g., `'include_risk_analysi' is not defined`).
       - Ensure code adheres to Python syntax and standards.
    6. **ReportLab Errors**:
       - Errors like : handle_pageBegin args=() cannot access free variable 'set_page_number' where it is not associated with a value in enclosing scope
       - Ensure that the code gives proper output and not just a blank pdf.
       - Errors like **'SimpleDocTemplate' object has no attribute 'page'**.
       - Errors like module 'reportlab.platypus' has no attribute 'ParagraphStyle'
       - Errors like Paragraph.__init__() got an unexpected keyword argument 'id'
       - Errors like <Table@0x1F1C7F3C950 0 rows x unknown cols>... must have at least a row and column

    **Requirements:**
    - Do not alter the intended functionality of the code.
    - Avoid unnecessary text, explanations, or comments unrelated to debugging.
    - Ensure the corrected code is ready to execute directly.

    **Additional Guidelines:**
    - Include a success message indicating the file has been saved, such as:
      `"File saved successfully at: <file_path>"`
    - Double-check for errors introduced during debugging, ensuring there are no typos or logical mistakes.

    CODE:
    {code}


    **Make sure that the code includes saving (eg saving matplotlib images, saving reportlab pdf files)
    **Output the corrected Python code only, without additional explanations.**
    """,
    input_variables=["code","error"]
)

report_writing_prompt = PromptTemplate(
    template="""
    You are an experienced financial report writer specializing in creating concise, visually appealing financial reports using 'reportlab'.

    - Generate Python code that creates a PDF file with a clear, well-organized financial report for the given query.
    - Use bullet points, tables, and images to keep the report concise and visually structured.
    - GIVEN USER QUERY:
    {query}
    **DATA TO BE INCLUDED**:
      - **QUERY:** {query}
      - **Companies:** {companies}
      - **CIKs:** {ciks}
      - **Tasks:** {tasks}
      - **Metrics:** {metrics}
      - **Data**: {datas}
      - **Financial Metric Analysis:** (Use bullet points for clarity) {financial_metric_analysis}
      - **Performance Analysis:** (Use bullet points for clarity) {performance_analysis}
      - **Current Data Visualizations:** (Use bullet points for clarity) {current_file_names}
      - **Narrative Summary:** (Use concise bullet points) {narrative_analysis}
      - **Strategic Recommendations:** (Use short, actionable points) {strategic_recommendations}
      - **Risk Analysis:** (Use bullet points for clarity) {risk_analysis}
      - **Future Predictions:** {future_datas}
      - **Prediction Explanations:** (Use bullet points for clarity) {prediction_explanation}
      - **Future Data Visualizations:** {future_file_names}
      - **Start Date:** {start_date}
      - **End Date:** {end_date}
      - Should the report include Visualizations? {require_visualization}
      - Should the report include Performance Summary? {require_summary}
      - Should the report include Risk Analysis? {require_risk_analysis}
      - Should the report include Strategic Recommenations? {require_strategic_recommendations}
      - hould the report include Future Predictions? {require_predictions}

    - Format images to be appropriately sized and positioned for professional presentation.
    **NOTE**:
    - Include only the headings for which the data is recieved. 
    - If "not available" is present in the recieved data for the heading then don't include the heading.
    - Ensure headings are clear, bold, and follow a logical structure
    - **POSSIBLE HEADINGS** (include only what is required):
        - **Report Title** (always has to be included)
          - Suitable Title for the Report. 

        - **Table of Contents** (always has to be included after the title)
          - Should be a table made using reportlab.
          - Should have all the needed sections.
          - Include 2 columns, section number and section name.
          
        - **Executive Summary** (always has to be included) 
          - High-level overview of the report's findings, key metrics, and recommendations.

        - **Introduction** (always has to be included)
          - Purpose of the report
          - Scope and Methodology
          - Companies being analyzed

        - **Information about the Required Companies** (always has to be included)
          - What they are
          - What their main aim is
          - What sector do they belong to
                  
      - Financial Analysis (include this section only if "financial_metric_analysis" task exists)
        - **Understanding Required Metrics** (Will be included only if task "financial_metric_analysis" exists)
            - Financial Definition of the metrics (Format is properly eg RevenueFromContractWithCustomerExcludingAssessedTax = Revenue From Contract With Customer Excluding Assessed Tax in bold)
            - Include explanation about all the metrics
            - What reasonse justify the rising trend.
            - What possible reasons justify the declinig trend.

        - **Current Trends and Analysis for all the metrics** (Will be required if 'require_visualization' = True)
          - Analysis of historical trends in key metrics.(Should include information from financial_analysis)
          - **Do not include tables for historical data representation due to token limit**
          **REMEBER THAT THE MAX DATE YOU CAN HAVE IS JULY OF 2025**
          - Include current data visualizations in between the explanation so that the file looks better.
          
        - **Performance Summary** (Will be included if 'require_summary' = True)
          - Include all the important trends and discrepancies as seen from the input "datas".
          - Do not make a table for present datas, however use tables wherever else required.
          - Give a good overview pf the perfomance with poper formatting.
          - Getting Data from input 'performance_analysis', provide a good report with proper bullet points and formatting.
        
      - Narrative Analysis (include this section only if "narrative_analysis" task exists)
        - **Summarizing 10K Documents** (Will be included if 'narrative_analysis' task exists)
          - 'narrative_analysis' is a list of strings that contain points
          - Fetching information from 'narrative_analysis' provide a proper journal summary with concise bullet points
          - Make concise and strong bullet points
          
        - **Risk Analysis** (Will be included only if input "risk_analysis" = True)
          - Key Risk Areas
          - Risk Evaluation
          - Risk Mitigation Strategies
          (Risk pertaining for the given user query and the Narrative Summary)  (infer this from narrative analysis)
          (Market dynamics, company strategies, and external influences)

        - **Strategic Recommendations** (Will be included only if "strategic_recommendations" = True exists)
          - Strategic Focus Areas
          - Recommended Strategie
          (Actionable steps for improving or sustaining financial performance)
          (Focus on investments, cost management, or operational improvements)
          Format it nicely

        **INCLUDE TABLES WHEREVER POSSIBLE**
        
      - **Predictive Analysis** (included only when "predictive_analysis" task exists)
        - **Future Trends and Forecast** (Will be included only if "predictive_analysis" task exists)
          - Always Inlcude the visulaizations with paths in "./Reports/Predictions/'future_file_names'"
          - Predictions of key metrics over specified periods.
          - Methodology and assumptions used for forecasting.
          - Present the forecasted data in reportlab tables. 1 table should include the forecast of 1 metric with both holt and arima. (total of 3 columns)

      - **Conclusion** (Always has to be included)
        - Summarize key findings.


    **INSTRUCTIONS**
    - Include images from:
        - './Reports/Images' folder for present data visualizations, filenames that need to be used are already provided.
        - './Reports/Predictions' folder for future data visualizations, filenames that need to be used are already provided.
    - Images should be scaled smaller and may be alligned beside each other for compact presentation.
    - The generated PDF should be saved in the 'Reports' folder with a descriptive title reflecting the query's aspects.
    - Ensure success message confirms execution at the end.
    - Make sue that Errors like **'SimpleDocTemplate' object has no attribute 'page'** do not exist.
    - Ensure that name **'section_num' is not defined** error does not exist

    **CODE REQUIREMENTS**:
    1. The code must:
       - Automatically load and use the provided input data.
       - Include all sections specified, but only if the relevant data is available.
       - Exclude sections marked as "not available."
       - Save the PDF report in the './Reports' folder with a descriptive title based on the query.
       - **Do not include comments as there is token issue**
    2. **Formatting Guidelines**:
       - Use bullet points, tables, and concise text for clarity.
       - Ensure that a row contains at max 2 images (eg: to include 3 images 1st row should have 2 images and the next row should have only 1 image)
       - Ensure all tables fit within the PDF margins, splitting into multiple lines if necessary.
       - **No empty sections should be there in the final report**
    3. **Execution Guidelines**:
       - Ensure the generated Python code executes without errors and directly outputs the report.
       - Include a success message at the end confirming the PDF's generation and its file path.
       - Avoid requiring manual input or additional parameters.

    4. **Code Restrictions**:
       - Do NOT include any example usage.
       - Do NOT generate markdown-like text (e.g., `*`, `#`, etc.).
       - Do NOT include placeholder text such as "replace with data"—use the provided input data directly.

    **NOTE**:
    - Avoid lengthy textual paragraphs; use bullet points, tables, and concise sentences.
    - Properly format all sections for clarity and professional appearance.
    - All the specified sections should be included only if the required data is present.
    - Include the heading if relevant data is present
    - Include a success message at the end to confirm execution.
    - **Include the code to save the report in the ./Reports folder.** 

    **DO NOT INCLUDE MARKDOWN TEXTS LIKE *,# etc BE SAFE WHILE USING \n SUCH THAT IT DOES NOT GIVE JSON PARSING ERROR**
    **INCLUDE A SUCCESS STATEMENT AT THE END OF THE GENERATED CODE**
    **DO NOT GIVE EXAMPLE USAGE, JUST USE THE GIVEN DATA FOR REPORT GENERATION**
    **THE CODE SHOULD BE SUCH THAT WHEN EXECUTED, IT DIRECTLY GIVES THE OUTPUT WITHOUT USER INTERVENTION**
    **GIVE THE FULL FUNCTIONING CODE, NOT A PART**
    **initialize the datas at the top after imports**
    """,
    input_variables=['query','companies', 'ciks', 'tasks', 'metrics', 'datas', 'financial_analysis', 'performance_analysis','current_file_names', 'narrative_analysis', 'strategic_recommendations', 'risk_analysis', 'future_datas', 'prediction_explanation', 'future_file_names', 'start_date', 'end_date','require_visualization','require_summary','require_risk_analysis','require_strategic_recommendations','require_predictions']
)

code_finisher_prompt = PromptTemplate(
    template="""
    You are an experienced financial report writer with expertise in finishing the given reportlab code with the objective of 
    giving final code the fianancial report.

    **NOTE**:
    - Include only the headings for which the data is there in the code. 
    - If "not available" is present in the recieved data for the heading then don't include the heading.
    - Ensure headings are clear, bold, and follow a logical structure
    - **POSSIBLE HEADINGS** (include only what is required):
        - **Report Title** (always has to be included)
          - Suitable Title for the Report. 

        - **Table of Contents** (always has to be included)
          - Should be a reportlab table.
          - Should have all the needed sections as well as the page numbers.
          - Page numbers should be added after all the sections are already written in the pdf.
          
        - **Executive Summary** (always has to be included) 
          - High-level overview of the report's findings, key metrics, and recommendations.

        - **Introduction** (always has to be included)
          - Purpose of the report
          - Scope and Methodology
          - Companies being analyzed

        - **Information about the Required Companies** (always has to be included)
          - What they are
          - What their main aim is
          - What sector do they belong to
                  
        - **Understanding Required Metrics** (Will be required only if "financial_analysis" task exists)
            - Financial Definition of the metrics
            - What reasonse justify the rising trend.
            - What reasons justify the declinig trend.

        - **Current Trends and Analysis for all the metrics** (Will be required only if "financial_analysis" task exists and visualization will be inserted if 'require_visualization' = True)
          - Analysis of historical trends in key metrics.(Should include information from financial_analysis)
          - Present data in reportlab tables
          - The tables should not be wide, they should properly fit inside the width of the margin of the pdf. So at max 5 columns then shift to the next line, along with proper dates.
          - Tables, charts, and bullet points summarizing the data.
          
        - **Performance Summary** (Will be included if 'require_summary' = True)
          - Getting Data from input 'performance_analysis', provide a good report with proper bullet points and formatting.
        
        - **According to 10K report** (Will be required if 'narrative_analysis' task exists)
          - Fetching information from 'narrative_analysis' provide a proper journal summary with concise bullet points
          
        - **Risk Analysis** (Will be required only if input "risk_analysis" = True)
          - Risk pertaining for the given user query and the Narrative Summary  (infer this from narrative analysis)
          - Market dynamics, company strategies, and external influences

        - **Strategic Recommendations** (Will be required only if "strategic_recommendations" = True exists)
          - Actionable steps for improving or sustaining financial performance
          - Focus on investments, cost management, or operational improvements
        
        - **Future Trends and Forecast** (Will be required only if "predictive_analysis" task exists)
          - Inlcude the visulaizations with paths in "./Reports/Predictions/'future_file_names'"
          - Predictions of key metrics over specified periods.
          - Methodology and assumptions used for forecasting.
          - Present the forecasted data in reportlab tables

        - **Conclusion**
          - Summarize key findings.

    **KEEP THE INDENTATION PROPER WITH THE LAST PART YOU RECIEVED SUCH THAT IT IS EXECUTED**
    INPUT CODE:
    {code}

    Only return the unfished part of the input code so that token limit is not reached.
    **KEEP THE INDENTATION PROPER WITH THE LAST PART YOU RECIEVED SUCH THAT IT IS EXECUTED**

    """,
    input_variables=["code"]
)

concise_answer_prompt = PromptTemplate(
    template="""
    You are an experienced financial data summarizer. Your main aim is to generate a concise summary from the input data.
    
    - You need to give good, concise bullet points that encapsulate all the data recieved from the user.

    - All the bullet points shouold be separated by \n.

    - GIVEN USER QUERY:
    {query}
    **DATA TO BE INCLUDED**:
      - **QUERY:** {query}
      - **Companies:** {companies}
      - **CIKs:** {ciks}
      - **Tasks:** {tasks}
      - **Metrics:** {metrics}
      - **Data**: {datas}
      - **Financial Metric Analysis:** (Use bullet points for clarity) {financial_metric_analysis}
      - **Performance Analysis:** (Use bullet points for clarity) {performance_analysis}
      - **Current Data Visualizations:** (Use bullet points for clarity) {current_file_names}
      - **Narrative Summary:** (Use concise bullet points) {narrative_analysis}
      - **Strategic Recommendations:** (Use short, actionable points) {strategic_recommendations}
      - **Risk Analysis:** (Use bullet points for clarity) {risk_analysis}
      - **Future Predictions:** {future_datas}
      - **Prediction Explanations:** (Use bullet points for clarity) {prediction_explanation}
      - **Future Data Visualizations:** {future_file_names}
      - **Start Date:** {start_date}
      - **End Date:** {end_date}
      - Should the report include Visualizations? {require_visualization}
      - Should the report include Performance Summary? {require_summary}
      - Should the report include Risk Analysis? {require_risk_analysis}
      - Should the report include Strategic Recommenations? {require_strategic_recommendations}
      - hould the report include Future Predictions? {require_predictions}

    **PLEASE ADHERE TO THIS STRUCTURE**
    {{
      points: List[str] (Each bullet point is a string)
    }}
    **DO NOT INCLUDE DANGLING PERIODS**

    MAKE SURE THAT NO JSON PARSING ERROR COMES.

    eg :
      {{
        "points": [
          "Challenges in managing frequent product and service transitions to remain
          competitive, including timely development, market acceptance, and production ramp-up.",
        .
          "Risk of new technologies leading to lower revenues and profit margins.",
        .
          "Compliance risks related to laws and regulations concerning technology,
          intellectual property, digital platforms, machine learning, AI, internet,
          telecommunications, media, labor, and environmental issues.",
        .
          "Failures in IT systems, increasing cybersecurity threats (including ransomware),
          and network disruptions impacting online services, customer transactions, and
          manufacturing.",
        .
          "Risks associated with high profile and the value of confidential information.",
        .
          "These risks are expected to accelerate in frequency and sophistication, especially
          during diplomatic or armed conflict."
        ]
      }} is wrong.

      {{
  "points": [
    "Challenges in managing frequent product and service transitions to remain competitive, including timely development, market acceptance, and production ramp-up.",
    "Risk of new technologies leading to lower revenues and profit margins.",
    "Compliance risks related to laws and regulations concerning technology, intellectual property, digital platforms, machine learning, AI, internet, telecommunications, media, labor, and environmental issues.",
    "Failures in IT systems, increasing cybersecurity threats (including ransomware), and network disruptions impacting online services, customer transactions, and manufacturing.",
    "Risks associated with high profile and the value of confidential information.",
    "These risks are expected to accelerate in frequency and sophistication, especially during diplomatic or armed conflict."
  ]is right
  }}

  Give good actionalble insights. Do no include vague answers and do not include generic points. 
  Give the answer such that all the output information should give good and actionable points
  Even if the answer is that you can not get relevant information, include all the information that can give good actionable insightful answeres.
    {format_instructions}

    """,
    input_variables=['query','companies', 'ciks', 'tasks', 'metrics', 'datas', 'financial_analysis', 'performance_analysis','current_file_names', 'narrative_analysis', 'strategic_recommendations', 'risk_analysis', 'future_datas', 'prediction_explanation', 'future_file_names', 'start_date', 'end_date','require_visualization','require_summary','require_risk_analysis','require_strategic_recommendations','require_predictions'],
    partial_variables={"format_instructions":concise_answer_parser.get_format_instructions()}
)