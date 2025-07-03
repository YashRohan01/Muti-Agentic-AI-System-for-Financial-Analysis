from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_relevant_douments,get_metric_for_all_ciks,execute_python_code,get_metric_name_for_ciks,process_and_forecast,plot_and_save_forecasts,execute_code
from langchain_core.output_parsers import StrOutputParser
from chains import query_chain,rag_report_writer_chain,rag_chain,task_chain,financial_analysis_chain,visualization_chain,performance_summary_chain,narrative_analysis_chain,strategic_reccomendation_chain,risk_analysis_chain,predcition_chain,report_writing_chain,verify_chain,code_finisher_chain,concise_answer_chain, rag_concise_chain
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")


# query = "What happened to Apple's profit in 2020 vs 2021 and why?" # Narrative Analysis
# query = "What will be the future Assets for Apple?"
# query = "What will be the future Assets for Apple and why?"
# query = "How does the bord of Director of Meta Look like?"
# query = "What was Apple's total Assets for 2022?" # Simple
# query = "How did Apple's total Assets vary from 2018 to 2024. What possible reasons can there be for it.Give Visualizations for better understanding. Also show how will Assets look in the future"
# query = "How much did South-East Asia contribute to Apple in 2022 vs in 2020?" # Simple
# query = "How has the Apple's and Meta's Revenue evolved over 2016 and 2018, and what strategic initiatives have driven significant changes in Revenue?"
# query = "How has Meta's debt profile changed over the last five years, and what implications does this have for its financial stability and investment capacity?"
# query = "What are the key risks identified by Apple related to emerging technologies, such as artificial intelligence, and how has the xc vhb risk mitigation strategy evolved in response to these technologies?"
# query = "How does the Apple's approach to sustainability reporting align with industry best practices, and what measurable outcomes have been achieved in environmental, social, and governance (ESG) areas over the past three years?"
# query = "What is Revenue and how has the trend for it for Apple has been from 2015-2018 and tell me the possible reasons for it?"
#query = "What are the trends in the Meta's revenue diversification across different geographical regions, and how have geopolitical factors influenced these trends over the past five years?"
query = "What happened to Apple's profit in 2020 vs 2021?"

print("Classifying Query")
query_response = query_chain.invoke({'query':query})

print(query_response.type)
print(query_response.explanation)
print()

datas = {}
financial_analysis_response = None
visualization_response = None
performance_summary_response = None
narrative_analysis_response = None
strategic_reccomendation_response = None
risk_analysis_response = None
future_datas = {}
prediction_response = None
prediction_image_file_names = []

require_visualization = False
require_summary = False
require_risk_analysis = False
require_strategic_recommendations = False
require_predictions = False

if query_response.type == 'simple':
    print("Processing Simple Query:")
    rag_response = rag_chain.invoke({'query':query,'docs':get_relevant_douments(query,10)})
    print("Generating concise report")
    brief_answer = rag_concise_chain.invoke({'query':query,'documents':get_relevant_douments(query,10), "summary":rag_response.rag_report})
    print(brief_answer)
    print(brief_answer.points)
    print("Genrating Report")

    rag_report_reponse = rag_report_writer_chain.invoke({"query":query, "summary":rag_response.rag_report})
    
    verified_code = verify_chain.invoke({'code':rag_report_reponse})
    print(verified_code)
    if verified_code.startswith("```python") and verified_code.endswith("```"):
        code_to_execute = verified_code[9:-3].strip()  # Remove both prefix and suffix
        execute_python_code(verified_code[9:-3])
    elif verified_code.startswith("```python"):
        print("Case 2: Only prefix exists")
        
    elif verified_code.endswith("```"):
        print("Case 3: Only suffix exists")
        code_to_execute = verified_code[:-3].strip()  # Remove only the suffix
        execute_python_code(verified_code[:-3])
    else:
        print("Case 4: No prefix or suffix")
        code_to_execute = verified_code.strip()  # No prefix/suffix, just strip whitespace
        execute_python_code(verified_code)


elif query_response.type == 'complex':
    print("Processing Complex Query:")
    print()
    print("Allocating Tasks:")
    task_response = task_chain.invoke({'query':query,"metrics":get_metric_name_for_ciks(query_response.ciks)})
    print(task_response.tasks)
    print(task_response.ciks)
    print(task_response.companies)
    print(task_response.metrics)
    print(task_response.explanation)

    if 'financial_metric_analysis' in task_response.tasks:
        print("Executing Financial Analysis")
        print()
        print("Fetching Required Data from DataFrames")
        datas = get_metric_for_all_ciks(task_response)
        print(datas.keys())
        # print(datas)
        print("Starting Financial Analysis")
        financial_analysis_response = financial_analysis_chain.invoke({"query":query,"data":datas})
        print(f"Require Visualizations: {financial_analysis_response.require_visualization}")
        
        if financial_analysis_response.require_visualization:
            require_visualization = True
            print("Generating Visualizations")
            visualization_response = visualization_chain.invoke({'datas':datas,'start_date':task_response.start_date,'end_date':task_response.end_date,'visualization_request':''})
            print(visualization_response.file_names)
            verified_code = verify_chain.invoke({'code':visualization_response.code})
            if verified_code.startswith("```python") and verified_code.endswith("```"):
                code_to_execute = verified_code[9:-3].strip()  # Remove both prefix and suffix
                execute_python_code(verified_code[9:-3])
            elif verified_code.startswith("```python"):
                print("Case 2: Only prefix exists")
        
            elif verified_code.endswith("```"):
                print("Case 3: Only suffix exists")
                code_to_execute = verified_code[:-3].strip()  # Remove only the suffix
                execute_python_code(verified_code[:-3])
            #print(verified_code)
        print(f"Require Performance Analysis: {financial_analysis_response.require_summarization}")
        if financial_analysis_response.require_summarization:
            require_summary = True
            print("Generating summary")
            print()
            performance_summary_response = performance_summary_chain.invoke({'query':query,'datas':datas})


    if 'narrative_analysis' in task_response.tasks:
        print("Summarizing Relevant 10K Documents")
        print()

        narrative_analysis_response = narrative_analysis_chain.invoke({"query":query,"documents":get_relevant_douments(query,15)})

        print(f"Require Strategic Recommendations: {narrative_analysis_response.require_strategic_recommendation}")
        if narrative_analysis_response.require_strategic_recommendation:
            require_strategic_recommendations = True
            print("Generating Strategic Recommendations")
            strategic_reccomendation_response = strategic_reccomendation_chain.invoke({'query':query,'documents':get_relevant_douments(f"Give Strategic Recommendations for the query : {query}",15),'summary':narrative_analysis_response.narrative_report})

        print(f"Require Risk Analysis: {narrative_analysis_response.require_risk_analysis}")
        if narrative_analysis_response.require_risk_analysis:
            require_risk_analysis = True
            print("Analyzing Risk")

            risk_analysis_response = risk_analysis_chain.invoke({'query':query,'documents':get_relevant_douments(f"Risk factors associated with the query: {query}"),'summary':narrative_analysis_response.narrative_report})

    if 'predictive_analysis' in task_response.tasks:
        require_predictions = True
        datas = get_metric_for_all_ciks(task_response)
        print("Forecasting")
        print()
        future_datas = process_and_forecast(datas)
        print(f"Forecasted values: \n{future_datas}")
        prediction_response = predcition_chain.invoke({"future_data":future_datas})
        #print(future_datas)
        prediction_image_file_names = plot_and_save_forecasts(datas,future_datas['holt'],future_datas['arima'])
        print(prediction_image_file_names)

    print("Consice Direct Answer : \n")
    brief_answer = concise_answer_chain.invoke({
    'query': query,
    'companies': task_response.companies or [],
    'ciks': task_response.ciks or [],
    'tasks': task_response.tasks or [],
    'metrics': task_response.metrics or [],
    'datas': datas or {},
    'financial_metric_analysis': getattr(financial_analysis_response, 'financial_metric_report', "Financial Metric analysis not required."),
    'performance_analysis': getattr(performance_summary_response, 'content', "Performance analysis not required."),
    'narrative_analysis': getattr(narrative_analysis_response, 'narrative_report', "Narrative analysis not required."),
    'strategic_recommendations': getattr(strategic_reccomendation_response, 'content', "Recommendations not required.") if strategic_reccomendation_response else "Recommendations not required.",
    'risk_analysis': getattr(risk_analysis_response, 'content', "Risk Analysis not required.") if strategic_reccomendation_response else "Risk Analysis not required.",
    'future_datas': future_datas or {},
    'prediction_explanation': getattr(prediction_response, 'content', "Prediction explanation not required."),
    'current_file_names': getattr(visualization_response, 'file_names', []),
    'future_file_names': prediction_image_file_names or [],
    'start_date': task_response.start_date or "N/A",
    'end_date': task_response.end_date or "N/A",
    'require_visualization': require_visualization,
    'require_summary': require_summary,
    'require_risk_analysis': require_risk_analysis,
    'require_strategic_recommendations': require_strategic_recommendations,
    'require_predictions': require_predictions

    })
    print(brief_answer)

    
    print("Now Report can be generated!")
    # Handle None values with default replacements
    final_response = report_writing_chain.invoke({
    'query': query,
    'companies': task_response.companies or [],
    'ciks': task_response.ciks or [],
    'tasks': task_response.tasks or [],
    'metrics': task_response.metrics or [],
    'datas': datas or {},
    'financial_metric_analysis': getattr(financial_analysis_response, 'financial_metric_report', "Financial Metric analysis not required."),
    'performance_analysis': getattr(performance_summary_response, 'content', "Performance analysis not required."),
    'narrative_analysis': getattr(narrative_analysis_response, 'narrative_report', "Narrative analysis not required."),
    'strategic_recommendations': getattr(strategic_reccomendation_response, 'content', "Recommendations not required.") if strategic_reccomendation_response else "Recommendations not required.",
    'risk_analysis': getattr(risk_analysis_response, 'content', "Risk Analysis not required.") if strategic_reccomendation_response else "Risk Analysis not required.",
    'future_datas': future_datas or {},
    'prediction_explanation': getattr(prediction_response, 'content', "Prediction explanation not required."),
    'current_file_names': getattr(visualization_response, 'file_names', []),
    'future_file_names': prediction_image_file_names or [],
    'start_date': task_response.start_date or "N/A",
    'end_date': task_response.end_date or "N/A",
    'require_visualization': require_visualization,
    'require_summary': require_summary,
    'require_risk_analysis': require_risk_analysis,
    'require_strategic_recommendations': require_strategic_recommendations,
    'require_predictions': require_predictions

    })
    print("Generated Report -> Verifying")
    verified_code = verify_chain.invoke({'code':final_response})
    print(verified_code)
    if verified_code.startswith("```python") and verified_code.endswith("```"):
        code_to_execute = verified_code[9:-3].strip()  # Remove both prefix and suffix
        execute_python_code(verified_code[9:-3])
    elif verified_code.startswith("```python"):
        print("Case 2: Only prefix exists")
        code_finisher_response = code_finisher_chain.invoke({"code":verified_code})
        print(code_finisher_response)
        
    elif verified_code.endswith("```"):
        print("Case 3: Only suffix exists")
        code_to_execute = verified_code[:-3].strip()  # Remove only the suffix
        execute_python_code(verified_code[:-3])
    else:
        print("Case 4: No prefix or suffix")
        code_to_execute = verified_code.strip()  # No prefix/suffix, just strip whitespace
        execute_python_code(verified_code)

    
