from structures import RAG_Agent, Query_Validator, Task_Allocator, Metric_Information, Data_Visualizer,Report_Writer_Agent,Financial_Analyzer, Prediction_Analyzer, Narrative_Analyzer, Stratergy_Recommending_Agent,RAG_Reporter_Agent, Risk_Analyzer, Concise_Answer_Agent, RAG_Concise_Agent, Financial_Performance_Analyer
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser, StrOutputParser

rag_parser = PydanticOutputParser(pydantic_object=RAG_Agent)

query_parser = PydanticOutputParser(pydantic_object=Query_Validator)

task_parser = PydanticOutputParser(pydantic_object=Task_Allocator)

metric_parser = PydanticOutputParser(pydantic_object=Metric_Information)

data_parser = PydanticOutputParser(pydantic_object=Data_Visualizer)

report_parser = StrOutputParser()

visualizer_parser = PydanticOutputParser(pydantic_object=Data_Visualizer)

financial_analysis_parser = PydanticOutputParser(pydantic_object=Financial_Analyzer)

performance_summary_parser = PydanticOutputParser(pydantic_object=Financial_Performance_Analyer)

predictive_analysis_parser = StrOutputParser()

narrative_analysis_parser = PydanticOutputParser(pydantic_object=Narrative_Analyzer)

strategic_recommendation_parser = PydanticOutputParser(pydantic_object=Stratergy_Recommending_Agent)

risk_analysis_parser = PydanticOutputParser(pydantic_object=Risk_Analyzer)

rag_report_parser = StrOutputParser()

concise_answer_parser = PydanticOutputParser(pydantic_object=Concise_Answer_Agent)

rag_concise_parser = PydanticOutputParser(pydantic_object=RAG_Concise_Agent)