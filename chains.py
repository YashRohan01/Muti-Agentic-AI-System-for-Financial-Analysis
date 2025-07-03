from langchain_core.output_parsers import StrOutputParser
from prompts import rag_prompt,query_prompt,task_allocating_prompt,financial_analysis_prompt,data_visualization_prompt,code_corrector_prompt,narrative_analysis_prompt,strategic_recommendation_prompt,prediction_analysis_prompt,report_writing_prompt,rag_report_writing_prompt,financial_performance_analysis_prompt,risk_analysis_prompt,code_finisher_prompt,concise_answer_prompt, concise_rag_prompt
from parsers import rag_parser,query_parser,task_parser,financial_analysis_parser,visualizer_parser,narrative_analysis_parser,strategic_recommendation_parser,predictive_analysis_parser,report_parser,rag_report_parser,financial_analysis_parser,risk_analysis_parser,concise_answer_parser, rag_concise_parser, strategic_recommendation_parser, performance_summary_parser
from models import basic_model

verify_chain = code_corrector_prompt | basic_model | StrOutputParser()

query_chain = query_prompt | basic_model | query_parser

rag_chain = rag_prompt | basic_model | rag_parser

rag_concise_chain = concise_rag_prompt | basic_model | rag_concise_parser

rag_report_writer_chain = rag_report_writing_prompt | basic_model | StrOutputParser()

task_chain = task_allocating_prompt | basic_model | task_parser

financial_analysis_chain = financial_analysis_prompt | basic_model | financial_analysis_parser

visualization_chain = data_visualization_prompt | basic_model | visualizer_parser

performance_summary_chain = financial_performance_analysis_prompt | basic_model | performance_summary_parser

narrative_analysis_chain = narrative_analysis_prompt | basic_model | narrative_analysis_parser

strategic_reccomendation_chain = strategic_recommendation_prompt | basic_model | strategic_recommendation_parser

risk_analysis_chain = risk_analysis_prompt | basic_model | risk_analysis_parser

predcition_chain = prediction_analysis_prompt | basic_model | predictive_analysis_parser

report_writing_chain = report_writing_prompt | basic_model | report_parser

code_finisher_chain = code_finisher_prompt | basic_model | StrOutputParser()

concise_answer_chain = concise_answer_prompt | basic_model | concise_answer_parser