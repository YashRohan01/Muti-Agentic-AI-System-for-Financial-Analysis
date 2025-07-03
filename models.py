from langchain_google_genai import ChatGoogleGenerativeAI
from tools import get_metric_for_all_ciks,get_relevant_douments
from dotenv import load_dotenv
load_dotenv()

basic_model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

rag_model = basic_model.bind_tools([get_relevant_douments])

financial_model = basic_model.bind_tools([get_metric_for_all_ciks])