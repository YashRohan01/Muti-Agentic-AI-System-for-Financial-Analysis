import streamlit as st
import sys
import io
import contextlib
import time
import os
from pathlib import Path
import base64

# Import your modules here
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from tools import get_relevant_douments, get_metric_for_all_ciks, execute_python_code, get_metric_name_for_ciks, process_and_forecast, plot_and_save_forecasts, execute_code
    from langchain_core.output_parsers import StrOutputParser
    from chains import (query_chain, rag_report_writer_chain, rag_chain, task_chain, 
                       financial_analysis_chain, visualization_chain, performance_summary_chain, 
                       narrative_analysis_chain, strategic_reccomendation_chain, risk_analysis_chain, 
                       predcition_chain, report_writing_chain, verify_chain, code_finisher_chain, concise_answer_chain,rag_concise_chain)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #FFFFFF;
        padding-top: 0rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF5F0 100%);
    }
    
    /* Navigation Buttons */
    .nav-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0 40px 0;
        padding: 0;
    }
    
    .nav-button {
        background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    
    .nav-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
        background: linear-gradient(135deg, #FF8C42 0%, #FF6B35 100%);
    }
    
    .nav-button.active {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3);
    }
    
    .nav-button.active:hover {
        background: linear-gradient(135deg, #20c997 0%, #28a745 100%);
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4);
    }
    
    .query-box {
        background-color: #FFFFFF;
        border: 2px solid #FF6B35;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(255, 107, 53, 0.1);
    }
    
    .thinking-container {
        background-color: #FFF5F0;
        border-left: 4px solid #FF6B35;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    .step-header {
        color: #FF6B35;
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }
    
    .status-badge {
        background-color: #FF6B35;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
    
    .complete-badge {
        background-color: #28a745;
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 5px 0;
    }
    
    .title {
        color: #FF6B35;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        margin: 20px 0 30px 0;
        text-shadow: 2px 2px 4px rgba(255, 107, 53, 0.1);
    }
    
    .subtitle {
        color: #666;
        text-align: center;
        font-size: 1.2rem;
        margin: 0 0 40px 0;
    }
    
    .concise-answer-container {
        background-color: #F8F9FA;
        border: 2px solid #E9ECEF;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        min-height: 150px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .concise-answer-placeholder {
        color: #6C757D;
        font-style: italic;
        text-align: center;
        padding: 50px 0;
        font-size: 1.1rem;
    }
    
    .bullet-point {
        margin: 10px 0;
        padding: 8px 0;
        border-bottom: 1px solid #E9ECEF;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .bullet-point:last-child {
        border-bottom: none;
    }
    
    .bullet-point::before {
        content: "•";
        color: #FF6B35;
        font-weight: bold;
        margin-right: 10px;
    }

    .query-area {
        max-width: 70%;
        margin: 0 auto;
        padding: 20px;
        background-color: #FFFFFF;
        border: 2px solid #FF6B35;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(255, 107, 53, 0.1);
        text-align: center;
    }

    .sample-queries {
        max-width: 10%;
        margin: 20px auto;
        padding: 20px;
        background-color: #FFF5F0;
        border: 1px solid #FF6B35;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(255, 107, 53, 0.1);
        text-align: left;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 1rem;
    }
            
    .custom-container {
        max-width: 700px;
        margin: auto;
        padding-left: 10px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 0px;
        /* background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 100%); */
        color: black;
        border-radius: 15px;
        /*box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);*/
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .main-header p {
        margin: 0px 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitCapture:
    def __init__(self):
        self.output = []
        
    def write(self, text):
        if text.strip():
            self.output.append(text.strip())
        
    def flush(self):
        pass

def render_navigation():
    """Render elegant navigation buttons"""
    # Initialize session state for page navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Financial Analysis'
    
    # Create two columns for the buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create button container
        st.markdown('<div class="nav-container">', unsafe_allow_html=True)
        
        # Financial Analysis Button
        button1_col, button2_col = st.columns(2)
        
        with button1_col:
            if st.button("🏠 Financial Analysis", 
                        key="nav_financial", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == 'Financial Analysis' else "secondary"):
                st.session_state.current_page = 'Financial Analysis'
                st.rerun()
        
        with button2_col:
            if st.button("🏗️ Multi-Agent Architecture", 
                        key="nav_architecture", 
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == 'Architecture' else "secondary"):
                st.session_state.current_page = 'Architecture'
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1 class="subtitle">Intelligent Financial Analysis & Reporting System</h1>
        <p class="title">Multi-Agentic Financial AI</p>
    </div>
    """, unsafe_allow_html=True)

def display_pdf(pdf_path):
    """Display PDF in Streamlit"""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
        
        # Also provide download button
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="📥 Download Report",
                data=file.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                key="download_pdf"
            )
    except Exception as e:
        st.error(f"Could not display PDF: {e}")

def get_initial_pdf_count():
    """Get initial count of PDF files in Reports directory"""
    reports_dir = Path("./Reports")
    if reports_dir.exists():
        return len(list(reports_dir.glob("*.pdf")))
    return 0

def find_new_pdf(initial_count):
    """Find newly generated PDF file in Reports directory"""
    reports_dir = Path("./Reports")
    if not reports_dir.exists():
        return None
    
    current_pdfs = list(reports_dir.glob("*.pdf"))
    current_count = len(current_pdfs)
    
    if current_count > initial_count:
        # Sort by creation time and return the newest
        newest_pdf = max(current_pdfs, key=lambda p: p.stat().st_ctime)
        return newest_pdf
    
    return None

def render_concise_answer(bullet_points=None):
    """Render the concise answer section"""
    st.markdown("## 💡 Concise Answer")
    
    if bullet_points and len(bullet_points) > 0:
        answer_html = '<div class="concise-answer-container">'
        for point in bullet_points:
            answer_html += f'<div class="bullet-point">{point}</div>'
        answer_html += '</div>'
        st.markdown(answer_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="concise-answer-container">
            <div class="concise-answer-placeholder">
                Concise Answer - Key insights will appear here as bullet points
            </div>
        </div>
        """, unsafe_allow_html=True)

def run_analysis_with_thinking_process(query, thinking_container):
    """Run the analysis and return both results and thinking process"""
    
    # This will store all the thinking process steps
    thinking_steps = []
    
    # Initialize brief_answer as None - will be populated during analysis
    brief_answer = None
    
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    
    # Initialize variables
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

    # Helper function to update thinking display
    def update_thinking_display():
        with thinking_container:
            with st.expander("🧠 Agent Thinking Process", expanded=False):
                st.markdown("""
                <div style="background-color: #FFF5F0; border-left: 4px solid #FF6B35; padding: 15px; 
                            border-radius: 0 10px 10px 0; font-family: 'Courier New', monospace; font-size: 14px;">
                """, unsafe_allow_html=True)
                
                for step in thinking_steps:
                    if step.startswith("✅"):
                        st.markdown(f'<div style="color: #28a745; margin: 5px 0;">{step}</div>', unsafe_allow_html=True)
                    elif step.startswith("⚠️"):
                        st.markdown(f'<div style="color: #ffc107; margin: 5px 0;">{step}</div>', unsafe_allow_html=True)
                    elif any(prefix in step for prefix in ["🔍","📋", "📊", "📈", "📖", "🧠", "🔮", "💡"]):
                        st.markdown(f'<div style="color: #FF6B35; font-weight: bold; margin: 10px 0;">{step}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="margin: 5px 0;">{step}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)

    # Step 1: Query Classification
    thinking_steps.append("🔍 Classifying Query")
    update_thinking_display()
    time.sleep(0.5)  # Small delay for visual effect
    
    query_response = query_chain.invoke({'query': query})
    thinking_steps.append(f"Query Type: {query_response.type}")
    thinking_steps.append(f"Explanation: {query_response.explanation}")
    update_thinking_display()
    
    if query_response.type == 'simple':
        thinking_steps.append("📊 Processing Simple Query")
        update_thinking_display()
        time.sleep(0.5)
        
        # RAG processing
        rag_response = rag_chain.invoke({'query': query, 'docs': get_relevant_douments(query, 10)})
        thinking_steps.append("✅ Retrieved relevant documents")
        update_thinking_display()
        
        # Report generation
        rag_report_response = rag_report_writer_chain.invoke({"query": query, "summary": rag_response.rag_report})
        thinking_steps.append("✅ Generated initial report")
        update_thinking_display()
        
        # Code verification and execution
        verified_code = verify_chain.invoke({'code': rag_report_response})
        
        if verified_code.startswith("```python") and verified_code.endswith("```"):
            code_to_execute = verified_code[9:-3].strip()
            execute_python_code(code_to_execute)
        elif verified_code.startswith("```python"):
            thinking_steps.append("⚠️ Code verification - handling prefix")
            update_thinking_display()
        elif verified_code.endswith("```"):
            code_to_execute = verified_code[:-3].strip()
            execute_python_code(code_to_execute)
        else:
            code_to_execute = verified_code.strip()
            execute_python_code(code_to_execute)
        
        thinking_steps.append("✅ Code executed successfully")
        update_thinking_display()
        
        # For simple queries, create a basic brief answer
        brief_answer = rag_concise_chain.invoke({'query':query,'documents':get_relevant_douments(query,10), "summary":rag_response.rag_report})

    elif query_response.type == 'complex':
        thinking_steps.append("🧠 Processing Complex Query")
        update_thinking_display()
        time.sleep(0.5)
        
        # Task allocation
        task_response = task_chain.invoke({'query': query, "metrics": get_metric_name_for_ciks(query_response.ciks)})
        thinking_steps.append(f"Tasks: {', '.join(task_response.tasks)}")
        thinking_steps.append(f"Companies: {', '.join(task_response.companies)}")
        thinking_steps.append(f"CIKs: {', '.join(map(str, task_response.ciks))}")
        thinking_steps.append(f"Metrics: {', '.join(task_response.metrics)}")
        update_thinking_display()

        # Financial Analysis
        if 'financial_metric_analysis' in task_response.tasks:
            thinking_steps.append("📈 Financial Analysis")
            update_thinking_display()
            time.sleep(0.5)
            
            datas = get_metric_for_all_ciks(task_response)
            thinking_steps.append(f"✅ Data fetched for: {', '.join(datas.keys())}")
            update_thinking_display()
            
            financial_analysis_response = financial_analysis_chain.invoke({"query": query, "data": datas})
            thinking_steps.append(f"Requires Visualization: {financial_analysis_response.require_visualization}")
            thinking_steps.append(f"Requires Performance Summary: {financial_analysis_response.require_summarization}")
            update_thinking_display()
            
            if financial_analysis_response.require_visualization:
                require_visualization = True
                visualization_response = visualization_chain.invoke({
                    'datas': datas,
                    'start_date': task_response.start_date,
                    'end_date': task_response.end_date,
                    'visualization_request': ''
                })
                
                verified_code = verify_chain.invoke({'code': visualization_response.code})
                if verified_code.startswith("```python") and verified_code.endswith("```"):
                    execute_python_code(verified_code[9:-3])
                elif verified_code.endswith("```"):
                    execute_python_code(verified_code[:-3])
                
                thinking_steps.append(f"✅ Generated visualizations: {', '.join(visualization_response.file_names)}")
                update_thinking_display()
            
                        
            if financial_analysis_response.require_summarization:
                require_summary = True
                performance_summary_response = performance_summary_chain.invoke({'query': query, 'datas': datas})
                thinking_steps.append("✅ Performance summary generated")
                update_thinking_display()

        # Narrative Analysis
        if 'narrative_analysis' in task_response.tasks:
            thinking_steps.append("📖 Narrative Analysis")
            update_thinking_display()
            time.sleep(0.5)
            
            narrative_analysis_response = narrative_analysis_chain.invoke({
                "query": query,
                "documents": get_relevant_douments(query, 15)
            })
            
            thinking_steps.append("✅ 10K documents analyzed")
            thinking_steps.append(f"Requires Strategic Recommendations: {narrative_analysis_response.require_strategic_recommendation}")
            update_thinking_display()
            
            if narrative_analysis_response.require_strategic_recommendation:
                require_strategic_recommendations = True
                strategic_reccomendation_response = strategic_reccomendation_chain.invoke({
                    'query': query,
                    'documents': get_relevant_douments(f"Give Strategic Recommendations for the query : {query}", 15),
                    'summary': narrative_analysis_response.narrative_report
                })
                thinking_steps.append("✅ Strategic recommendations generated")
                update_thinking_display()
            
            thinking_steps.append(f"Requires Risk Analysis: {narrative_analysis_response.require_risk_analysis}")
            update_thinking_display()
            
            if narrative_analysis_response.require_risk_analysis:
                require_risk_analysis = True
                risk_analysis_response = risk_analysis_chain.invoke({
                    'query': query,
                    'documents': get_relevant_douments(f"Risk factors associated with the query: {query}"),
                    'summary': narrative_analysis_response.narrative_report
                })
                thinking_steps.append("✅ Risk analysis completed")
                update_thinking_display()

        # Predictive Analysis
        if 'predictive_analysis' in task_response.tasks:
            thinking_steps.append("🔮 Predictive Analysis")
            update_thinking_display()
            time.sleep(0.5)
            
            require_predictions = True
            datas = get_metric_for_all_ciks(task_response)
            
            future_datas = process_and_forecast(datas)
            thinking_steps.append("✅ Forecasting completed")
            thinking_steps.append(f"Predictions = {future_datas}")
            thinking_steps.append(f"Forecasted Metrics: {', '.join(future_datas.keys())}")
            update_thinking_display()
            
            prediction_response = predcition_chain.invoke({"future_data": future_datas})
            prediction_image_file_names = plot_and_save_forecasts(datas, future_datas['holt'], future_datas['arima'])
            
            thinking_steps.append(f"✅ Prediction charts generated: {', '.join(prediction_image_file_names)}")
            update_thinking_display()

        # Generate concise answer before final report
        thinking_steps.append("💡 Generating Concise Answer")
        update_thinking_display()
        
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
            'risk_analysis': getattr(risk_analysis_response, 'content', "Risk Analysis not required.") if risk_analysis_response else "Risk Analysis not required.",
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

        
        thinking_steps.append("✅ Concise answer generated")
        update_thinking_display()
        
        # Final Report Generation
        thinking_steps.append("📋 Generating Final Report")
        update_thinking_display()
        time.sleep(0.5)
        
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
            'risk_analysis': getattr(risk_analysis_response, 'content', "Risk Analysis not required.") if risk_analysis_response else "Risk Analysis not required.",
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
        
        thinking_steps.append("✅ Report compiled successfully")
        update_thinking_display()
        
        verified_code = verify_chain.invoke({'code': final_response})
        
        if verified_code.startswith("```python") and verified_code.endswith("```"):
            execute_python_code(verified_code[9:-3])
        elif verified_code.startswith("```python"):
            code_finisher_response = code_finisher_chain.invoke({"code": verified_code})
            thinking_steps.append("⚠️ Code completion handled")
            update_thinking_display()
        elif verified_code.endswith("```"):
            execute_python_code(verified_code[:-3])
        else:
            execute_python_code(verified_code)
        
        thinking_steps.append("✅ Final report generated successfully")
        update_thinking_display()
    
    return thinking_steps, brief_answer

def financial_analysis_page():
    """Render the financial analysis page"""
    # Query input
    st.markdown("""
    <h3 style="color: black; font-weight: 700; font-family: 'Segoe UI', sans-serif;">
        💬 Enter Your Financial Query
    </h3>
""", unsafe_allow_html=True)


    st.markdown('<div class="custom-container">', unsafe_allow_html=True)

    with st.expander("📝 **Sample Queries**"):
        st.markdown("""
    <ul>
        <li>What are the key risks identified by Apple related to emerging technologies, such as artificial intelligence, and how has Apple's risk mitigation strategy evolved in response to these technologies?</li>
        <li>What happened to Apple's profit in 2020 vs 2021 and why?</li>
        <li>What will be the future Assets for Apple?</li>
        <li>How did Apple's total Assets vary from 2018 to 2024?</li>
        <li>How has Meta's debt profile changed over the last five years, and what implications does this have for its financial stability and investment capacity?</li>
        <li>How has Meta's debt profile changed over the last five years?</li>
        <li>What are the key risks identified by Apple related to emerging technologies?</li>
        <li>How has Meta's Revenue evolved over the past five years, and what strategic initiatives have driven significant changes in it?</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    query = st.text_area(
        "Query",
        placeholder="Enter your financial analysis query here...",
        height=100,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if analyze_button and query:
        # Get initial PDF count before starting analysis
        initial_pdf_count = get_initial_pdf_count()
        
        # Create placeholder for thinking process (right after analyze button)
        thinking_container = st.empty()
        
        # Run the analysis with real-time thinking process updates
        with st.spinner("🤖 AI agents are working on your query..."):
            thinking_steps, brief_answer = run_analysis_with_thinking_process(query, thinking_container)
        
        # After thinking process is complete, show the results
        st.markdown("---")
        
        # Show concise answer only after thinking process is complete
        if brief_answer and hasattr(brief_answer, 'points'):
            render_concise_answer(brief_answer.points)
        else:
            render_concise_answer()
        
        # Look for newly generated PDF files
        st.markdown("---")
        st.markdown("## 📄 Generated Report")
        
        # Check for new PDF files in Reports directory
        new_pdf = find_new_pdf(initial_pdf_count)
        
        if new_pdf:
            st.success(f"✅ New report generated: {new_pdf.name}")
            display_pdf(str(new_pdf))
        else:
            # Fallback: check Reports directory for any PDF files
            reports_dir = Path("./Reports")
            if reports_dir.exists():
                pdf_files = list(reports_dir.glob("*.pdf"))
                if pdf_files:
                    # Get the most recently modified PDF
                    latest_pdf = max(pdf_files, key=lambda p: p.stat().st_mtime)
                    st.success(f"✅ Found report: {latest_pdf.name}")
                    display_pdf(str(latest_pdf))
                else:
                    st.warning("⚠️ No PDF report found in ./Reports directory.")
            else:
                st.warning("⚠️ ./Reports directory not found. Please ensure the Reports folder exists.")
    
    elif analyze_button and not query:
        st.error("⚠️ Please enter a query to analyze.")

def architecture_page():
    """Render the multi-agent architecture page"""
    
    
    # Placeholder content for architecture page
    st.markdown("## 🏗️ System Architecture Overview")

    try:
        st.image("./Architecture.png", caption="Multi-Agent System Architecture", use_container_width=True)
    except FileNotFoundError:
        st.error("Architecture image not found.")
    
    st.markdown("""
    This page will contain detailed information about the multi-agentic architecture including:
    
    ### 🤖 Agent Types
    - **Query Classification Agent**: Determines query complexity and routing
    - **RAG Agent**: Retrieval-Augmented Generation for document processing
    - **Financial Analysis Agent**: Quantitative financial data analysis
    - **Visualization Agent**: Chart and graph generation
    - **Narrative Analysis Agent**: 10K document interpretation
    - **Risk Analysis Agent**: Risk factor identification and assessment
    - **Strategic Recommendation Agent**: Business strategy suggestions
    - **Prediction Agent**: Time series forecasting and predictions
    - **Report Writing Agent**: Comprehensive report compilation
    - **Code Verification Agent**: Code validation and execution
    
    ### 🔄 Agent Workflow
    - Multi-step processing pipeline
    - Dynamic task allocation based on query type
    - Inter-agent communication and data sharing
    - Quality assurance and verification layers
    
    ### 📊 Data Flow
    - Document retrieval and processing
    - Financial data extraction and transformation
    - Analysis result aggregation
    - Report generation and visualization
    """)
    
    st.info("🚧 Architecture diagrams and detailed documentation will be added here.")

def main():
    render_navigation()

    # Render the page content first
    if st.session_state.current_page == 'Financial Analysis':
        financial_analysis_page()
    elif st.session_state.current_page == 'Architecture':
        architecture_page()

if __name__ == "__main__":
    main()