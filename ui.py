import streamlit as st
from typing import Dict

# --- CSS STYLES ---
def load_custom_css():
    # BEAUTIFUL DESIGN CSS (PRESERVED)
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            font-weight: 400;
        }
        
        .stApp {
            background: linear-gradient(135deg, #fafbfc 0%, #f4f6f8 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        
        .main-header h1 {
            font-size: 2.2rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            text-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .main-header p {
            font-size: 1rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .css-1d391kg {
            background: white;
            border-right: 1px solid #e2e8f0;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.03);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: white;
            padding: 6px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            margin-bottom: 1.5rem;
            border: 1px solid #f1f5f9;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            padding: 0 20px;
            background: transparent;
            border-radius: 8px;
            color: #64748b;
            font-weight: 400;
            transition: all 0.2s ease;
            border: none;
            font-size: 0.95rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        }
        
        .metric-card {
            background: white;
            border: 1px solid #f1f5f9;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            transition: transform 0.15s ease;
            margin: 6px;
        }
        
        .metric-card:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.08);
        }
        
        .metric-number {
            font-size: 2.2rem;
            font-weight: 500;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            color: #64748b;
            font-weight: 400;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .summary-section {
            background: linear-gradient(135deg, #fafbfc 0%, #f8fafc 100%);
            border-left: 3px solid #667eea;
            padding: 1rem;
            margin: 0.6rem 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
        }
        
        .summary-section strong {
            color: #1e293b;
            font-weight: 500;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
            border: 1px solid #f59e0b;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(245, 158, 11, 0.06);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
            font-size: 0.95rem;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.25);
        }
        
        .status-full {
            background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
            color: #15803d;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid #86efac;
        }
        
        .status-abstract {
            background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
            color: #a16207;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid #facc15;
        }
        
        .status-extracted {
            background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
            color: #3730a3;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid #a5b4fc;
        }
        
        .cluster-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            border: 1px solid #f1f5f9;
            transition: all 0.2s ease;
            margin: 1rem 0;
        }
        
        .cluster-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.08);
        }
        
        .cluster-title {
            font-size: 1.1rem;
            font-weight: 500;
            color: #1e293b;
            margin-bottom: 0.5rem;
        }
        
        .welcome-step {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        }
        
        .step-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 500;
            font-size: 0.9rem;
            margin-right: 0.8rem;
        }
        
        .gap-card {
            background: white;
            border: 1px solid #f1f5f9;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            border-left: 4px solid #8b5cf6;
        }
        
        .suggested-card {
            background: white;
            border: 1px solid #f1f5f9;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
            border-left: 3px solid #f59e0b;
        }
        
        .stTextInput input {
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            padding: 0.7rem;
            font-size: 0.95rem;
            font-weight: 400;
            transition: all 0.2s ease;
        }
        
        .stTextInput input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
        }
        
        .source-status {
            background: #f0f9ff;
            border: 1px solid #38bdf8;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin: 0.5rem 0;
        }
        
        .extraction-status {
            background: #f0f4ff;
            border: 1px solid #6366f1;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)

# ==================== RENDER FUNCTIONS ====================
def render_paper_ui(paper: dict):
    """
    Render a clean, user-friendly view (no JSON) showing only non-empty LLM-extracted fields.
    Matches the requested format (header, metadata, content status, numbered Research Paper Summary,
    and access links). Ignores empty fields.
    """
    # Helpers
    def safe_str(x):
        return "" if x is None else str(x)

    # --- 1. Data Preparation ---
    
    #Header 
    title = safe_str(paper.get("title") or (paper.get("ai_summary") or {}).get("Title") or "Research Paper Summary")
    year = safe_str(paper.get("year") or (paper.get("ai_summary") or {}).get("Year",""))
    cites = safe_str(paper.get("citations") or paper.get("citation_count") or (paper.get("ai_summary") or {}).get("Citations",""))
    source = safe_str(paper.get("source") or paper.get("fetch_source") or (paper.get("ai_summary") or {}).get("Source",""))
       
    # Status Badges
    is_abstract_only = paper.get("abstract_summary_status") == "generated_from_abstract"
    status_emoji = "⚡" if is_abstract_only else "📄"

    # Author Formatting
    authors_list = paper.get("authors") or (paper.get("ai_summary") or {}).get("Authors") or []
    if isinstance(authors_list, list):
        authors = ", ".join(authors_list)
    else:
        authors = safe_str(authors_list)   

    # Extract AI Summary Data
    summary = paper.get("ai_summary") or paper.get("abstract_summary") or {}

    # Data extraction with fallbacks
    problem = summary.get("Research_Problem") or summary.get("problem_statement") or summary.get("Purpose") or ""
    objective = summary.get("Research_Objective") or summary.get("objective") or ""
    implications = summary.get("Aim_of_Study") or summary.get("reusability_practical_value") or ""
    limitations = summary.get("limitations_and_future_work") or summary.get("limitations") or []     
    # Methodology
    mma = summary.get("Methodology_Approach") or {}  
    method = mma.get("Method") if isinstance(mma, dict) else summary.get("Methodology_Approach", {}).get("Method") or ""
    process = mma.get("Process") if isinstance(mma, dict) else summary.get("Methodology_Approach", {}) or ""
    data_handling = mma.get("Data_Handling") if isinstance(mma, dict) else summary.get("Methodology_Approach", {}).get("Data_handling") or ""
    results_format = mma.get("Results_Format") if isinstance(mma, dict) else summary.get("Methodology_Approach", {}).get("Results_Format") or ""          
    #Key Findings
    findings = summary.get("Key_Findings") or summary.get("key_findings") or []
    if isinstance(findings, str): findings = [findings]

    # Button Links
    working_url = paper.get('working_url') or paper.get('url')
    pdf_url = paper.get('pdf_url')

    # --- 2. UI Render ---
    with st.expander(f"{status_emoji} {title}", expanded=False):

        # --- A. Header Badges ---
        if is_abstract_only:
            st.markdown("""<span style="background: #fef3c7; color: #b45309; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500;">⚡ Generated from Abstract (Full Text Unavailable)</span>""", unsafe_allow_html=True)
        else:
            st.markdown("""<span style="background: #dcfce7; color: #15803d; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 500;">📄 Full Content Analysis</span>""", unsafe_allow_html=True)
                
 
        # Top Row: Metadata Badges
        c1, c2, c3 = st.columns([1, 1, 2])
        c1.markdown(f"**📅 Year:** {year}")
        c2.markdown(f"**🎓 Citations:** {cites}")
        c3.markdown(f"**🏛️ Source:** {source}")
        st.markdown(f"**✍️ Authors:** {authors}")
        st.divider()

        # --- B. The 6-Block Grid ---

        # ROW 1: The "Why" (Context & Goal)
        r1c1, r1c2 = st.columns(2, gap="medium")
        with r1c1:
            #Problem Statement
            if problem:
                st.info(f"**🧐 Problem Statement**\n\n{problem}")
            else:
                st.info("**🧐 Problem Statement**\n\n*Not explicitly extracted.*")

        with r1c2:
            # Objective
            if objective:
                st.success(f"**🎯 Research Objective**\n\n{objective}")
            else:
                st.success("**🎯 Research Objective**\n\n*Not explicitly extracted.*")

        # ROW 2: The "What" (Method & Findings)
        r2c1, r2c2 = st.columns(2, gap="medium")
        with r2c1:
            # Methodology 
            with st.container(border=True):   
                st.markdown("#### ⚙️ Methodology")                    
                if method or process or data_handling or results_format:
                    if method: st.markdown(f"- **Method:** {method}")
                    if process: st.markdown(f"- **Process:** {process}")
                    if data_handling: st.markdown(f"- **Data handling:** {data_handling}")
                    if results_format: st.markdown(f"- **Results format:** {results_format}")       
                else:
                    st.write(str(mma) if mma else "No details extracted.")
        with r2c2:
            with st.container(border=True):
                st.markdown("#### 💡 Key Findings")
                if findings:
                    for f in findings:
                        st.markdown(f"- {f}")
                else:
                    st.write("No specific findings extracted.")
        
        # ROW 3: The "So What" (Value & Limits)
        r3c1, r3c2 = st.columns(2, gap="medium")
        with r3c1:
            with st.container(border=True):
                st.markdown("#### 🚀 Implications & Value")
                if implications:
                    st.write(implications)
                else:
                    st.caption("No specific implications extracted.")

        with r3c2:
            if limitations:
                with st.container(border=True):
                    st.markdown("#### ⚠️ Limitations & Future Work ")
                    if isinstance(limitations, list):
                        for l in limitations: st.markdown(f"- {l}")
                    else:
                        st.write(limitations)
            else:
                st.caption("No limitations extracted.")
                       
        # Keywords
        keywords = summary.get("Keywords") or paper.get("keywords") or []
        if isinstance(keywords, str) and keywords.strip():
            st.caption(f"**Keywords:** {keywords}")
        elif isinstance(keywords, list) and keywords:
            st.caption(f"**Keywords:** {', '.join([str(k) for k in keywords[:20]])}")

        
        # --- C. Action Buttons ---
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        working_url = paper.get('working_url') or paper.get('url')
        access_type = paper.get('access_type', 'direct')
        
        button_label = (
            "Access Paper (PDF)"
            if access_type == 'direct_pdf' 
            else "Access Full Paper"
        )

        if working_url:
            with col1:
                st.link_button(
                    button_label, 
                    working_url,
                    use_container_width=True
                )

        # Direct PDF fallback
        pdf_url = paper.get('pdf_url')
        if pdf_url and pdf_url != working_url:
            with col2:
                st.link_button(
                    "Direct PDF Download", 
                    pdf_url,
                    use_container_width=True
                )

def render_suggested_paper(paper: Dict):
    """Render truly restricted paper card"""
    
    st.markdown(f"""
    <div class="suggested-card">
        <h4 style="margin-bottom: 0.5rem; color: #1e293b;">{paper.get('title', 'Unknown Title')}</h4>
        <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 0.8rem;">
            <strong>Authors:</strong> {', '.join(paper.get('authors', ['Unknown'])[:3])}
            {' et al.' if len(paper.get('authors', [])) > 3 else ''}<br>
            <strong>Source:</strong> {paper.get('source', 'Unknown')} | 
            <strong>Year:</strong> {paper.get('year', 0)} | 
            <strong>Citations:</strong> {paper.get('citations', 0)}
        </p>
        <a href="{paper.get('url', '#')}" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">
            Requires Subscription Access
        </a>
    </div>
    """, unsafe_allow_html=True)
