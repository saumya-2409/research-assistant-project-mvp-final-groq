import streamlit as st
import warnings
import os
import logging
import time
import plotly.express as px
import concurrent.futures
from dotenv import load_dotenv

# Modular Imports
from embedding_utils import compute_relevance_embedding_score
from summarizer import FullPaperSummarizer
from fetchers import IntelligentMultiSourceFetcher
from src.config import get_api_key
from analysis import ResearchGapAnalyzer
from ui import load_custom_css, render_paper_ui, render_suggested_paper

load_dotenv()

# Optional Imports
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False


# Config
warnings.filterwarnings("ignore")
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

st.set_page_config(page_title="AI Research Assistant", page_icon=" § ", layout="wide", initial_sidebar_state="expanded")

# Load CSS from ui.py
load_custom_css()

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = FullPaperSummarizer()
    print("[App Debug] Summarizer singleton created")
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = []
if 'full_text_papers' not in st.session_state:
    st.session_state.full_text_papers = []
if 'suggested_papers' not in st.session_state:
    st.session_state.suggested_papers = []
if 'clusters' not in st.session_state:
    st.session_state.clusters = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False


# ==================== MAIN APPLICATION ====================

# Header
st.markdown("""
<div class="main-header">
    <h1>AI Research Assistant</h1>
    <p>Extract, Analyze, and Summarize Research Papers</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 Research Parameters")
    
    query = st.text_input(
        "Research Topic",
        placeholder="e.g., Generative AI in Healthcare",
        help="Enter the specific topic you want to analyze"
    )
    
    st.markdown("### 📚 Data Sources")
    st.caption("Select at least one source to fetch papers from.")

    # Create two columns for the checkboxes
    col_arxiv, col_semantic = st.columns(2)

    with col_arxiv:
        use_arxiv = st.checkbox("arXiv", value=True)
        if use_arxiv and not ARXIV_AVAILABLE:
            st.error("⚠️ ArXiv library missing!")
            
    with col_semantic:
        use_semantic = st.checkbox("Semantic Scholar", value=True) 

    

    st.markdown("### ⚙️ Configuration")
    papers_per_source = st.slider(
        "**Papers to Fetch (per source)**",
        min_value=10, max_value=100, value=30, step=10,
        help="Higher values provide more comprehensive results but take longer to process."
    )
    
    # Build source list
    sources = []
    if use_arxiv: sources.append('arxiv')
    if use_semantic: sources.append('semantic_scholar')
    
    if sources:
        expected_total = papers_per_source * len(sources)
        st.info(f"📊 **Estimate:** Analyzing up to **{expected_total}** papers.")
    else:
        st.error("❌ Please select at least one source!")
    
    # --- Buttons Layout (Side-by-Side) ---
    col_start, col_clear = st.columns(2)

    with col_start:
        # Start Analysis Button
        start_btn = st.button(
            "🚀 Start", 
            type="primary", 
            disabled=st.session_state.processing or not sources or not query,
            use_container_width=True,
            help="Begin the intelligent research analysis."
        )

    with col_clear:
        # Clear Results Button
        clear_btn = st.button(
            "🗑️ Clear", 
            type="secondary",
            use_container_width=True,
            help="Reset all results and start fresh."
        )

    # --- Logic Handling ---
    
    # Handle Start Logic
    if start_btn:
        if query.strip() and sources:
            st.session_state.processing = True
        
        # Create the status container
        with st.status("🚀 Initiating Research Sequence...", expanded=True) as status:
            try:
                # Step 1: Fetching
                st.write("1️⃣ **Scouting Sources:** Fetching latest research from Sources...")
                start_time = time.time()
                fetcher = IntelligentMultiSourceFetcher()
                papers = fetcher.fetch_papers(query, sources, papers_per_source)
                fetch_time = time.time() - start_time

                if len(papers) == 0:
                    status.update(label="❌ No papers found! Try distinct keywords.", state="error")
                    st.session_state.processing = False
                    st.stop()

                # Step 2: Filtering
                st.write("3️⃣ **Smart Filter:** Removing duplicates and irrelevant results...")
                valid_papers = []
                for p in papers:
                    # 1. Compute Score
                    score = compute_relevance_embedding_score(query, p)
                    p['relevance_score'] = round(score, 3)                    
                    # 2. Strict Filter (Threshold 0.35)
                    # This drops random/irrelevant papers entirely
                    if score >= 0.35: 
                        valid_papers.append(p)

                if not valid_papers:
                    status.update(label="⚠️ No relevant papers found after filtering.", state="error")
                    st.warning("Try broader terms or check spelling.")
                    st.session_state.processing = False
                    st.stop()
                
                # 3. Sort by Relevance + Year (Newest & Most Relevant first)
                valid_papers.sort(key=lambda x: (x.get('relevance_score', 0.0), x.get('year') or 0), reverse=True)
                st.info(f"Processing {len(valid_papers)} highly relevant papers...")
                
                summarizer_instance = st.session_state.summarizer

                def process_single_paper(p):
                    summary = summarizer_instance.summarize_paper(p, use_full_text=True, query=query)
                    p['ai_summary'] = summary if isinstance(summary, dict) else {"summary": str(summary)}
                    p['accessibility'] = summary.get('accessibility') if isinstance(summary, dict) else 'inaccessible'
                    p['abstract_summary_status'] = summary.get('abstract_summary_status', 'inaccessible') if isinstance(summary, dict) else 'inaccessible'
                    return p
                
                # Step 4: Summary Generation
                with st.spinner("3️⃣ **Deep Reading:** Generating AI summaries..."):
                    starttime = time.time()
                    papers_data = []
                    full_text_papers = []
                    suggested_papers = []

                    # PARALLEL EXECUTION (Max 3 workers to respect rate limits)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                        results = list(executor.map(process_single_paper, valid_papers))

                    for paper in results:
                        if paper.get('accessibility') == 'accessible':
                            papers_data.append(paper)
                            if paper.get('extracted_content') or paper.get('abstract_summary_status') == 'generated_from_fulltext':
                                full_text_papers.append(paper)
                        else:
                            suggested_papers.append(paper)

                    # Save timings and state
                    summarytime = time.time() - starttime
                    status.update(label="✅ Research Analysis Complete!", state="complete", expanded=False)

                    # Update State 
                    st.session_state.papers_data = papers_data
                    st.session_state.full_text_papers = full_text_papers
                    st.session_state.suggested_papers = suggested_papers
                    st.session_state.clusters = {}
                    st.session_state.processing = False
                        
                    st.balloons()
                    
            except Exception as e:
                status.update(label="❌ An error occurred during analysis.", state="error")
                st.error(f"Error during analysis: {str(e)}")
                st.session_state.processing = False

    # Handle Clear Logic
    if clear_btn:
        st.session_state.papers_data = []
        st.session_state.full_text_papers = []
        st.session_state.suggested_papers = []
        st.session_state.clusters = {}
        st.session_state.current_page = 1  # Reset pagination
        st.rerun()
    
    # Footer
    st.markdown("---")
    if not BEAUTIFULSOUP_AVAILABLE:
        st.caption("⚠️ **Note:** Install `beautifulsoup4` for better web scraping.")
    st.caption("© 2025 Intelligent Research Assistant")

# ==================== MAIN CONTENT ====================
if st.session_state.papers_data:
    # Enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(st.session_state.papers_data)}</div>
            <div class="metric-label">Papers Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        accessible_count = len(st.session_state.full_text_papers)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{accessible_count}</div>
            <div class="metric-label">Accessible Papers</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        extracted_count = len([p for p in st.session_state.papers_data if p.get('extracted_content')])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{extracted_count}</div>
            <div class="metric-label">Content Extracted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        sources_count = len(set([p.get('source', 'unknown') for p in st.session_state.papers_data]))
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{sources_count}</div>
            <div class="metric-label">Sources Used</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Clean tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        " Dashboard", 
        " Papers & Summaries", 
        " Research Gaps",
        " Restricted Reading"
    ])
    
    with tab1:
        st.markdown("### Research Dashboard")
        st.markdown("*Intelligent analysis with enhanced content extraction*")
        
        if st.session_state.clusters:
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_names = []
                cluster_counts = []
                for cluster_id, cluster_info in st.session_state.clusters.items():
                    cluster_names.append(cluster_info['name'])
                    cluster_counts.append(cluster_info.get('papercount', len(cluster_info.get('papers', []))))

                
                colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#ddd6fe', '#ede9fe', '#667eea', '#764ba2']
                
                fig = px.pie(
                    values=cluster_counts,
                    names=cluster_names,
                    title="Research Areas Distribution",
                    color_discrete_sequence=colors
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    showlegend=True
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                citation_data = []
                area_names = []
                for cluster_info in st.session_state.clusters.values():
                    citation_data.append(cluster_info.get('avg_citations', 0))
                    area_names.append(cluster_info['name'])
                
                fig = px.bar(
                    x=citation_data,
                    y=area_names,
                    orientation='h',
                    title="Average Citations by Research Area",
                    color=citation_data,
                    color_continuous_scale=["#f8fafc", "#667eea", "#764ba2"]
                )
                fig.update_layout(
                    font_family="Inter",
                    title_font_size=16,
                    font_size=12,
                    xaxis_title="Average Citations",
                    yaxis_title="Research Area"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Enhanced cluster cards
            st.markdown("### Research Themes")
            
            for cluster_id, cluster_info in st.session_state.clusters.items():
                # Filter papers to those relevant to query (simple keyword match for safety)
                relevant_papers = [p for p in cluster_info['papers'] if any(word in (p.get('title', '') + p.get('abstract', '')).lower() for word in query.lower().split())]
                paper_count = len(relevant_papers)  # Safe count from filtered papers list
                extracted_in_cluster = len([p for p in relevant_papers if p.get('extracted_content')])  # Update count
                cluster_info['papers'] = relevant_papers  # Override for display
                
                st.markdown(f"""
                <div class="cluster-card">
                    <div class="cluster-title">{cluster_info['name']}</div>
                    <p style="color: #64748b; margin-bottom: 1rem;">{cluster_info['description']}</p>
                    <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {paper_count} papers
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {extracted_in_cluster} content extracted
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {cluster_info.get('avg_citations', 0)} avg citations
                        </span>
                        <span style="background: #f1f5f9; padding: 6px 10px; border-radius: 6px; font-size: 0.85rem; color: #475569;">
                            {cluster_info.get('avg_year', 2024)}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Complete analysis to see research themes and dashboard metrics.")
   
    with tab2:
        st.markdown("### Papers & Summaries")
        papers_data = st.session_state.get("papers_data", [])
        
        if not papers_data:
            st.info("No accessible papers available. Try another query or enable more sources.")
        else:
            # --- PAGINATION CONFIG ---
            ITEMS_PER_PAGE = 10
            
            if 'current_page' not in st.session_state:
                st.session_state.current_page = 1
                
            total_count = len(papers_data)
            total_pages = max(1, (total_count + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE)
            
            if st.session_state.current_page > total_pages:
                st.session_state.current_page = 1

            # --- INFO TEXT (TOP) ---
            # Just text, no buttons here
            st.markdown(f"""
            <div style="color: #64748b; margin-bottom: 15px; font-size: 0.95rem;">
                Showing Page <strong>{st.session_state.current_page}</strong> of <strong>{total_pages}</strong> 
                <span style="color: #94a3b8;">({total_count} total papers)</span>
            </div>
            """, unsafe_allow_html=True)

            # --- DISPLAY CURRENT PAGE ---
            start_idx = (st.session_state.current_page - 1) * ITEMS_PER_PAGE
            end_idx = start_idx + ITEMS_PER_PAGE
            
            current_batch = papers_data[start_idx:end_idx]
            
            for paper in current_batch:
                render_paper_ui(paper)
            
            st.markdown("---")

            # --- PAGINATION CONTROLS (BOTTOM ONLY) ---
            c1, c2, c3 = st.columns([1, 3, 1])
            
            with c1:
                if st.button("← Previous", key="btn_prev", disabled=(st.session_state.current_page == 1), use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
            
            with c2:
                # Centered page indicator for easy reading at bottom
                st.markdown(f"""
                <div style="text-align: center; padding-top: 8px; color: #64748b; font-weight: 500;">
                    Page {st.session_state.current_page}
                </div>
                """, unsafe_allow_html=True)

            with c3:
                if st.button("Next →", key="btn_next", disabled=(st.session_state.current_page == total_pages), use_container_width=True):
                    st.session_state.current_page += 1
                    st.rerun()

    with tab3:
        st.markdown("### Research Gaps Analysis")
        st.markdown("*Enhanced gap analysis using extracted content*")
        
        if st.session_state.papers_data:
            gap_analyzer = ResearchGapAnalyzer()
            gaps = gap_analyzer.analyze_gaps(st.session_state.papers_data)
            
            for gap_type, gap_list in gaps.items():
                if gap_list:
                    gap_title = gap_type.replace('_', ' ').title()
                    
                    st.markdown(f"""
                    <div class="gap-card">
                        <h4 style="margin-bottom: 0.8rem; color: #7c3aed;">{gap_title}</h4>
                        <ul style="color: #374151; margin: 0; padding-left: 1.5rem;">
                    """, unsafe_allow_html=True)
                    
                    for gap in gap_list:
                        st.markdown(f"<li style='margin-bottom: 0.5rem;'>{gap}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.info("Complete paper analysis to identify research gaps and opportunities.")
    
    with tab4:
        st.markdown("### Restricted Reading")

        
        st.markdown("""
        <div class="warning-box">
            <strong>Restricted Access:</strong> Access to these papers requires a subscription or paid access.
        </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.suggested_papers:
            st.success("Excellent! All papers are accessible. Check 'Papers & Summaries' for complete analysis with extracted content.")
        else:
            st.markdown(f"**{len(st.session_state.suggested_papers)} papers requiring paid/institutional access**")
            
            for i, paper in enumerate(st.session_state.suggested_papers, 1):
                render_suggested_paper(paper)

else:
    # Enhanced welcome screen
    st.markdown("### Intelligent Research Analysis")
    
    steps = [
        {
            'title': 'Enter Research Topic',
            'description': 'Type your research keywords for intelligent multi-source analysis',
            'expected': 'Focused topics yield better content extraction results'
        },
        {
            'title': 'Select Enhanced Sources', 
            'description': 'Choose from real APIs with intelligent access detection capabilities',
            'expected': f'ArXiv: {"" if ARXIV_AVAILABLE else ""}, Semantic Scholar:'
        },
        {
            'title': 'Set Paper Count',
            'description': 'Choose papers per source (15-100) - system will intelligently analyze accessibility',
            'expected': 'More papers = comprehensive analysis but longer processing time'
        },
        {
            'title': 'Start Intelligent Analysis',
            'description': 'AI system will fetch, extract content, detect access, and generate enhanced summaries',
            'expected': '30-90 seconds for complete intelligent analysis with content extraction'
        }
    ]

    col1, col2 = st.columns(2)

    for idx, step in enumerate(steps):
        column = col1 if idx % 2 == 0 else col2
        with column:
            st.markdown(f"""
            <div class="welcome-step">
                <div style="display: flex; align-items: flex-start;">
                    <div class="step-number">{idx + 1}</div>
                    <div>
                        <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">{step['title']}</h4>
                        <p style="color: #64748b; margin: 0 0 0.8rem 0; line-height: 1.5;">
                            {step['description']}
                        </p>
                        <div style="background: #f1f5f9; padding: 0.5rem 0.8rem; border-radius: 6px; font-size: 0.9rem; color: #475569;">
                            <strong>Expected:</strong> {step['expected']}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Dependencies and capabilities
    st.markdown("### System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if ARXIV_AVAILABLE:
            st.success(" **ArXiv API** - Real papers available")
        else:
            st.error("**ArXiv** - Install: `pip install arxiv`")
    
    with col2:
        st.success("**Semantic Scholar** - Enhanced API ready")
    
    with col3:
        if BEAUTIFULSOUP_AVAILABLE:
            st.success("**Content Extraction** - Advanced parsing")
        else:
            st.warning("**Install BeautifulSoup** - `pip install beautifulsoup4`")
    
    st.markdown("### Example Results")
    st.markdown("**Query:** `deep learning transformers` ")
    st.markdown("**Expected:** 60-300 papers ")
    st.markdown("**Intelligent Analysis:** Content extraction, access detection, enhanced summaries ")
    st.markdown("**Time:** 30-90 seconds ")

# Footer
if not st.session_state.processing:
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'> Intelligent research assistant with content extraction</div>", unsafe_allow_html=True)

 
