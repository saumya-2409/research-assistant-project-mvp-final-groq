from typing import List, Dict, Optional
from datetime import datetime

#  ==================== CLUSTERING & GAP ANALYSIS (SAME AS BEFORE) ====================
class ImprovedClusterer:
    """Simple area-based clusterer for fast research theme grouping"""
    
    def __init__(self, query: str = ""):
        self.query = query or "general research"  # Store query for relevance biasing

    def cluster_papers(self, papers: List[Dict], query: str = "") -> Dict[int, Dict]:
        effective_query = query or self.query  # Use passed or stored query
        
        if len(papers) < 2:
            return {0: {'name': 'All Papers', 'description': f'Complete research collection relevant to "{effective_query}"', 'papers': papers}}
        
        def identify_research_area(paper: Dict, query: str) -> str:
            abstract = paper.get('abstract', '') or paper.get('title', '')
            text = (abstract + ' ' + query).lower()
            # Keyword mapping for areas (tied to query; expand as needed)
            areas = {
                'machine learning': ['machine learning', 'ml', 'neural', 'deep learning', 'model', 'algorithm'],
                'stock prediction': ['stock', 'prediction', 'financial', 'market', 'forecast', 'trading'],
                'ai applications': ['ai', 'artificial intelligence', 'nlp', 'cv', 'robotics'],
                'other': []  # Default
                }
            for area, keywords in areas.items():
                if any(kw in text for kw in keywords):
                    return area
            return 'general research'  # Fallback

        # Group by research areas, biased toward query
        area_groups = {}
        for paper in papers:
            area = identify_research_area(paper, effective_query)  # Use simple function with query
            if area not in area_groups:
                area_groups[area] = []
            area_groups[area].append(paper)
        
        # Create clusters from groups
        clusters = {}
        for i, (area, papers_list) in enumerate(area_groups.items()):
            # Calculate stats
            avg_citations = sum(int(p.get('citations') or 0) for p in papers_list) / len(papers_list)
            years = [int(p.get('year') or datetime.now().year) for p in papers_list]
            avg_year = sum(years) / len(years) if years else datetime.now().year
            
            # Query-aware description
            description = f"Research papers focusing on {area.lower()} methodologies and applications, relevant to '{effective_query}'"
            
            clusters[i] = {
                'name': area,
                'description': description,
                'paper_count': len(papers_list),
                'avg_citations': round(avg_citations, 1),
                'avg_year': round(avg_year),
                'papers': papers_list
            }
        
        return clusters

class ResearchGapAnalyzer:
    def analyze_gaps(self, papers: List[Dict]) -> Dict[str, List[str]]:
        gaps = {
            'methodological_gaps': [],
            'evaluation_gaps': [], 
            'application_gaps': [],
            'theoretical_gaps': []
        }
        
        # Use extracted content for better gap analysis
        all_content = []
        for paper in papers:
            content = paper.get('extracted_content', '') or paper.get('abstract', '')
            all_content.append(content)
        
        combined_content = ' '.join(all_content).lower()
        
        if 'dataset' in combined_content or 'limited' in combined_content:
            gaps['methodological_gaps'].extend([
                'Limited dataset diversity across different domains and applications',
                'Lack of standardized evaluation protocols for cross-method comparison',
                'Insufficient attention to computational efficiency and scalability issues'
            ])
        
        if 'experiment' in combined_content or 'evaluation' in combined_content:
            gaps['evaluation_gaps'].extend([
                'Need for more comprehensive real-world testing scenarios',
                'Lack of longitudinal studies assessing long-term performance',
                'Limited evaluation on edge cases and adversarial conditions'
            ])
        
        if 'application' in combined_content or 'real-world' in combined_content:
            gaps['application_gaps'].extend([
                'Gap between laboratory results and industrial deployment',
                'Limited integration with existing systems and workflows',
                'Insufficient consideration of regulatory and ethical constraints'
            ])
        
        if 'theoretical' in combined_content or 'analysis' in combined_content:
            gaps['theoretical_gaps'].extend([
                'Lack of theoretical foundations for empirical observations',
                'Limited understanding of failure modes and boundary conditions',
                'Insufficient mathematical analysis of convergence properties'
            ])
        
        return gaps
