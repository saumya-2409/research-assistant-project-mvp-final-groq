import logging

logger = logging.getLogger(__name__)

# ArXiv
try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False
    logger.warning("ArXiv not found. Install via 'pip install arxiv'")

# PyPDF2
try:
    from PyPDF2 import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# BeautifulSoup
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
