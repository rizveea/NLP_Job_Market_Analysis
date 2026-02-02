# NLP-Powered Job Matching System

## Business Context
Matching supply with demand is a core supply chain challenge—whether it's products to customers or candidates to roles. This project applies **Natural Language Processing (NLP)** to automate the matching of job seekers with relevant opportunities, demonstrating text analytics skills applicable to supplier matching, contract analysis, and procurement document processing.

## Project Highlights

### Problem Addressed
Built an **intelligent matching engine** that analyzes unstructured text to rank job opportunities based on resume fit, using techniques transferable to:
- Supplier capability matching
- RFP/RFQ response analysis
- Contract clause extraction

### Key Outcomes
- **Multi-Factor Scoring**: Combined keyword matching, topic similarity, and semantic understanding
- **Scalable Pipeline**: Processed 1,000+ job descriptions against candidate profiles
- **Configurable Weights**: Adjustable scoring to prioritize different matching criteria
- **Interpretable Results**: Breakdown of match scores by component for transparency

## Methodology
| Technique | Purpose |
|-----------|---------|
| **TF-IDF** | Keyword importance scoring between documents |
| **LDA Topic Modeling** | Latent topic extraction (e.g., "Data Engineering", "Machine Learning") |
| **Document Embeddings** | Semantic similarity beyond exact keyword matches |

## Technologies
| Category | Tools |
|----------|-------|
| NLP | NLTK, SpaCy, Gensim |
| ML | Scikit-learn (TF-IDF, LDA) |
| Document Processing | PyPDF2, python-docx |
| Analytics | Python, Pandas, NumPy |

## How to Run
```bash
pip install numpy pandas scikit-learn gensim nltk spacy pypdf2 python-docx
python JobMain.py --resume my_resume.pdf --jobs jobs.json --top 10
```

## Files
- `JobMain.py` - Main execution script
- `JobMatcher.py` - Core matching algorithm
- `ResumeExtractor.py` - Document parsing utilities
- `demo.py` - Example usage demonstration

## Skills Demonstrated
- **Natural Language Processing** - Text preprocessing, tokenization, topic modeling
- **Machine Learning** - TF-IDF, LDA, document embeddings, similarity metrics
- **Text Analytics** - Unstructured data processing, information extraction
- **Python Development** - Modular code design, CLI applications
- **Document Processing** - PDF/DOCX parsing, text extraction

