# 📄 Resume Screening & Ranking System

## Project Overview
An NLP-based Machine Learning system that automatically screens,
scores, and ranks resumes based on a given job description.
Built using Python with TF-IDF and Cosine Similarity.

## Business Problem
Hiring teams receive hundreds of resumes for a single job role.
Manually reading each resume is slow, inconsistent, and error-prone.
This system automatically ranks candidates so recruiters can focus
on top candidates instead of reading every resume.

## Tools Used
- Python, Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- NLTK (Text Preprocessing)
- Matplotlib (Visualization)

## Key Features
| Feature | Description |
|---------|-------------|
| Text Cleaning | Lowercasing, stopword removal |
| Skill Extraction | NLP-based keyword matching |
| Similarity Scoring | TF-IDF + Cosine Similarity |
| Candidate Ranking | Combined score ranking |
| Skill Gap Analysis | Missing skills identification |

## Results
| Metric | Value |
|--------|-------|
| Total Candidates | 8 |
| Shortlisted | 5 |
| Rejected | 3 |
| Required Skills | 19 |
| Top Candidate | Candidate_03_Sneha (47.8%) |

## Scoring Logic
- 60% — Resume to Job Description Similarity (TF-IDF)
- 40% — Skill Match Percentage

## Business Impact
- **Recruiter**: Shortlist top candidates in seconds
- **HR Manager**: Identify skill gaps before interviews
- **HR-Tech Startup**: Automate resume screening at scale

## Deliverables
- `resume_screening.py` — Complete NLP screening model
- `resume_screening_dashboard.png` — 4-chart visual dashboard
- `README.md` — Business explanation

## Author
BHAVANI | Future Interns Machine Learning Internship
