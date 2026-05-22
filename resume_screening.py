# ============================================
# RESUME SCREENING & RANKING SYSTEM
# Task 3 - Future Interns ML Internship
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: JOB DESCRIPTION
# ============================================

job_description = """
We are looking for a Data Scientist with strong Python programming skills.
The ideal candidate should have experience in machine learning, deep learning,
and natural language processing. Must know pandas, numpy, scikit-learn,
TensorFlow or PyTorch. Experience with data visualization using matplotlib
or seaborn is required. Knowledge of SQL databases and cloud platforms like
AWS or Azure is a plus. Strong analytical and problem solving skills needed.
Experience with Git version control and agile methodology preferred.
"""

# Required skills for the job
required_skills = [
    'python', 'machine learning', 'deep learning', 'nlp',
    'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'matplotlib', 'seaborn', 'sql', 'aws', 'azure', 'git',
    'data visualization', 'statistics', 'regression', 'classification'
]

print("=" * 60)
print("RESUME SCREENING & RANKING SYSTEM")
print("Job Role: Data Scientist")
print("=" * 60)

# ============================================
# STEP 2: SIMULATED RESUMES
# ============================================

resumes = {
    "Candidate_01_Priya": """
        Experienced Data Scientist with 3 years of experience.
        Proficient in Python, pandas, numpy, scikit-learn, and matplotlib.
        Strong knowledge of machine learning algorithms including regression
        and classification. Experience with TensorFlow and deep learning.
        Used SQL for data extraction and Git for version control.
        Worked on NLP projects using NLTK. AWS certified professional.
        Strong analytical and problem solving skills.
    """,

    "Candidate_02_Rahul": """
        Software Engineer with 2 years experience in Python development.
        Basic knowledge of pandas and numpy for data manipulation.
        Familiar with machine learning concepts and scikit-learn library.
        Experience with SQL databases and Git version control.
        No experience with deep learning or TensorFlow.
        Basic data visualization using matplotlib.
        Good problem solving and communication skills.
    """,

    "Candidate_03_Sneha": """
        Machine Learning Engineer with 4 years of experience.
        Expert in Python, TensorFlow, PyTorch, and scikit-learn.
        Deep learning specialist with CNN and RNN experience.
        NLP expert with BERT and transformer models experience.
        Proficient in pandas, numpy, matplotlib, and seaborn.
        AWS and Azure cloud experience. Strong SQL knowledge.
        Git version control and agile methodology experience.
        Data visualization and statistical analysis expert.
    """,

    "Candidate_04_Amit": """
        Fresh graduate with knowledge of Python programming.
        Completed online courses in machine learning basics.
        Basic pandas and numpy skills from college projects.
        No professional experience in data science.
        Familiar with matplotlib for simple plots.
        Basic SQL knowledge from database course.
        Eager to learn and grow in data science field.
    """,

    "Candidate_05_Divya": """
        Data Analyst with 2 years experience in business intelligence.
        Strong SQL and Excel skills for data analysis.
        Python scripting for data automation tasks.
        Experience with data visualization tools and matplotlib.
        Basic machine learning knowledge using scikit-learn.
        No deep learning or NLP experience.
        Good analytical skills and business understanding.
        Familiar with agile methodology and Git basics.
    """,

    "Candidate_06_Kiran": """
        AI Research Engineer with 5 years of experience.
        Expert Python developer with pandas, numpy, scikit-learn.
        Deep learning with TensorFlow, PyTorch, and Keras.
        NLP specialist with transformers and language models.
        Data visualization with matplotlib and seaborn.
        AWS certified with Azure experience.
        Strong statistics and mathematics background.
        Git, agile, and team leadership experience.
        Published research papers in machine learning.
    """,

    "Candidate_07_Meena": """
        Business Analyst with 3 years experience.
        Strong Excel and PowerPoint presentation skills.
        Basic Python knowledge from online courses.
        No machine learning or data science experience.
        SQL for business reporting and dashboards.
        Good communication and stakeholder management.
        Familiar with project management tools.
    """,

    "Candidate_08_Sai": """
        Data Science intern with 6 months experience.
        Python programming with pandas and numpy skills.
        Machine learning projects using scikit-learn.
        Basic deep learning with TensorFlow tutorials.
        Data visualization with matplotlib.
        SQL database queries and Git version control.
        Statistics and regression analysis knowledge.
        Eager to work on NLP and cloud projects.
    """
}

print(f"Total Candidates: {len(resumes)}")
print(f"Required Skills: {len(required_skills)}")

# ============================================
# STEP 3: TEXT PREPROCESSING
# ============================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

cleaned_resumes = {name: clean_text(resume) for name, resume in resumes.items()}
cleaned_jd = clean_text(job_description)

print("\n" + "=" * 60)
print("TEXT PREPROCESSING COMPLETE")
print("=" * 60)

# ============================================
# STEP 4: SKILL EXTRACTION
# ============================================

def extract_skills(text, skills_list):
    text_lower = text.lower()
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text_lower:
            found_skills.append(skill)
    return found_skills

print("\n" + "=" * 60)
print("SKILL EXTRACTION RESULTS:")
print("=" * 60)

candidate_skills = {}
for name, resume in resumes.items():
    skills = extract_skills(resume, required_skills)
    candidate_skills[name] = skills
    print(f"\n{name}:")
    print(f"  Skills Found ({len(skills)}): {', '.join(skills)}")

# ============================================
# STEP 5: SIMILARITY SCORING (TF-IDF)
# ============================================

all_texts = [cleaned_jd] + list(cleaned_resumes.values())
candidate_names = list(resumes.keys())

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(all_texts)

jd_vector = tfidf_matrix[0]
resume_vectors = tfidf_matrix[1:]

similarity_scores = cosine_similarity(jd_vector, resume_vectors)[0]

# ============================================
# STEP 6: SKILL MATCH SCORING
# ============================================

skill_scores = []
missing_skills_dict = {}

for name in candidate_names:
    found = candidate_skills[name]
    score = len(found) / len(required_skills) * 100
    skill_scores.append(score)
    missing = [s for s in required_skills if s not in found]
    missing_skills_dict[name] = missing

# ============================================
# STEP 7: FINAL RANKING
# ============================================

# Combined score: 60% similarity + 40% skill match
final_scores = (similarity_scores * 0.6) + (np.array(skill_scores) / 100 * 0.4)
final_scores_percent = final_scores * 100

results_df = pd.DataFrame({
    'Candidate': candidate_names,
    'Similarity_Score': (similarity_scores * 100).round(2),
    'Skill_Match_%': np.array(skill_scores).round(2),
    'Final_Score': final_scores_percent.round(2),
    'Skills_Found': [len(candidate_skills[n]) for n in candidate_names],
    'Skills_Missing': [len(missing_skills_dict[n]) for n in candidate_names]
})

results_df = results_df.sort_values('Final_Score', ascending=False)
results_df['Rank'] = range(1, len(results_df) + 1)

print("\n" + "=" * 60)
print("CANDIDATE RANKING RESULTS:")
print("=" * 60)
print(f"\n{'Rank':<5} {'Candidate':<25} {'Final Score':<12} {'Skill Match':<12} {'Skills Found'}")
print("-" * 70)
for _, row in results_df.iterrows():
    status = "SHORTLISTED" if row["Final_Score"] >= 30 else "REJECTED"
    print(f"{int(row['Rank']):<5} {row['Candidate']:<25} {row['Final_Score']:<12.1f} {row['Skill_Match_%']:<12.1f} {int(row['Skills_Found'])}/19  {status}")

# ============================================
# STEP 8: MISSING SKILLS ANALYSIS
# ============================================

print("\n" + "=" * 60)
print("MISSING SKILLS ANALYSIS (Top 3 Candidates):")
print("=" * 60)

top3 = results_df.head(3)
for _, row in top3.iterrows():
    name = row['Candidate']
    missing = missing_skills_dict[name]
    print(f"\n{name} (Rank {int(row['Rank'])}):")
    if missing:
        print(f"  Missing Skills: {', '.join(missing[:5])}")
    else:
        print(f"  No missing skills!")

# ============================================
# STEP 9: VISUALIZATION DASHBOARD
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle('Resume Screening & Ranking Dashboard\nJob Role: Data Scientist',
             fontsize=16, fontweight='bold')
fig.patch.set_facecolor('#F8F9FA')

short_names = [n.replace('Candidate_0', 'C').replace('Candidate_', 'C') for n in results_df['Candidate']]

# Chart 1: Final Score Ranking
ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')
colors1 = ['#4CAF50' if s >= 30 else '#F44336' for s in results_df['Final_Score']]
bars1 = ax1.barh(short_names[::-1], results_df['Final_Score'][::-1],
                 color=colors1[::-1], edgecolor='white', linewidth=1.5)
ax1.set_title('Candidate Final Score Ranking', fontweight='bold', fontsize=12)
ax1.set_xlabel('Final Score (%)')
ax1.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Shortlist Threshold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars1, results_df['Final_Score'][::-1]):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Chart 2: Skill Match %
ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')
colors2 = ['#2196F3' if s >= 30 else '#FF9800' for s in results_df['Skill_Match_%']]
bars2 = ax2.bar(short_names, results_df['Skill_Match_%'],
                color=colors2, edgecolor='white', linewidth=1.5)
ax2.set_title('Skill Match Percentage', fontweight='bold', fontsize=12)
ax2.set_xlabel('Candidate')
ax2.set_ylabel('Skill Match (%)')
ax2.set_ylim(0, 110)
ax2.grid(True, alpha=0.3, axis='y')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right')
for bar, val in zip(bars2, results_df['Skill_Match_%']):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 3: Skills Found vs Missing
ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')
x = np.arange(len(short_names))
width = 0.35
bars3a = ax3.bar(x - width/2, results_df['Skills_Found'], width,
                 label='Skills Found', color='#4CAF50', edgecolor='white')
bars3b = ax3.bar(x + width/2, results_df['Skills_Missing'], width,
                 label='Skills Missing', color='#F44336', edgecolor='white')
ax3.set_title('Skills Found vs Missing', fontweight='bold', fontsize=12)
ax3.set_xlabel('Candidate')
ax3.set_ylabel('Number of Skills')
ax3.set_xticks(x)
ax3.set_xticklabels(short_names, rotation=30, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: Shortlisted vs Rejected
ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')
shortlisted = len(results_df[results_df['Final_Score'] >= 30])
rejected = len(results_df[results_df['Final_Score'] < 30])
ax4.pie([shortlisted, rejected],
        labels=[f'Shortlisted\n({shortlisted})', f'Rejected\n({rejected})'],
        colors=['#4CAF50', '#F44336'],
        autopct='%1.0f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops={'fontsize': 12})
ax4.set_title('Shortlisted vs Rejected', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('resume_screening_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.show()

print("\n" + "=" * 60)
print("Dashboard saved: resume_screening_dashboard.png")
print("=" * 60)
print("\nBUSINESS INSIGHTS:")
print(f"  Total Candidates Screened : {len(resumes)}")
print(f"  Shortlisted (Score >= 50%) : {shortlisted}")
print(f"  Rejected                   : {rejected}")
print(f"  Top Candidate              : {results_df.iloc[0]['Candidate']}")
print(f"  Top Score                  : {results_df.iloc[0]['Final_Score']:.1f}%")
print(f"  Required Skills Total      : {len(required_skills)}")
print("\nProject Complete!")
