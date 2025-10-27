from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import os
import docx2txt
import PyPDF2
from pdfminer.high_level import extract_text
import docx
import re
from datetime import timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# --- NLTK Setup ---
try:
    stopwords.words('english')
    nltk.data.find('tokenizers/punkt/english.pickle') 
except LookupError:
    try:
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        pass

# Define standard stop words and punctuation for cleaning
STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = string.punctuation

# Allowed resume file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx'}

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Rushikesh123@localhost/analyzer'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.permanent_session_lifetime = timedelta(days=7)

db = SQLAlchemy(app)

# User database model
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(30), unique=True, nullable=False)
    phone = db.Column(db.String(11), nullable=False)
    password = db.Column(db.String(128), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception:
        return ""

def parse_docx(file_path):
    try:
        doc = docx.Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
    except Exception:
        return ""

def parse_resume(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return parse_pdf(file_path)
    elif ext == 'docx':
        return parse_docx(file_path)
    else:
        return ""

def preprocess_text(text):
    """Tokenizes and cleans text by removing stopwords and punctuation."""
    if not text:
        return []
    # Replace non-word characters (except spaces and hyphens) with a space
    text = re.sub(r'[^\w\s-]', ' ', text.lower())
    tokens = word_tokenize(text)
    # Filter out stopwords, and single-character words
    return [
        word for word in tokens 
        if word not in STOP_WORDS and len(word) > 1
    ]

def score_resume_detailed(resume_text, job_description_text):
    text = resume_text.lower()
    results = []
    
    # --- 1. Job Match Score (Highly Weighted Part, Max 60 points) ---
    
    jd_tokens = preprocess_text(job_description_text)
    resume_tokens = preprocess_text(resume_text)
    
    # Use TF-IDF and Cosine Similarity for the raw match score (0 to 100)
    texts = [text, job_description_text.lower()]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    raw_match_score_percent = int(cosine_sim * 100)
    
    # Determine critical JD keywords (top 15 most frequent non-stop words)
    jd_counts = Counter(jd_tokens)
    critical_jd_keywords = [item[0] for item in jd_counts.most_common(15)]
    
    matched_critical_keywords = sum(1 for keyword in critical_jd_keywords if keyword in resume_tokens)
    
    # New JD Alignment Score (Max 60 points)
    max_jd_score = 60
    jd_score = 0
    jd_alignment_flaws = []
    jd_alignment_fixes = []
    
    if critical_jd_keywords:
        jd_score = min(max_jd_score, int((matched_critical_keywords / len(critical_jd_keywords)) * max_jd_score))
        
        # --- Specific Feedback Generation for Job Alignment ---
        if matched_critical_keywords < len(critical_jd_keywords):
            missing_keywords = [k for k in critical_jd_keywords if k not in resume_tokens][:5]
            if missing_keywords:
                jd_alignment_flaws.append(f"Missing critical keywords found in the job description: {', '.join(missing_keywords)}.")
                jd_alignment_fixes.append(f"Integrate these missing terms: **{', '.join([k.upper() for k in missing_keywords])}** into your experience bullet points or summary.")
        
        if jd_score < max_jd_score * 0.7:
             jd_alignment_flaws.append(f"Your Job Alignment score is low. Focus on improving keyword density.")
             
    results.append({
        'category': 'Job Alignment',
        'status': jd_score >= (max_jd_score * 0.7), # Pass if 70% or higher
        'score': jd_score,
        'max_score': max_jd_score,
        'flaws': jd_alignment_flaws,
        'fix_tips': jd_alignment_fixes,
        'suggestions': ["Focus on quantifying your achievements using key terms from the JD."]
    })

    # --- 2. General Best Practice Scores (Max 40 points) ---
    
    # Contact Information (Max 5)
    email_pattern = r'\b[\w.-]+@[\w.-]+\.\w{2,4}\b'
    phone_pattern = r'\b(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
    contact_passed = re.search(email_pattern, text) is not None and re.search(phone_pattern, text) is not None
    contact_score = 5 if contact_passed else 0
    contact_flaws, contact_fixes = [], []
    if not contact_passed:
        contact_flaws.append("Missing essential contact details (email and/or phone number).")
        contact_fixes.append("Ensure your **Email** and **Phone** are prominently placed at the top of the resume.")
    results.append({
        'category': 'Contact Info',
        'status': contact_passed, # Pass/Fail status
        'score': contact_score,
        'max_score': 5,
        'flaws': contact_flaws,
        'fix_tips': contact_fixes,
        'suggestions': ["Ensure contact details are in a readable format for Applicant Tracking Systems (ATS)."] if not contact_passed else []
    })

    # Professional Summary (Max 5)
    summary_passed = 'summary' in text or 'objective' in text or 'profile' in text
    summary_score = 5 if summary_passed else 0
    summary_flaws, summary_fixes = [], []
    if not summary_passed:
        summary_flaws.append("No dedicated professional summary or objective was found.")
        summary_fixes.append("Write a brief, 3-4 sentence professional summary that immediately highlights your relevant skills and experience.")
    results.append({
        'category': 'Summary',
        'status': summary_passed, # Pass/Fail status
        'score': summary_score,
        'max_score': 5,
        'flaws': summary_flaws,
        'fix_tips': summary_fixes,
        'suggestions': ["Tailor your summary to fit the job description's core requirements."] if not summary_passed else []
    })

    # Work Experience (Max 10)
    exp_passed = any(word in text for word in ['experience', 'worked', 'managed', 'developed', 'achieved', 'led', 'optimized', 'implemented'])
    experience_score = 10 if exp_passed else 0
    exp_flaws, exp_fixes = [], []
    if not exp_passed:
        exp_flaws.append("Work experience section seems missing or weak. We recommend at least two previous roles or significant projects.")
        exp_fixes.append("Add detailed job titles, company names, dates, and 3-5 achievement-focused bullet points for each role.")
    results.append({
        'category': 'Work Experience',
        'status': exp_passed, # Pass/Fail status
        'score': experience_score,
        'max_score': 10,
        'flaws': exp_flaws,
        'fix_tips': exp_fixes,
        'suggestions': ["Use strong action verbs (e.g., led, optimized) and quantify results with numbers/metrics wherever possible."] if not exp_passed else []
    })

    # Education (Max 10)
    edu_passed = any(word in text for word in ['bachelor', 'master', 'university', 'college', 'degree', 'mba', 'phd'])
    education_score = 10 if edu_passed else 0
    edu_flaws, edu_fixes = [], []
    if not edu_passed:
        edu_flaws.append("Education details are missing or incomplete. Include your highest level of study.")
        edu_fixes.append("Include your degree/major, university name, and graduation year (or expected date).")
    results.append({
        'category': 'Education',
        'status': edu_passed, # Pass/Fail status
        'score': education_score,
        'max_score': 10,
        'flaws': edu_flaws,
        'fix_tips': edu_fixes,
        'suggestions': ["List education in reverse chronological order."] if not edu_passed else []
    })

    # Certifications (Max 5)
    cert_passed = 'certification' in text or 'certified' in text or 'license' in text
    cert_score = 5 if cert_passed else 0
    cert_flaws, cert_fixes = [], []
    if not cert_passed:
        cert_flaws.append("No certifications or professional trainings were mentioned.")
        cert_fixes.append("If applicable, add relevant certifications (e.g., PMP, AWS, Google Certifications) to boost credibility.")
    results.append({
        'category': 'Certifications',
        'status': cert_passed, # Pass/Fail status
        'score': cert_score,
        'max_score': 5,
        'flaws': cert_flaws,
        'fix_tips': cert_fixes,
        'suggestions': ["List all professional courses and certifications relevant to the target job."] if not cert_passed else []
    })

    # Formatting & Length (Max 5)
    length_passed = len(text) > 500
    length_score = 5 if length_passed else 0
    length_flaws, length_fixes = [], []
    if not length_passed:
        length_flaws.append("The resume appears too short and lacks sufficient detail (less than ~400-500 words).")
        length_fixes.append("Expand sections with relevant projects, detailed experiences, and skills to provide more content for ATS analysis.")
    results.append({
        'category': 'Formatting & Length',
        'status': length_passed, # Pass/Fail status
        'score': length_score,
        'max_score': 5,
        'flaws': length_flaws,
        'fix_tips': length_fixes,
        'suggestions': ["Ensure the resume is easy to read with clear headings and consistent font sizes."] if not length_passed else []
    })
    
    # Final Scoring Calculation (summing up the numerical scores to keep the total score circle functional)
    total_score = sum(item['score'] for item in results)
    total_score = min(total_score, 100)
    
    return total_score, results, raw_match_score_percent

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file_path):
    return docx2txt.process(file_path)

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    else:
        return ""

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/product')
def product():
    return render_template('product.html')

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        flash('Please log in to view your profile.', 'danger')
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('signup'))
        if User.query.filter_by(phone=phone).first():
            flash('Phone number already exists.', 'danger')
            return redirect(url_for('signup'))
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('signup'))

        new_user = User(
            username=username,
            email=email,
            phone=phone,
            password=generate_password_hash(password, method='pbkdf2:sha256')
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Signup successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session.permanent = True
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in.', 'danger')
        return redirect(url_for('login'))

    user = User.query.get(session['user_id'])
    return render_template('dashboard.html', user=user)

@app.route('/analyze-and-match', methods=['GET', 'POST'])
def analyze_and_match():
    if 'user_id' not in session:
        flash('Please log in to use the Resume Analyzer.', 'danger')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            job_description = request.form.get('job_description', '')
            
            if 'resume' not in request.files:
                flash("No resume file uploaded.", 'danger')
                return redirect(url_for('analyze_and_match'))

            file = request.files['resume']

            if file.filename == '':
                flash("No selected file.", 'danger')
                return redirect(url_for('analyze_and_match'))

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                
                unique_filename = f"{session['user_id']}_{filename}"
                unique_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                
                file.stream.seek(0) 
                file.save(unique_filepath) 
                
                resume_text = parse_resume(unique_filepath)
                
                os.remove(unique_filepath) 

                if not resume_text.strip():
                    flash("Failed to parse resume. Please upload a valid PDF or DOCX file.", 'danger')
                    return redirect(url_for('analyze_and_match'))
                
                total_score, detailed_results, match_score = score_resume_detailed(resume_text, job_description)
                
                return render_template('combined_result.html', 
                                       total_score=total_score, 
                                       match_score=match_score, 
                                       analysis_results=detailed_results)

            else:
                flash("Allowed file types are PDF and DOCX only.", 'danger')
                return redirect(url_for('analyze_and_match'))
        except Exception as e:
            flash(f"An unexpected error occurred: {e}", 'danger')
            return redirect(url_for('analyze_and_match'))

    return render_template('analyze_and_match_form.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out.', 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    app.run(debug=True)