from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
import nltk
import spacy
import pandas as pd
import base64
import random
import time
import datetime
import os
import io
import json
import cv2
import numpy as np
import pymysql
import asyncio
import edge_tts
import pdfplumber
import docx
import google.generativeai as genai
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.models import Sequential, model_from_json
from threading import Thread
from collections import Counter
from custom_resume_parser import CustomResumeParser
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
from werkzeug.utils import secure_filename
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file, send_from_directory
import os
import asyncio
import edge_tts
import uuid
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = 'Uploaded_Resumes'
AUDIO_FOLDER = 'static/temp_audio'  # Changé pour être dans static
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Ensure upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Download NLTK data
try:
    nltk.download('stopwords', quiet=True)
    spacy.load('en_core_web_sm')
except:
    pass

# Configure Gemini API
os.environ["GOOGLE_API_KEY"] = "Your API KEY Here"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-1.5-pro")

# Database connection
connection = pymysql.connect(host='localhost', user='', password='')
cursor = connection.cursor()

# Create database and table
def init_database():
    db_sql = """CREATE DATABASE IF NOT EXISTS SRA;"""
    cursor.execute(db_sql)
    connection.select_db("sra")
    
    DB_table_name = 'user_data'
    table_sql = f"""CREATE TABLE IF NOT EXISTS {DB_table_name} (
                    ID INT NOT NULL AUTO_INCREMENT,
                    Name varchar(100) NOT NULL,
                    Email_ID VARCHAR(50) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field VARCHAR(25) NOT NULL,
                    User_level VARCHAR(30) NOT NULL,
                    Actual_skills VARCHAR(300) NOT NULL,
                    Recommended_skills VARCHAR(300) NOT NULL,
                    Recommended_courses VARCHAR(600) NOT NULL,
                    Dominant_Emotion VARCHAR(30),
                    PRIMARY KEY (ID));"""
    cursor.execute(table_sql)

# Initialize database on startup
init_database()

# Emotion Detection Class
@register_keras_serializable()
class MySequential(Sequential):
    pass

class EmotionDetector:
    def __init__(self, model_json_path, model_weights_path):
        self.is_running = False
        self.current_emotion = "neutral"
        self.model_json_path = model_json_path
        self.model_weights_path = model_weights_path
        self.model = None
        self.webcam = None
        self.thread = None
        self.labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    def load_model(self):
        try:
            # Vérifier si les fichiers existent
            if not os.path.exists(self.model_json_path):
                print(f"Fichier modèle JSON non trouvé: {self.model_json_path}")
                return False
                
            if not os.path.exists(self.model_weights_path):
                print(f"Fichier poids du modèle non trouvé: {self.model_weights_path}")
                return False
            
            with open(self.model_json_path, "r") as json_file:
                model_json = json_file.read()
            self.model = model_from_json(model_json, custom_objects={"Sequential": MySequential})
            self.model.load_weights(self.model_weights_path)
            print("Modèle d'émotion chargé avec succès")
            return True
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            return False

    def get_current_emotion(self):
        return self.current_emotion

# Route pour tester si le modèle est chargé
@app.route('/test_emotion_model')
def test_emotion_model():
    """Test si le modèle d'émotion est correctement chargé"""
    success = emotion_detector.load_model()
    model_files_exist = {
        'json': os.path.exists(emotion_detector.model_json_path),
        'weights': os.path.exists(emotion_detector.model_weights_path)
    }
    
    return jsonify({
        'model_loaded': success,
        'files_exist': model_files_exist,
        'current_emotion': emotion_detector.get_current_emotion()
    })

# Global emotion detector instance
emotion_detector = EmotionDetector("./models/facialemotionmodel.json", "./models/facialemotionmodel.h5")

# Utility functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        with pdfplumber.open(file_path) as pdf:
            text = ''.join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif ext == 'docx':
        doc = docx.Document(file_path)
        text = ' '.join([para.text for para in doc.paragraphs])
    elif ext == 'txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = ""
    return text

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def Question_mcqs_generator(input_text, num_questions):
    prompt = f"""
    Vous êtes un assistant recruteur intelligent. À partir du contenu du CV suivant :
    '{input_text}'
    Générez {num_questions} questions d'entretien ciblées (technique, projets, expériences).
    Format attendu:
    ## Question
    Q: [Votre question]
    """
    response = model.generate_content(prompt).text.strip()
    return response

# Fonction corrigée pour la génération de questions avec audio
def prepare_questions_with_audio(questions):
    structured = []
    for idx, q in enumerate(questions):
        q_text = q.strip().replace("Q:", "").strip()
        if not q_text:  # Skip empty questions
            continue
            
        # Génère un nom de fichier unique
        audio_filename = f"question_{uuid.uuid4().hex[:8]}_{idx}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)
        
        try:
            # Création de l'audio
            async def create_audio(text, path):
                communicate = edge_tts.Communicate(text=text, voice="fr-FR-DeniseNeural")
                await communicate.save(path)
            
            # Exécute la création audio de manière synchrone
            asyncio.run(create_audio(q_text, audio_path))
            
            # Vérifie si le fichier a été créé
            if os.path.exists(audio_path):
                structured.append({
                    "question": q_text,
                    "audio_filename": audio_filename,  # Juste le nom du fichier
                    "audio_url": url_for('serve_audio', filename=audio_filename)  # URL complète
                })
            else:
                print(f"Erreur: Fichier audio non créé pour la question {idx}")
                structured.append({
                    "question": q_text,
                    "audio_filename": None,
                    "audio_url": None
                })
        except Exception as e:
            print(f"Erreur lors de la création de l'audio pour la question {idx}: {e}")
            structured.append({
                "question": q_text,
                "audio_filename": None,
                "audio_url": None
            })
    
    return structured

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses, dominant_emotion="none"):
    DB_table_name = 'user_data'
    insert_sql = f"INSERT INTO {DB_table_name} VALUES (0,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    rec_values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses, dominant_emotion)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

def get_recommendations(skills):
    ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask', 'streamlit']
    web_keyword = ['react', 'django', 'node js', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask']
    android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
    ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
    uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes']

    recommended_skills = []
    reco_field = ''
    rec_course = []

    skills_lower = [skill.lower() for skill in skills]
    
    for skill in skills_lower:
        if skill in ds_keyword:
            reco_field = 'Data Science'
            recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling', 'Data Mining', 'Clustering & Classification']
            rec_course = random.sample(ds_course, min(4, len(ds_course)))
            break
        elif skill in web_keyword:
            reco_field = 'Web Development'
            recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento', 'wordpress']
            rec_course = random.sample(web_course, min(4, len(web_course)))
            break
        elif skill in android_keyword:
            reco_field = 'Android Development'
            recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java']
            rec_course = random.sample(android_course, min(4, len(android_course)))
            break
        elif skill in ios_keyword:
            reco_field = 'IOS Development'
            recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode']
            rec_course = random.sample(ios_course, min(4, len(ios_course)))
            break
        elif skill in uiux_keyword:
            reco_field = 'UI-UX Development'
            recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq']
            rec_course = random.sample(uiux_course, min(4, len(uiux_course)))
            break

    return reco_field, recommended_skills, rec_course

# Routes
@app.route('/')
def index():
    return render_template('index.html')

# Ajoutez cette route pour servir les fichiers audio
@app.route('/audio/<filename>')
def serve_audio(filename):
    """Sert les fichiers audio générés"""
    try:
        return send_from_directory(AUDIO_FOLDER, filename)
    except FileNotFoundError:
        return "Fichier audio non trouvé", 404
@app.route('/user')
def user_dashboard():
    return render_template('user_dashboard.html')

@app.route('/resume_analyzer')
def resume_analyzer():
    return render_template('resume_analyzer.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['resume']
    if file.filename == '':
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Parse resume
        parser = CustomResumeParser(file_path)
        resume_data = parser.get_extracted_data()
        
        # Vérification : est-ce un CV valide ?
        if not resume_data['name'] or not resume_data['email']:
            flash('Le fichier fourni ne semble pas être un CV valide. Veuillez réessayer avec un CV.')
            return redirect(url_for('resume_analyzer'))
        
        if resume_data:
            # Get recommendations
            reco_field, recommended_skills, rec_course = get_recommendations(resume_data['skills'])
            
            # Calculate resume score
            resume_text = pdf_reader(file_path)
            resume_score = 0
            sections = ['Compétences', 'Langues', 'Hobbies', 'Certifications', 'Expériences']
            for section in sections:
                if section in resume_text:
                    resume_score += 20
            
            # Determine candidate level
            cand_level = 'Fresher' if resume_data['no_of_pages'] == 1 else 'Intermediate' if resume_data['no_of_pages'] == 2 else 'Experienced'
            
            # Insert data
            ts = time.time()
            cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = str(cur_date + '_' + cur_time)
            
            insert_data(resume_data['name'], resume_data['email'], resume_score, timestamp,
                       resume_data['no_of_pages'], reco_field, cand_level,
                       str(resume_data['skills']), str(recommended_skills), str(rec_course))
            
            # Store results in session
            session['resume_results'] = {
                'resume_data': resume_data,
                'resume_score': resume_score,
                'reco_field': reco_field,
                'recommended_skills': recommended_skills,
                'rec_course': rec_course,
                'cand_level': cand_level
            }
            
            return redirect(url_for('resume_results'))
    
    flash('Type de fichier non autorisé')
    return redirect(url_for('resume_analyzer'))
@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    """Route pour analyser l'émotion à partir d'une image de la webcam"""
    try:
        # Récupérer l'image encodée en base64 depuis le frontend
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Aucune image fournie'}), 400
        
        # Décoder l'image base64
        image_data = image_data.split(',')[1]  # Enlever le préfixe data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        
        # Convertir en array numpy
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Image invalide'}), 400
        
        # Analyser l'émotion
        emotion = analyze_emotion_from_frame(img)
        
        # Mettre à jour l'émotion courante dans le détecteur
        emotion_detector.current_emotion = emotion
        
        return jsonify({'emotion': emotion})
        
    except Exception as e:
        print("Erreur lors de l'analyse d'emotion:", e)
        return jsonify({'error': "Erreur lors de l'analyse"}), 500


def analyze_emotion_from_frame(frame):
    """Analyse l'émotion à partir d'une frame de caméra"""
    try:
        # Charger le modèle si ce n'est pas fait
        if emotion_detector.model is None:
            if not emotion_detector.load_model():
                return "neutral"
        
        # Charger le classificateur de visage d'OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Détecter les visages
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return "neutral"
        
        # Prendre le premier visage détecté
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        
        # Redimensionner à 48x48 pixels (taille attendue par le modèle)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normaliser et préparer pour le modèle
        roi_gray = roi_gray.astype('float32') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        
        # Prédire l'émotion
        prediction = emotion_detector.model.predict(roi_gray, verbose=0)
        max_index = np.argmax(prediction[0])
        
        # Retourner l'émotion correspondante
        return emotion_detector.labels.get(max_index, "neutral")
        
    except Exception as e:
        print(f"Erreur dans analyze_emotion_from_frame: {e}")
        return "neutral"

# Route pour sauvegarder la configuration des questions
@app.route('/admin/save_question_config', methods=['POST'])
def save_question_config():
    if not session.get('admin_authenticated'):
        return jsonify({'error': 'Non autorisé'}), 401
    
    data = request.json
    num_questions = data.get('num_questions', 5)
    
    # Sauvegarder dans la base de données (vous pouvez créer une table config)
    # Ou utiliser un fichier de configuration
    # Pour l'exemple, on utilise la session
    session['admin_question_config'] = num_questions
    
    return jsonify({'success': True})

# Route pour récupérer la configuration des questions
@app.route('/admin/get_question_config')
def get_question_config():
    num_questions = session.get('admin_question_config', 10)
    return jsonify({'num_questions': num_questions})

# Route API pour les candidats
@app.route('/api/get_admin_config')
def get_admin_config():
    num_questions = session.get('admin_question_config', 10)
    return jsonify({'num_questions': num_questions})

@app.route('/resume_results')
def resume_results():
    # Vérifie si les résultats sont dans la session
    if 'resume_results' not in session:
        return redirect(url_for('resume_analyzer'))

    results = session['resume_results']
    reco_field = None
    rec_course = []

    # Extraction du domaine détecté dans les résultats
    if results:
        if isinstance(results, dict):
            reco_field = results.get('reco_field')
        else:
            reco_field = getattr(results, 'reco_field', None)

        # Mapping domaine → cours
        course_map = {
            'Data Science': ds_course,
            'Web Development': web_course,
            'Android Development': android_course,
            'iOS Development': ios_course,
            'UI/UX': uiux_course
        }

        rec_course = course_map.get(reco_field, [])

    return render_template(
        'resume_results.html',
        results=results,
        rec_course=rec_course,
        # Ajoute ici resume, resume_vid, interview_vid si nécessaire
    )

@app.route('/interview_questions')
def interview_questions():
    return render_template('interview_questions.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    if 'resume' not in request.files:
        flash('Aucun fichier sélectionné')
        return redirect(request.url)
    
    file = request.files['resume']
    num_questions = int(request.form.get('num_questions', 5))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        try:
            # Extract text and generate questions
            text = extract_text_from_file(file_path)
            questions_text = Question_mcqs_generator(text, num_questions)
            question_list = [q.strip() for q in questions_text.split("## Question")[1:] if q.strip()]
            
            # Essaie d'abord avec edge-tts
            try:
                structured_questions = prepare_questions_with_audio(question_list)
            except Exception as e:
                print(f"Erreur avec edge-tts, utilisation du fallback HTML5: {e}")
                structured_questions = prepare_questions_with_html5_speech(question_list)
            
            if not structured_questions:
                flash('Erreur lors de la génération des questions')
                return redirect(url_for('interview_questions'))
            
            # Store in session
            session['questions'] = structured_questions
            session['current_index'] = 0
            session['emotion_records'] = []
            session['uploaded_filename'] = filename
            
            # Load emotion detector model
            if emotion_detector.model is None:
                emotion_detector.load_model()
            
            return redirect(url_for('interview_session'))
            
        except Exception as e:
            print(f"Erreur lors du traitement: {e}")
            flash('Erreur lors du traitement du fichier')
            return redirect(url_for('interview_questions'))
    
    flash('Type de fichier non autorisé')
    return redirect(url_for('interview_questions'))

# Ajoutez une route de nettoyage pour supprimer les anciens fichiers audio
@app.route('/cleanup_audio')
def cleanup_old_audio():
    """Nettoie les anciens fichiers audio (à appeler périodiquement)"""
    try:
        import time
        current_time = time.time()
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                # Supprime les fichiers de plus d'une heure
                if current_time - os.path.getctime(file_path) > 3600:
                    os.remove(file_path)
        return jsonify({'status': 'success', 'message': 'Nettoyage effectué'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/interview_session')
def interview_session():
    if 'questions' not in session:
        return redirect(url_for('interview_questions'))
    
    current_index = session.get('current_index', 0)
    questions = session['questions']
    
    if current_index >= len(questions):
        return redirect(url_for('interview_complete'))
    
    current_question = questions[current_index]
    return render_template('interview_session.html', 
                         question=current_question, 
                         current_index=current_index,
                         total_questions=len(questions))

@app.route('/next_question', methods=['POST'])
def next_question():
    if 'questions' in session:
        current_index = session.get('current_index', 0)
        session['current_index'] = current_index + 1
        
        # Record emotion for current question
        emotion = emotion_detector.get_current_emotion()
        if 'emotion_records' not in session:
            session['emotion_records'] = []
        
        session['emotion_records'].append({
            'question_index': current_index,
            'question': session['questions'][current_index]['question'],
            'emotion': emotion,
            'timestamp': time.strftime("%H:%M:%S")
        })
        
        if session['current_index'] >= len(session['questions']):
            return redirect(url_for('interview_complete'))
    
    return redirect(url_for('interview_session'))
# Fonction utilitaire pour tester la génération audio
def test_audio_generation():
    """Fonction de test pour vérifier si edge-tts fonctionne"""
    try:
        test_text = "Ceci est un test de génération audio."
        test_path = os.path.join(AUDIO_FOLDER, "test_audio.mp3")
        
        async def create_test_audio():
            communicate = edge_tts.Communicate(text=test_text, voice="fr-FR-DeniseNeural")
            await communicate.save(test_path)
        
        asyncio.run(create_test_audio())
        
        if os.path.exists(test_path):
            os.remove(test_path)  # Nettoie le fichier test
            return True
        return False
    except Exception as e:
        print(f"Test audio échoué: {e}")
        return False
    
@app.route('/test_audio')
def test_audio():
    success = test_audio_generation()
    return jsonify({'audio_working': success})

@app.route('/interview_complete')
def interview_complete():
    if 'emotion_records' not in session:
        return redirect(url_for('interview_questions'))
    
    # Calculate dominant emotion
    emotions = [record['emotion'] for record in session['emotion_records']]
    dominant_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
    
    # Update database with dominant emotion if we have resume data
    if 'uploaded_filename' in session:
        file_path = os.path.join(UPLOAD_FOLDER, session['uploaded_filename'])
        parser = CustomResumeParser(file_path)
        resume_data = parser.get_extracted_data()
        
        if resume_data:
            reco_field, recommended_skills, rec_course = get_recommendations(resume_data['skills'])
            resume_text = pdf_reader(file_path)
            resume_score = sum(20 for section in ['Compétences', 'Langues', 'Hobbies', 'Certifications', 'Expériences'] if section in resume_text)
            cand_level = 'Fresher' if resume_data['no_of_pages'] == 1 else 'Intermediate' if resume_data['no_of_pages'] == 2 else 'Experienced'
            
            ts = time.time()
            cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            timestamp = str(cur_date + '_' + cur_time)
            
            insert_data(resume_data['name'], resume_data['email'], resume_score, timestamp,
                       resume_data['no_of_pages'], reco_field, cand_level,
                       str(resume_data['skills']), str(recommended_skills), str(rec_course),
                       dominant_emotion)
    
    results = {
        'dominant_emotion': dominant_emotion,
        'emotion_records': session['emotion_records']
    }
    
    # Clear session data
    for key in ['questions', 'current_index', 'emotion_records', 'uploaded_filename']:
        session.pop(key, None)
    
    return render_template('interview_complete.html', results=results)

@app.route('/get_current_emotion')
def get_current_emotion():
    emotion = emotion_detector.get_current_emotion()
    return jsonify({'emotion': emotion})

@app.route('/admin')
def admin_login():
    return render_template('admin_login.html')

@app.route('/admin_auth', methods=['POST'])
def admin_auth():
    username = request.form['username']
    password = request.form['password']
    
    # Simple authentication (in production, use proper hashing)
    if username == 'mjid' and password == 'mjid':
        session['admin_authenticated'] = True
        return redirect(url_for('admin_dashboard'))
    else:
        flash('Invalid credentials')
        return redirect(url_for('admin_login'))

@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    
    # Fetch data from database
    cursor.execute('SELECT * FROM user_data')
    data = cursor.fetchall()
    
    if not data:
        return render_template('admin_dashboard.html', data=[], stats={})
    
    # Create DataFrame for analysis
    df = pd.DataFrame(data, columns=['ID','Name','Email_ID','resume_score','Timestamp','Page_no','Predicted_Field',
                                      'User_level','Actual_skills','Recommended_skills','Recommended_courses','Dominant_Emotion'])

    
    # Calculate statistics
    stats = {
        'total_candidates': len(df),
        'avg_score': df['resume_score'].astype(float).mean(),
        'field_distribution': df['Predicted_Field'].value_counts().to_dict(),
        'level_distribution': df['User_level'].value_counts().to_dict(),
        'emotion_distribution': df[df['Dominant_Emotion'] != 'none']['Dominant_Emotion'].value_counts().to_dict()
    }
    
    return render_template("admin_dashboard.html", data=data, stats=stats, df_json=df.to_json(orient='records'))


@app.route('/admin_field_analysis')
def admin_field_analysis():
    if not session.get('admin_authenticated'):
        return redirect(url_for('admin_login'))
    
    field = request.args.get('field')
    if not field:
        return redirect(url_for('admin_dashboard'))
    
    cursor.execute('SELECT * FROM user_data WHERE Predicted_Field = %s', (field,))
    data = cursor.fetchall()
    
    if not data:
        flash(f'No data found for field: {field}')
        return redirect(url_for('admin_dashboard'))
    
    df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                   'Predicted_Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                   'Recommended Course', 'Dominant Emotion'])
    
    stats = {
        'total_candidates': len(df),
        'avg_score': df['Resume Score'].astype(float).mean(),
        'level_distribution': df['User Level'].value_counts().to_dict(),
        'emotion_distribution': df[df['Dominant Emotion'] != 'none']['Dominant Emotion'].value_counts().to_dict()
    }
    
    return render_template('field_analysis.html', field=field, data=data, stats=stats)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)