from flask import Flask, render_template, flash, redirect, url_for, session, request, jsonify, send_from_directory, send_file
from wtforms import Form, StringField, PasswordField, FileField, TextAreaField, DateField, TimeField, SelectField, validators
from passlib.hash import sha256_crypt
from functools import wraps
import os, json, torch, pymongo, pickle
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pymongo import MongoClient
from gridfs import GridFS
from flask_mysqldb import MySQL
from transformers import pipeline
import moviepy.editor as mp
from datetime import datetime
from bson import ObjectId
from huggingface_hub import InferenceClient
from fpdf import FPDF

app = Flask(__name__)

#for image
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

UPLOAD_FOLDER = 'uploads'
# Ensure that the directory exists, create it if it doesn't
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
# Set the Flask app's upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#Folder for classwork
CLASSWORK_UPLOAD_FOLDER = 'classwork_uploads'
# Ensure that the directory exists, create it if it doesn't
if not os.path.exists(CLASSWORK_UPLOAD_FOLDER):
    os.makedirs(CLASSWORK_UPLOAD_FOLDER)
# Set the Flask app's upload folder configuration
app.config['CLASSWORK_UPLOAD_FOLDER'] = CLASSWORK_UPLOAD_FOLDER

# Config MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'learning-platform'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'

# init MYSQL
mysql = MySQL(app)

# Config MongoDB
app.config['MONGO_URI'] = 'mongodb://localhost:27017/vecdb'
# Initialize MongoDB client
mongo = MongoClient(app.config['MONGO_URI'])

#Setup Database Connection
db = mongo.vecdb  
collection = db.files
active_chat_sessions_collection = db.active_chat_sessions
chat_sessions_collection = db.chat_sessions
session_history_collection=db.session_history
transcription_collection = db.transcriptions
bank_of_questions_collection = db.bank_of_questions
fs = GridFS(db)

# Check if user logged in
def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

# CHATBOT
## HF_TOKEN
HF_TOKEN = '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'
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Embedding Model
embeddings = HuggingFaceEmbeddings()

# LLM-Open Source
llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature= 0.5,max_new_tokens=512,
                         huggingfacehub_api_token=HF_TOKEN)

chatTemplate = """
    You are Bluebot and you are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
    Answer the question based on the chat history(delimited by <hs></hs>) and context(delimited by <ctx> </ctx>) below. Answer only from the context. If you don't know the answer, just say that you don't know not provided in the context.
    Don't provide information not directly related to the context.Only answer questions directly from the provided context. 
    Say hello when I say "hello" and bye when I say "bye".

    <ctx>
    {context}
    </ctx>
    <hs>
    {chat_history}
    </hs>
    Question: {question}
    Answer:
    """
promptHist = PromptTemplate(
      input_variables=["context", "question", "chat_history"],
      template=chatTemplate
  )

@app.route('/chat', methods=['GET', 'POST'])
@is_logged_in
def chat():
   return render_template('chatbot.html')

# Route to start a new chat session
@app.route('/start_new_chat_session', methods=['POST'])
@is_logged_in
def start_new_chat_session():
    # Get the latest session ID
    class_id = session['class_id']
    email = session['email']
    date = datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M:%S")
    latest_session = active_chat_sessions_collection.find_one(sort=[('session_id', pymongo.DESCENDING)])
    # If there are no sessions yet, start from 0
    if latest_session:
        session_id = latest_session['session_id'] + 1
    else:
        session_id = 0
    # Store the session ID in the active chat sessions collection
    active_chat_sessions_collection.insert_one({'date':date,'class_id':class_id,'email':email,'session_id': session_id})
    # Return the session ID to the client
    return jsonify({'success': True, 'sessionId': session_id})

@app.route('/get_chat_sessions', methods=['GET'])
def get_chat_sessions():
    # Retrieve existing chat sessions from the active chat sessions collection
    sessions = list(active_chat_sessions_collection.find({}, {'_id': 0, 'session_id': 1}))
    
    session_ids = [session['session_id'] for session in sessions]

    # Return the list of session IDs to the client
    return jsonify({'success': True, 'sessions': session_ids})

# Route for individual chat sessions
@app.route('/chat/<string:session_id>', methods=['GET', 'POST'])
@is_logged_in
def chat_session(session_id):
    email = session['email']
    class_id = session['class_id']

    if request.method == 'GET':
        # Retrieve chat history for the specified session ID
        session_history = get_session_history(class_id, session_id)
        return render_template('chat.html', session_history=session_history, session_id=session_id)

    elif request.method == 'POST':
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_input = request.json['user_input']

        # Saving user input to session history collection
        session_history_collection.insert_one({'date': date, 'class_id': class_id, 'session_id': session_id, 'email': email, 'sender': 'user', 'message': user_input})

        # Load PDF documents from the upload folder
        CLASSWORK_UPLOAD_FOLDER = 'classwork_uploads'
        classwork_path = os.path.join(CLASSWORK_UPLOAD_FOLDER, f'classwork_class_{class_id}')

        # Ensure the classwork directory exists
        if not os.path.exists(classwork_path):
            return jsonify({'response': 'Classwork directory not found', 'success': False, 'session_id': session_id})

        # Iterate over PDF files in the classwork directory
        docs = []
        for filename in os.listdir(classwork_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(classwork_path, filename)
                # Load PDF documents
                loader = PyPDFLoader(file_path)
                loader.requests_per_second = 1
                doc = loader.load()
                docs.extend(doc)  # Extend the list with the loaded documents instead of appending them individually

        # Ensure that docs is not empty
        if not docs:
            return jsonify({'response': 'No documents found', 'success': False, 'session_id': session_id})

        # Chunking - Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
        chunks = text_splitter.split_documents(docs)

        # Vectorization
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=f"db_classwork_{class_id}")
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 3})

        # Memory management
        memory_file = f'memory_{class_id}_{session_id}.pkl'
        if os.path.exists(memory_file):
            with open(memory_file, 'rb') as f:
                memory = pickle.load(f)
        else:
           memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer', input_key='question')

        # Add the current user input to the memory
        memory.chat_memory.add_user_message(user_input)
        memory.chat_memory.add_ai_message('')  # Initialize the AI response to an empty string

        rag_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever, memory=memory, combine_docs_chain_kwargs={'prompt': promptHist})

        # Bot response generation
        response = rag_chain(inputs={'question': user_input, 'chat_history': memory.chat_memory.messages})
        bot_response = response['answer'].replace('\n', '<br>')  # Replace newlines with <br>

        # Update the AI response in the memory
        memory.chat_memory.messages[-1].content = bot_response

        # Save memory state
        with open(memory_file, 'wb') as f:
            pickle.dump(memory, f)

        # Saving bot response to session history collection
        session_history_collection.insert_one({'date': date, 'class_id': class_id, 'session_id': session_id, 'email': email, 'sender': 'bot', 'message': bot_response})

        # Update session history
        update_session_history(class_id, email, session_id, user_input, bot_response)

        return jsonify({'response': bot_response, 'success': True, 'session_id': session_id})

# Function to retrieve session history
def get_session_history(class_id,session_id):
    return list(session_history_collection.find({"class_id": class_id,"session_id": session_id}))

# Function to update session history
def update_session_history(class_id,email,session_id, user_input, bot_response):
    date = datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M:%S")
    # Retrieve session history for the specified session ID from the session_history collection
    session_history = session_history_collection.find_one({'session_id': session_id})
    session_history = session_history.get('session_history', []) if session_history else []
    session_history.append({'user_input': user_input, 'bot_response': bot_response})
    # Update or insert session history in the session_history collection
    session_history_collection.update_one({'session_id': session_id}, {'$set': {'class_id':class_id, 'email':email,'session_history': session_history}}, upsert=True)

    # Update or insert session history in the session_history collection
    chat_sessions_collection.update_one({'session_id': session_id}, {'$set': {'class_id':class_id, 'email':email,'session_history': session_history,'date':date}}, upsert=True)

@app.route('/delete_chat_session/<session_id>', methods=['DELETE'])
def delete_chat_session(session_id):
    try:
        # Delete the chat session with the given session_id
        active_chat_sessions_collection.delete_one({'session_id': int(session_id)})
        chat_sessions_collection.delete_one({'session_id': int(session_id)})
        session_history_collection.delete_many({'session_id': session_id})
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#Generate Quiz 
client = InferenceClient("mistralai/Mixtral-8x7B-Instruct-v0.1")

def system_instructions(question_difficulty, tone, topic):
   return f"""<s> [INST] Your are a great teacher and your task is to create 10 questions
     with 4 choices with a {question_difficulty} difficulty in a {tone} tone about {topic}, 
     then create an answers. Index in text format, the questions as "Q#":"" to "Q#":"", 
     the four choices as "Q#:C1":"" to "Q#:C4":"", and the answers as "A#":"Q#:C#" to "A#":"Q#:C#". [/INST]"""

@app.route('/generate/<class_id>')
def generate(class_id):
    return render_template('generate.html', class_id=class_id)

@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    email = session['email']
    class_id = session['class_id']
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = request.json
    question_difficulty = data['question_difficulty']
    tone = data['tone']
    user_prompt = data['user_prompt']

    formatted_prompt = system_instructions(question_difficulty, tone, user_prompt)

    generate_kwargs = dict(
        temperature=0.1,
        max_new_tokens=2048,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=True,
        seed=42,
    )

    response = client.text_generation(
        formatted_prompt, **generate_kwargs, stream=False, details=False, return_full_text=False,
    )

    quiz_text = response

    bank_of_questions_collection.insert_one({'date': date, 'class_id': class_id, 'email': email, 'output': quiz_text })

    flash('Quiz generated successfully!', 'success')  # Add a flash message

    return jsonify(quiz_text)

@app.route('/bank_of_questions/<class_id>', methods=['GET'])
def bank_of_questions(class_id):
    questions = bank_of_questions_collection.find({'class_id': class_id})

    formatted_questions = []
    for question in questions:
        instructor_email = question['email']
        first_name, last_name = extract_name_from_email(instructor_email)
        instructor_name = f"{first_name} {last_name}"
        
        output = question['output']
        if not isinstance(output, str):
            output = json.dumps(output)
        
        output_lines = output.replace('\n', '<br>').split('<br>')
        
        formatted_question = {
            'id': str(question['_id']),
            'output_lines': output_lines,
            'instructor_name': instructor_name,
            'date': question['date'],
        }
        formatted_questions.append(formatted_question)
    return render_template('bank_of_questions.html', questions=formatted_questions, class_id=class_id)

@app.route('/edit_quiz_question/<question_id>', methods=['POST'])
def edit_quiz_question(question_id):
    data = request.json
    updated_output = data['output']

    bank_of_questions_collection.update_one({'_id': ObjectId(question_id)}, {'$set': {'output': updated_output}})
    return jsonify({'status': 'success'})

@app.route('/delete_quiz_question/<question_id>', methods=['POST'])
def delete_quiz_question(question_id):
    bank_of_questions_collection.delete_one({'_id': ObjectId(question_id)})
    return jsonify({'status': 'success'})

def extract_name_from_email(email):
    # Split email address based on '@' to get the local part
    local_part = email.split('@')[0]
    
    # Split the local part based on '.' to get first name and last name
    parts = local_part.split('.')
    
    if len(parts) == 2:  # Assuming first name and last name are separated by a dot
        first_name, last_name = parts
    else:
        # Handle cases where the email format doesn't match expectations
        first_name = parts[0]
        last_name = ""  # Assuming no last name is provided
    
    return first_name.capitalize(), last_name.capitalize()  # Capitalizing the names for consistency

######################################################################################################
@app.route('/')
def index():
    return render_template('home.html')

#About
@app.route('/about')
def about():
    return render_template('about.html')

#policy
@app.route('/privacypolicy')
def privacypolicy():
    return render_template('privacypolicy.html')
#terms
@app.route('/terms')
def terms():
    return render_template('terms.html')

#faq
@app.route('/faq')
def faq():
    return render_template('faq.html')

#cookies
@app.route('/cookies')
def cookies():
    return render_template('cookies.html')

#contact
@app.route('/contact')
def contact():
    return render_template('contact.html')

#Settings
@app.route('/settings')
def settings():
    return render_template('settings.html')

class EditNameForm(Form):
    first_name = StringField('First Name', [validators.Length(min=1, max=50)])
    last_name = StringField('Last Name', [validators.Length(min=1, max=50)])

@app.route('/edit_name', methods=['GET', 'POST'])
def edit_name():
    form = EditNameForm(request.form)
    if request.method == 'POST' and form.validate():
        first_name = form.first_name.data
        last_name = form.last_name.data
        email = session['email']  # Assuming you have the user's email stored in session

        cursor = mysql.connection.cursor()

        # Check if the user is a student
        cursor.execute("SELECT email FROM student WHERE email = %s", [email])
        student = cursor.fetchone()

        if student:
            cursor.execute("UPDATE student SET first_name = %s, last_name = %s WHERE email = %s",
                           (first_name, last_name, email))
            role = 'student'
        else:
            # Check if the user is an instructor
            cursor.execute("SELECT email FROM instructor WHERE email = %s", [email])
            instructor = cursor.fetchone()

            if instructor:
                cursor.execute("UPDATE instructor SET first_name = %s, last_name = %s WHERE email = %s",
                               (first_name, last_name, email))
                role = 'instructor'
            else:
                flash('User not found. Please contact support.', 'danger')
                return redirect(url_for('edit_name'))

        mysql.connection.commit()
        cursor.close()
        flash(f'Your name has been updated successfully as {role}', 'success')
        return redirect(url_for('dashboard'))  # Adjust to your application's flow

    return render_template('edit_name.html', form=form)

class EditEmailForm(Form):
    email = StringField('Email', [validators.Email(), validators.DataRequired()])
    password = StringField('Password', [validators.DataRequired()])  # To verify the user's identity

@app.route('/edit_email', methods=['GET', 'POST'])
def edit_email():
    form = EditEmailForm(request.form)
    
    if request.method == 'POST' and form.validate():
        new_email = form.new_email.data
        current_email = session.get('email')  # Assuming you have the user's current email stored in session

        try:
            cursor = mysql.connection.cursor()

            # Check if the new email is already in use
            cursor.execute("SELECT email FROM student WHERE email = %s", [new_email])
            existing_student = cursor.fetchone()

            cursor.execute("SELECT email FROM instructor WHERE email = %s", [new_email])
            existing_instructor = cursor.fetchone()

            if existing_student or existing_instructor:
                flash('Email already in use. Please choose a different email.', 'danger')
                return redirect(url_for('edit_email'))

            # Update email in the appropriate table based on the user's role
            cursor.execute("SELECT email FROM student WHERE email = %s", [current_email])
            student = cursor.fetchone()

            if student:
                cursor.execute("UPDATE student SET email = %s WHERE email = %s",
                               (new_email, current_email))
                role = 'student'
            else:
                cursor.execute("SELECT email FROM instructor WHERE email = %s", [current_email])
                instructor = cursor.fetchone()

                if instructor:
                    cursor.execute("UPDATE instructor SET email = %s WHERE email = %s",
                                   (new_email, current_email))
                    role = 'instructor'
                else:
                    flash('User not found. Please contact support.', 'danger')
                    return redirect(url_for('edit_email'))

            mysql.connection.commit()
            cursor.close()

            # Update session with the new email if needed
            session['email'] = new_email

            flash(f'Your email has been updated successfully as {new_email}', 'success')
            return redirect(url_for('dashboard'))  # Adjust to your application's flow

        except Exception as e:
            flash(f'An error occurred while updating your email: {str(e)}', 'danger')
            return redirect(url_for('edit_email'))

    return render_template('edit_email.html', form=form)


# Define TopicForm
class TopicForm(Form):
    name = StringField('Name', validators=[validators.DataRequired()])

class RegisterForm(Form):
    role = StringField('Role (student/instructor)', validators=[validators.DataRequired()])
    first_name = StringField('First Name', validators=[validators.DataRequired()])
    last_name = StringField('Last Name', validators=[validators.DataRequired()])
    email = StringField('Email', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.Length(min=6, max=25, message="Password must be between 6 and 25 characters")
    ])
    security_question = SelectField('Security Question', choices=[
        ('What is your mother\'s maiden name?', 'What is your mother\'s maiden name?'),
        ('What is the name of your first pet?', 'What is the name of your first pet?'),
        ('What city were you born in?', 'What city were you born in?'),
        ('What is your favorite book?', 'What is your favorite book?'),
        ('What was the name of your first school?', 'What was the name of your first school?'),
        ('What is the name of your favorite childhood friend?', 'What is the name of your favorite childhood friend?'),
        ('In what city or town did your mother and father meet?', 'In what city or town did your mother and father meet?')
    ], validators=[validators.DataRequired()])
    security_answer = PasswordField('Security Answer', validators=[validators.DataRequired()])
    
    def validate_email(self, field):
        cursor = mysql.connection.cursor()
        if self.role.data == 'student':
            cursor.execute("SELECT * FROM student WHERE email=%s", (field.data,))
        elif self.role.data == 'instructor':
            cursor.execute("SELECT * FROM instructor WHERE email=%s", (field.data,))
        user = cursor.fetchone()
        cursor.close()
        if user:
            raise validators.ValidationError('Email Already Taken')

# Login Form Class
class LoginForm(Form):
    email = StringField('Email', [validators.DataRequired(), validators.Email()])
    password = PasswordField('Password', [
        validators.DataRequired(),
        validators.Length(min=6, max=25)
    ])

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm(request.form)
    if request.method == 'POST' and form.validate():
        role = form.role.data.lower()  # Normalize role to lowercase
        first_name = form.first_name.data
        last_name = form.last_name.data
        email = form.email.data
        password = sha256_crypt.encrypt(str(form.password.data))
        security_question = form.security_question.data
        security_answer = sha256_crypt.encrypt(str(form.security_answer.data))  # Encrypt security answer
        cursor = mysql.connection.cursor()
        
        if role == 'student':
            cursor.execute(
                "INSERT INTO student (first_name, last_name, email, password, security_question, security_answer) \
                 VALUES (%s, %s, %s, %s, %s, %s)", (first_name, last_name, email, password, security_question, security_answer)
            )
            flash('You are now registered as a student and can login', 'success')
        elif role == 'instructor':
            cursor.execute(
                "INSERT INTO instructor (first_name, last_name, email, password, security_question, security_answer) \
                 VALUES (%s, %s, %s, %s, %s, %s)", (first_name, last_name, email, password, security_question, security_answer)
            )
            flash('You are now registered as an instructor and can login', 'success')
        else:
            flash('Invalid role. Please specify either "student" or "instructor"', 'danger')
            return redirect(url_for('register'))
        
        mysql.connection.commit()
        cursor.close()
        return redirect(url_for('login'))
    
    return render_template('register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        email = request.form['email']
        password_correct = request.form['password']
        cursor = mysql.connection.cursor()
        
        # Check if the user exists as a student
        result = cursor.execute("SELECT * FROM student WHERE email=%s", [email])
        if result > 0:
            data = cursor.fetchone()
            password = data['password']
            if sha256_crypt.verify(password_correct, password):
                session['logged_in'] = True
                session['role'] = 'student'
                session['email'] = email
                session['first_name'] = data['first_name']  
                session['last_name'] = data['last_name']
                flash(f"Welcome Student {session['first_name']} {session['last_name']}", 'success')
                return redirect(url_for('enroll'))
            else:
                flash('Invalid login. Please check your email and password', 'danger')
                return redirect(url_for('login'))
            
        # Check if the user exists as an instructor
        result = cursor.execute("SELECT * FROM instructor WHERE email=%s", [email])
        if result > 0:
            data = cursor.fetchone()
            password_hash = data['password']
            if sha256_crypt.verify(password_correct, password_hash):
                session['logged_in'] = True
                session['role'] = 'instructor'
                session['email'] = email
                session['first_name'] = data['first_name']  
                session['last_name'] = data['last_name']
                flash(f"Welcome Instructor {session['first_name']} {session['last_name']}", 'success')
                return redirect(url_for('classes'))
            else:
                flash('Invalid login. Please check your email and password', 'danger')
                return redirect(url_for('login'))
    
    return render_template('login.html', form=form)

#Forgot_password Route
@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        cursor = mysql.connection.cursor()
        # Check if the email exists in the student table
        result = cursor.execute("SELECT * FROM student WHERE email=%s", [email])
        if result > 0:
            role = 'student'
        else:
            # Check if the email exists in the instructor table
            result = cursor.execute("SELECT * FROM instructor WHERE email=%s", [email])
            if result > 0:
                role = 'instructor'
            else:
                # Email not found
                flash('Email not found.', 'danger')
                cursor.close()
                return render_template('forgot_password.html')
        # Fetch the security question
        cursor.execute(f"SELECT security_question FROM {role} WHERE email=%s", [email])
        security_question_data = cursor.fetchone()
        if security_question_data:
            security_question = security_question_data['security_question']
            # Set session variable indicating email confirmation pending
            session['confirm_email'] = email
        else:
            # Security question not found
            flash('Security question not found.', 'danger')
            cursor.close()
            return render_template('forgot_password.html', email=email)
        # Redirect to security question confirmation page with security question
        return redirect(url_for('reset_password', role=role, security_question=security_question))

    # Return the default view in case of GET request or if email not found
    return render_template('forgot_password.html')

#Reset Password Route
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    # Check if email confirmation is pending
    if 'confirm_email' not in session:
        # Redirect to the forgot_password page if email confirmation is pending
        flash('Please confirm your email address first.', 'danger')
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        email = session['confirm_email']
        role = request.form['role']
        security_answer = request.form['security_answer']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']

        if new_password == confirm_password:
            cursor = mysql.connection.cursor()
            cursor.execute(f"SELECT security_question, security_answer FROM {role} WHERE email=%s", [email])
            user_security_data = cursor.fetchone()
            if user_security_data and sha256_crypt.verify(security_answer, user_security_data['security_answer']):
                # Security answer correct, update password
                hashed_password = sha256_crypt.hash(new_password)
                cursor.execute(f"UPDATE {role} SET password=%s WHERE email=%s", (hashed_password, email))
                mysql.connection.commit()
                flash(f'Password reset successfully for {role}.', 'success')
                cursor.close()
                # Remove the session variable after confirming email
                session.pop('confirm_email', None)
                return redirect(url_for('login'))
            else:
                # Security answer incorrect
                flash('Security answer incorrect.', 'danger')
                cursor.close()
                return render_template('reset_password.html', email=email, role=role, security_question=user_security_data['security_question'])
        else:
            # Passwords don't match
            flash('Passwords do not match.', 'danger')
            return render_template('reset_password.html', email=email, role=role, security_question=user_security_data['security_question'])

    # If GET request, render the page with security question data
    email = session['confirm_email']
    role = request.args.get('role')
    security_question = request.args.get('security_question')
    return render_template('reset_password.html', email=email, role=role, security_question=security_question)

# Logout
@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))

@app.route('/people/<string:class_id>')
@is_logged_in
def people(class_id):
    # Check if the user is logged in
    if 'email' not in session:
        flash("You need to log in to view this page", "danger")
        return redirect(url_for('login'))  # Assuming you have a 'login' route
    # Fetch user information from the session
    student_first_name = session.get('first_name')
    student_last_name = session.get('last_name')
    # Fetch class details including instructor's name
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT c.*, i.first_name AS instructor_first_name, i.last_name AS instructor_last_name FROM classes c JOIN instructor i ON c.instructor_email = i.email WHERE c.id=%s", (class_id,))
    class_data = cursor.fetchone()
    if class_data:
        # Fetch enrolled students for this class
        cursor.execute("SELECT s.first_name, s.last_name FROM student s JOIN enrollments e ON s.email = e.student_email WHERE e.class_id=%s", (class_id,))
        enrolled_students = cursor.fetchall()
        cursor.close()
        return render_template('people.html', student_first_name=student_first_name, studnet_last_name=student_last_name, instructor_first_name=class_data['instructor_first_name'], instructor_last_name=class_data['instructor_last_name'], enrolled_students=enrolled_students, class_data=class_data)
    else:
        flash("Class not found", "danger")
        return redirect(url_for('dashboard')) 
    
# Dashboard Route
@app.route('/dashboard')
@is_logged_in
def dashboard():
    if 'logged_in' in session:
        if session['role'] == 'student':
            return render_template('dashboard.html', first_name=session.get('first_name'), last_name=session.get('last_name'))
        elif session['role'] == 'instructor':
            cursor = mysql.connection.cursor()
            cursor.execute("SELECT * FROM materials WHERE instructor=%s", [session['email']])
            materials = cursor.fetchall()
            cursor.close()
            return render_template('dashboard.html', materials=materials, first_name=session.get('first_name'), last_name=session.get('last_name'))
    else:
        flash('Unauthorized access. Please login', 'danger')
        return redirect(url_for('login'))

#########################Create Class ################################

@app.route('/classes', methods=['GET', 'POST'])
@is_logged_in
def classes():
    if session.get('role') == 'instructor':
        if request.method == 'POST':
            selected_banner = request.form.get('selected_banner')
            if selected_banner:
                cursor = mysql.connection.cursor()
                cursor.execute("UPDATE classes SET banners_filename = %s WHERE instructor_email = %s", (selected_banner, session['email']))
                mysql.connection.commit()
                cursor.close()
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM classes WHERE archived = 0 AND instructor_email = %s", (session['email'],))
        active_classes = cursor.fetchall()
        cursor.close()
        instructor_first_name = session.get('first_name')
        instructor_last_name = session.get('last_name')
        return render_template('classes.html', classes=active_classes, instructor_first_name=instructor_first_name, instructor_last_name=instructor_last_name)
    else:
        return redirect(url_for('enroll'))

@app.route('/archive_class/<int:class_id>')
@is_logged_in
def archive_class(class_id):
    # Get the instructor's email
    instructor_email = session['email']
    # Update the database to mark the class as archived
    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE classes SET archived = 1 WHERE id = %s AND instructor_email = %s", (class_id, instructor_email))
    mysql.connection.commit()
    cursor.close()
     # Get instructor's first name and last name from session
    instructor_firstname = session.get('firstname')
    instructor_lastname = session.get('lastname')
    # Flash a success message
    flash("Class has been archived", "success")
    return redirect(url_for('archived_classes', class_id=class_id,
                            instructor_firstname=instructor_firstname, instructor_lastname=instructor_lastname))

@app.route('/archived_classes')
@is_logged_in
def archived_classes():
    # Fetch archived classes from the database
  
    cursor = mysql.connection.cursor()
    if session.get('role') == 'instructor':
        # If the user is an instructor, fetch all archived classes
            cursor.execute("SELECT c.*, i.first_name AS instructor_first_name, i.last_name AS instructor_last_name FROM classes c JOIN instructor i ON c.instructor_email = i.email WHERE archived = 1")
    else:
        # If the user is a student, fetch only archived classes with the "Go" button
        cursor.execute("SELECT c.*, i.first_name AS instructor_first_name, i.last_name AS instructor_last_name FROM classes c JOIN instructor i ON c.instructor_email = i.email WHERE archived = 1")
    archived_classes = cursor.fetchall()
    cursor.close()

    # Get instructor's first name and last name from session
    instructor_firstname = session.get('firstname')
    instructor_lastname = session.get('lastname')

    return render_template('archived_classes.html', archived_classes=archived_classes,instructor_firstname=instructor_firstname, instructor_lastname=instructor_lastname)

@app.route('/unarchive_class/<int:class_id>')
@is_logged_in
def unarchive_class(class_id):
    # Get the instructor's email
    instructor_email = session['email']
    # Update the database to mark the class as unarchived
    cursor = mysql.connection.cursor()
    cursor.execute("UPDATE classes SET archived = 0 WHERE id = %s AND instructor_email = %s", (class_id, instructor_email))
    mysql.connection.commit()
    cursor.close()
    # Flash a success message
    flash("Class has been unarchived", "success")
    return redirect(url_for('classes'))

# Insert Class Route
@app.route('/insert', methods=['POST'])
@is_logged_in  # Require user to be logged in
def insert():
    if session['role'] == 'instructor':
        classname = request.form['classname']
        classcode = request.form['classcode']
        classsection = request.form['classsection']
        instructor_email = session['email']  # Get the email of the logged-in instructor
        # Check if the classcode already exists
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM classes WHERE classcode = %s", (classcode,))
        existing_class = cursor.fetchone()
        cursor.close()
        if existing_class:
            flash("Class code already exists. Please choose a different one.", "danger")
            return redirect(url_for('classes'))
        # Insert the class if class code doesn't exist
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO classes (classname, classcode, classsection, instructor_email) VALUES (%s,%s,%s,%s)",
                       (classname, classcode, classsection, instructor_email))
        mysql.connection.commit()
        cursor.close()
        flash("Data Inserted Successfully!")
        return redirect(url_for('classes'))
    else:
        flash("Only instructors can create classes", "danger")
        return redirect(url_for('classes'))

# Delete Class Route
@app.route('/delete/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def delete(id):
    flash("Record Has Been Deleted Successfully!")
    cursor = mysql.connection.cursor()
    cursor.execute("DELETE FROM classes WHERE id=%s", (id,))
    mysql.connection.commit()
    cursor.close()
    return redirect(url_for('classes'))

# Update Class Route
@app.route('/update', methods=['POST'])
@is_logged_in
def update():
    if request.method == 'POST':
        id = request.form['id']
        classname = request.form['classname']
        classcode = request.form['classcode']
        classsection = request.form['classsection']
        cursor = mysql.connection.cursor()
        # Check if the class code already exists in classes table, excluding the current class being updated
        cursor.execute("SELECT id FROM classes WHERE classcode=%s AND id!=%s", (classcode, id))
        existing_class = cursor.fetchone()
        if existing_class:
            flash("The class code you provided is already in use. Please choose a different one.", 'warning')
            cursor.close()
            return redirect(url_for('classes'))
        # Update the class details
        cursor.execute("UPDATE classes SET classname=%s, classcode=%s, classsection=%s WHERE id=%s", (classname, classcode, classsection, id))
        mysql.connection.commit()
        cursor.close()
        flash("Data Updated Successfully!", 'success')
        # Redirect to the classes page after update
        return redirect(url_for('classes'))


##################(ADD POST(Material)- Instructor Dashboard####################

#Material Form Class
class MaterialCForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=200)])
    description = TextAreaField('Description')  
    file = FileField('Upload File')

@app.route('/materialsc')
@is_logged_in
def materialsc():
    # Check if user is logged in
    if 'logged_in' in session:
        cursor = mysql.connection.cursor()
        # Fetch materials based on user role
        if session['role'] == 'instructor':
            cursor.execute("SELECT * FROM materialsc WHERE instructor=%s", [session['email']])
        else:  # For students, show all materials
            cursor.execute("SELECT * FROM materialsc")
        materialsc = cursor.fetchall()
         # Fetch class data
        cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
        class_data = cursor.fetchone()
        cursor.close()
        # Render template with materials
        if materials:
            return render_template('materialsc.html', materialsc=materialsc,class_data=class_data)
        else:
            message = 'No Materials Found'
            return render_template('materialsc.html', message=message,class_data=class_data)
    else:
        flash('Unauthorized access. Please login', 'danger')
        return redirect(url_for('login'))

#Single Material
@app.route('/materialc/<string:id>/')
@is_logged_in
def materialc(id):
    # Create cursor
    cursor = mysql.connection.cursor()
    # Get Material (Single material from database)
    result = cursor.execute("SELECT * FROM materialsc WHERE id=%s", [id])
    materialc = cursor.fetchone()

    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()

    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()
    cursor.close()

    return render_template('materialc.html', materialc=materialc, instructor_first_name=session.get('first_name'), instructor_last_name=session.get('last_name'),class_data=class_data )

# Constants for transcription and translation
MODEL_NAME = "openai/whisper-large-v3"
BATCH_SIZE = 8
CHUNK_LENGTH = 10  # Duration of each chunk in seconds
device = 0 if torch.cuda.is_available() else "cpu"
pipeline = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=CHUNK_LENGTH,
    device=device,
)
output_dir = "splitting_video_chunks"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

def split_video(video_path):
    video = mp.VideoFileClip(video_path)
    total_duration = video.duration
    start_times = range(0, int(total_duration), CHUNK_LENGTH)
    end_times = range(CHUNK_LENGTH, int(total_duration) + 1, CHUNK_LENGTH)
    chunks = []
    for start, end in zip(start_times, end_times):
        chunk = video.subclip(start, min(end, total_duration))
        audio_path = os.path.join(output_dir, f"audio_chunk_{start}_{end}.wav")
        chunk.audio.write_audiofile(audio_path)
        chunks.append(audio_path)
    return chunks

def transcribe(inputs, task):
    if inputs is None:
        raise ValueError("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipeline(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    return {"text": text}


@app.route('/add_materialc', methods=['POST'])
def add_materialc():
    form = MaterialCForm()
    email=session['email']
    date = datetime.now()
    date = date.strftime("%Y-%m-%d %H:%M:%S")
    email = session['email']
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if request.method == 'POST':
        name = request.form['name']
        description = request.form['description']
        class_id = request.form['class_id']
        topic_id = request.form['topic']
        file = request.files['file']

        # Ensure a file is selected
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        # Determine file type based on extension
        allowed_video_extensions = {'mp4', 'avi', 'mkv'}
        allowed_image_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        allowed_pdf_extensions = {'pdf'}

        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else None

        if file_extension in allowed_video_extensions:
            file_type = 'video'
        elif file_extension in allowed_image_extensions:
            file_type = 'image'
        elif file_extension in allowed_pdf_extensions:
            file_type = 'pdf'
        else:
            flash('Unsupported file format', 'error')
            return redirect(request.url)

        # Construct folder path based on class ID
        classwork_folder = os.path.join(app.config['CLASSWORK_UPLOAD_FOLDER'], f'classwork_class_{class_id}')
        if not os.path.exists(classwork_folder):
            os.makedirs(classwork_folder)

        # Save the file into the classwork folder with a secure filename
        file_path = os.path.join(classwork_folder, secure_filename(file.filename))
        file.save(file_path)

        # If it's a video, transcribe and translate it
        if file_type == 'video':
            # Split the video into chunks
            video_chunks = split_video(file_path)
            results = []
            for chunk_path in video_chunks:
                transcription_result = transcribe(chunk_path, 'transcribe')
                translation_result = transcribe(chunk_path, 'translate')

                # Save results to MongoDB
                data_to_save = {
                    'date': date,
                    'class_id':class_id,
                    'class_id': class_id,
                    'email': email,
                    'video_name': file.filename,
                    'transcription': transcription_result['text'],
                    'translation': translation_result['text'],
                    'chunk_path': chunk_path
                }
                transcription_collection.insert_one(data_to_save)
                # Append results for further processing if needed
                results.append(data_to_save)
            print("Transcription and Translation Results:", results)

        # Insert file details into the database
            # Create PDF with transcription and translation
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            for result in results:
                pdf.cell(200, 10, txt=f"Video Chunk: {result['chunk_path']}", ln=True)
                pdf.multi_cell(0, 10, txt=f"Translation: {result['translation']}")
                pdf.ln(10)

            pdf_file_path = os.path.join(classwork_folder, f"{secure_filename(file.filename).rsplit('.', 1)[0]}_translation.pdf")
            pdf.output(pdf_file_path)

        cursor = mysql.connection.cursor()
        if session['role'] == 'instructor':
            cursor.execute("INSERT INTO materialsc(name, description, instructor, file_path, file_type, class_id, topic_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (name, description, session['email'], file_path, file_type, class_id, topic_id))
        else:
            cursor.execute("INSERT INTO materialsc(name, description, student, file_path, file_type, class_id, topic_id) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (name, description, session['email'], file_path, file_type, class_id, topic_id))
        
        # Commit the changes to the database
        mysql.connection.commit()
        cursor.close()
        
        flash('Material Created', 'success')
        return redirect(url_for('classwork', class_id=class_id))

    # Fetch list of topics for the class
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, name FROM topic WHERE class_id = %s", (class_id,))
    topics = cursor.fetchall()
    cursor.close()
    flash('Invalid form data', 'error')
    return render_template('add_materialc.html', form=form, class_id=class_id, topics=topics)

@app.route('/delete_materialc/<string:id>', methods=['POST'])
@is_logged_in
def delete_materialc(id):
    # Create cursor
    cursor = mysql.connection.cursor()

    # Fetch file_path from the database before deletion
    cursor.execute("SELECT file_path FROM materialsc WHERE id=%s", [id])
    material = cursor.fetchone()
    file_path = material['file_path'] if material else None

    # Execute the deletion query
    cursor.execute("DELETE FROM materialsc WHERE id=%s", [id])

    # Commit to DB
    mysql.connection.commit()

    flash('Material Deleted', 'success')

    # Close cursor
    cursor.close()

    if file_path:
        # Delete the file from the system if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
            flash('File Deleted from System', 'success')
        else:
            flash('File Not Found in System', 'error')

    class_id = request.args.get('class_id')  # Get class_id from URL parameters

    # Redirect to classwork with the correct class_id
    return redirect(url_for('classwork', class_id=class_id))

 # Update Material
@app.route('/edit_materialc/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_materialc(id):
    # Create cursor 
    cursor = mysql.connection.cursor()

    # Get material by id
    result = cursor.execute("SELECT * FROM materialsc WHERE id=%s", [id])
    materialc = cursor.fetchone() 
    cursor.close()

    # Get form 
    form = MaterialCForm(request.form)

    # Populate form fields with material data
    if request.method != 'POST':
        form.name.data = materialc['name']
        form.description.data = materialc['description']
    
    class_id = materialc['class_id']

    if request.method == 'POST' and form.validate():
        name = request.form['name']
        description = request.form['description']
        file = request.files['file']

        # Check if a new file has been uploaded
        if file:
            # Ensure the classwork folder exists
            classwork_folder = os.path.join(app.config['CLASSWORK_UPLOAD_FOLDER'], f'classwork_class_{class_id}')
            if not os.path.exists(classwork_folder):
                os.makedirs(classwork_folder)

            # Save the uploaded file in the classwork folder with a secure filename
            new_file_path = os.path.join(classwork_folder, secure_filename(file.filename))
            file.save(new_file_path)

            # Delete the previous file if it exists
            if os.path.exists(materialc['file_path']):
                os.remove(materialc['file_path'])

            # Update material with new file path
            cursor = mysql.connection.cursor()
            cursor.execute("UPDATE materialsc SET name=%s, description=%s, file_path=%s WHERE id=%s",
                           (name, description, new_file_path, id))
            mysql.connection.commit()
            cursor.close()

            flash('Material Updated', 'success')
            return redirect(url_for('classwork', class_id=class_id))
        else:
            # No file uploaded, display a message
            danger_message = 'Please upload a file.'
            return render_template('edit_materialc.html', form=form, danger_message=danger_message, current_file=materialc.get('file_path'), class_id=class_id)

    # Render the template with the form and current file path
    return render_template('edit_materialc.html', form=form, current_file=materialc.get('file_path'), class_id=class_id)


#Material Form Class
class MaterialForm(Form):
    name = StringField('Name', [validators.Length(min=1, max=200)])
    description = TextAreaField('Description')  
    file = FileField('Upload File')

# Materials Route
@app.route('/materials')
@is_logged_in
def materials():
    # Check if user is logged in
    if 'logged_in' in session:
        cursor = mysql.connection.cursor()
        # Fetch materials based on user role
        if session['role'] == 'instructor':
            cursor.execute("SELECT * FROM materials WHERE instructor=%s", [session['email']])
        else:  # For students, show all materials
            cursor.execute("SELECT * FROM materials")
        materials = cursor.fetchall()
         # Fetch class data
        cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
        class_data = cursor.fetchone()
        cursor.close()
        # Render template with materials
        if materials:
            return render_template('materials.html', materials=materials,class_data=class_data)
        else:
            message = 'No Materials Found'
            return render_template('materials.html', message=message,class_data=class_data)
    else:
        flash('Unauthorized access. Please login', 'danger')
        return redirect(url_for('login'))

@app.route('/material/<string:id>/')
@is_logged_in
def material(id):
    # Create cursor
    cursor = mysql.connection.cursor()
    
    # Get Material (Single material from database)
    result = cursor.execute("SELECT * FROM materials WHERE id=%s", [id])
    material = cursor.fetchone()
    
    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()
    cursor.close()

    
    return render_template(
        'material.html',
        material=material,
        instructor_first_name=session.get('first_name'),
        instructor_last_name=session.get('last_name'),
        class_data=class_data
    )

# Save Material Route
@app.route('/add_material', methods=['GET', 'POST'])
@is_logged_in
def add_material():
    form = MaterialForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        description = form.description.data
        files = request.files.getlist('file[]')  # Get list of files
        class_id = request.form.get('class_id')  # Get class_id from form
        # Iterate over each file
        for file in files:
            if file:
                # Determine file type based on extension
                if file.filename.endswith(('.pdf', '.mp4', '.avi', '.mkv')):
                    file_type = 'pdf' if file.filename.endswith('.pdf') else 'video'
                elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_type = 'image'
                else:
                    # Handle unsupported file types
                    flash('Unsupported file format', 'error')
                    return redirect(url_for('add_material', class_id=class_id))

                
                # Create folder with class_id name if it doesn't exist
                class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'class_{class_id}')
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                # Save the file in the class_id folder
                file_path = os.path.join(class_folder, secure_filename(file.filename))
                file.save(file_path)

                # Save assignment to the database
                cursor = mysql.connection.cursor()

                # Use the user's role to determine whether it's an instructor or student uploading the material
                if session['role'] == 'instructor':
                    cursor.execute("INSERT INTO materials(name, description, instructor, file_path, file_type, class_id) VALUES (%s,%s,%s,%s,%s,%s)",
                                   (name, description, session['email'], file_path, file_type, class_id))
                else:
                    cursor.execute("INSERT INTO materials(name, description, student, file_path, file_type, class_id) VALUES (%s,%s,%s,%s,%s,%s)",
                                   (name, description, session['email'], file_path, file_type, class_id))

                mysql.connection.commit()
                cursor.close()

        flash('Material(s) Created', 'success')
        return redirect(url_for('class_content', class_id=class_id))  # Redirect to class_content with class_id

    # If method is GET or form validation fails, render the form
    return render_template('add_material.html', form=form)

# Update Material Route
@app.route('/edit_material/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_material(id):
    # Create cursor
    cursor = mysql.connection.cursor()

    # Get material by id
    result = cursor.execute("SELECT * FROM materials WHERE id=%s", [id])
    material = cursor.fetchone()
    cursor.close()

    # Get form
    form = MaterialForm(request.form)

    # Populate form fields with material data
    if request.method != 'POST':
        form.name.data = material['name']
        form.description.data = material['description']

    class_id = material['class_id']

    if request.method == 'POST' and form.validate():
        name = request.form['name']
        description = request.form['description']
        file = request.files['file']

        # Check if a new file has been uploaded
        if file:
            # Remove previous file
            if os.path.exists(material['file_path']):
                os.remove(material['file_path'])

            # Create folder with class_id name if it doesn't exist
            class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'class_{class_id}')
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Save the uploaded file in the class_id folder
            file_path = os.path.join(class_folder, secure_filename(file.filename))
            file.save(file_path)

            # Update material with new file path
            cursor = mysql.connection.cursor()
            cursor.execute("UPDATE materials SET name=%s, description=%s, file_path=%s WHERE id=%s",
                           (name, description, file_path, id))
        else:
            # No file uploaded, update material without file path
            cursor = mysql.connection.cursor()
            cursor.execute("UPDATE materials SET name=%s, description=%s WHERE id=%s",
                           (name, description, id))

        mysql.connection.commit()
        cursor.close()

        flash('Material Updated', 'success')
        return redirect(url_for('class_content', class_id=class_id))

    # Render the template with the form and current file path
    return render_template('edit_material.html', form=form, current_file=material.get('file_path'), class_id=class_id)

# Delete Material Route
@app.route('/delete_material/<string:id>', methods=['POST'])
@is_logged_in 
def delete_material(id):
    # Create cursor
    cursor = mysql.connection.cursor()

    # Get material information
    cursor.execute("SELECT * FROM materials WHERE id=%s", [id])
    material = cursor.fetchone()

    # Execute (delete from the materials table in the database)
    cursor.execute("DELETE FROM materials WHERE id=%s", [id])

    # Commit to DB
    mysql.connection.commit()

    # Close connection
    cursor.close()

    class_id = material['class_id']
    file_path = material['file_path']

    # Remove file from storage
    if os.path.exists(file_path):
        os.remove(file_path)
        flash('Material Deleted', 'success')
    else:
        flash('File not found', 'error')

    return redirect(url_for('class_content', class_id=class_id))


@app.route('/download_file/<path:file_path>')
@is_logged_in
def download_file(file_path):
    return send_file(file_path, as_attachment=True)

@app.route('/view_file/<path:file_path>')
@is_logged_in
def view_file(file_path):
  
    # Determine MIME type based on file extension
    if file_path.endswith('.pdf'):
        return send_file(file_path, mimetype='application/pdf')
    elif file_path.endswith(('.mp4', '.avi', '.mkv')):
        # Assuming your video files are stored in a separate directory named 'videos'
        # Adjust this path based on your actual file storage configuration
        return send_file(file_path, mimetype='video/mp4')
    elif file_path.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return send_file(file_path, mimetype='image/jpeg')  # Adjust mimetype according to your image formats
    else:
        # Handle other file types or unsupported formats
        return "Unsupported file format"

##################(ADD POST(Assignment)- Instructor Dashboard####################
#Assignment Form Class
class AssignmentForm(Form):
    name = StringField('Title', [validators.Length(min=1, max=200)])
    description = TextAreaField('Description')  
    file = FileField('Upload File')
    due_date = DateField('Due date & time', format='%Y-%m-%d', validators=(validators.DataRequired(),))
    duration=TimeField('Time (optional)')    
    score=StringField('Points',[validators.Length(min=1, max=100)])
   
# Assignments Route
@app.route('/assignments')
@is_logged_in
def assignments():
    # Check if user is logged in
    if 'logged_in' in session:
        cursor = mysql.connection.cursor()
        
        # Fetch materials based on user role
        if session['role'] == 'instructor':
            cursor.execute("SELECT * FROM assignments WHERE instructor=%s", [session['email']])
        else:  # For students, show all materials
            cursor.execute("SELECT * FROM assignments")
        assignments = cursor.fetchall()
        cursor.close()

        assignments = cursor.fetchone()
        cursor.execute("SELECT * FROM assignments WHERE id=%s", [session.get("class_id")])
        class_data= cursor.fetchone()

        # Render template with materials
        if assignments:
            return render_template('assignments.html', assignments=assignments, class_data=class_data)
        else:
            message = 'No Materials Found'
            return render_template('assignments.html', message=message, class_data=class_data)
    else:
        flash('Unauthorized access. Please login', 'danger')
        return redirect(url_for('login'))

# Single Assignment
@app.route('/assignment/<string:id>/')
@is_logged_in
def assignment(id):
    # Create cursor
    cursor = mysql.connection.cursor()

    # Get Assignment (Single Assignment from database)
    result = cursor.execute("SELECT * FROM assignments WHERE id=%s", [id])

    assignment = cursor.fetchone()

    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()
    cursor.close()

    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()
    cursor.close()
    return render_template('assignment.html', assignment=assignment, instructor_first_name=session.get('first_name'), instructor_last_name=session.get('last_name'), class_data=class_data )

#Add Assignment
@app.route('/add_assignment', methods=['GET', 'POST'])
@is_logged_in
def add_assignment():
    form = AssignmentForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        description = form.description.data
        due_date = form.due_date.data
        score = form.score.data
        topic_id = request.form['topic']  # Get the selected topic ID from the form
        files = request.files.getlist('file')  # Corrected field name
        class_id = request.form.get('class_id')  # Get class_id from form
        class_data = request.args.get(class_id)

        # Iterate over each file
        for file in files:
            if file:
                # Determine file type based on extension
                if file.filename.endswith(('.pdf', '.mp4', '.avi', '.mkv')):
                    file_type = 'pdf' if file.filename.endswith('.pdf') else 'video'
                elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    file_type = 'image'
                else:
                    # Handle unsupported file types
                    flash('Unsupported file format', 'error')
                    return redirect(url_for('add_assignment'))

                # Ensure the classwork folder exists
                classwork_folder = os.path.join(app.config['CLASSWORK_UPLOAD_FOLDER'], f'classwork_class_{class_id}')
                if not os.path.exists(classwork_folder):
                    os.makedirs(classwork_folder)

                # Save the file in the classwork folder with a secure filename
                file_path = os.path.join(classwork_folder, secure_filename(file.filename))
                file.save(file_path)

                # Save assignment to the database
                cursor = mysql.connection.cursor()
                cursor.execute("INSERT INTO assignments (name, description, score, due_date, instructor, file_path, file_type, class_id, topic_id) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s)",
                               (name, description, score, due_date, session['email'], file_path, file_type, class_id, topic_id))
                mysql.connection.commit()
                cursor.close()

        flash('Assignment(s) Created', 'success')
        return redirect(url_for('classwork', class_id=class_id))  # Redirect to classwork page after successful assignment creation

    # Fetch list of topics for the class
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, name FROM topic WHERE class_id = %s", (class_id,))
    topics = cursor.fetchall()
    cursor.close()
    flash('Invalid form data', 'error')

    return render_template('add_assignment.html', form=form, class_id=class_id, class_data=class_data, topics=topics)

#Delete Assignment
@app.route('/delete_assignment/<string:assignment_id>', methods=['POST'])
@is_logged_in
def delete_assignment(assignment_id):
    if request.method == 'POST':
        # Create cursor
        cursor = mysql.connection.cursor()

        # Fetch file_path from the database before deletion
        cursor.execute("SELECT file_path FROM assignments WHERE id=%s", [assignment_id])
        assignment = cursor.fetchone()
        file_path = assignment['file_path'] if assignment else None

        # Execute deletion query
        cursor.execute("DELETE FROM assignments WHERE id=%s", [assignment_id])

        # Commit to DB
        mysql.connection.commit()

        # Close cursor
        cursor.close()

        # Delete file from system folder if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            flash('File Deleted from System', 'success')
        else:
            flash('File Not Found in System', 'error')

        class_id = request.args.get('class_id')  # Get class_id from URL parameters
        flash('Assignment Deleted', 'success')
        return redirect(url_for('classwork', class_id=class_id))

#Edit Assignment
@app.route('/edit_assignment/<string:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_assignment(id):
    cursor = mysql.connection.cursor()
    result = cursor.execute("SELECT * FROM assignments WHERE id=%s", [id])
    assignment = cursor.fetchone()
    cursor.close()
    
    form = AssignmentForm(request.form)
    
    if request.method == 'POST' and form.validate():
        name = request.form['name']
        description = request.form['description']
        due_date = request.form['due_date']
        score = request.form['score']
        file = request.files['file']
        class_id = assignment['class_id']
        if file:
            # Ensure the classwork folder exists
            classwork_folder = os.path.join(app.config['CLASSWORK_UPLOAD_FOLDER'], f'classwork_class_{class_id}')
            if not os.path.exists(classwork_folder):
                os.makedirs(classwork_folder)

            # Save the uploaded file in the classwork folder with a secure filename
            new_file_path = os.path.join(classwork_folder, secure_filename(file.filename))
            file.save(new_file_path)

            # Delete the previous file if it exists
            if os.path.exists(assignment['file_path']):
                os.remove(assignment['file_path'])

            # Update assignment with new file path
            cursor = mysql.connection.cursor()
            cursor.execute("UPDATE assignments SET name=%s, description=%s, score=%s, due_date=%s, file_path=%s WHERE id=%s",
                           (name, description, score, due_date, new_file_path, id))
            mysql.connection.commit()
            cursor.close()
            flash('Assignment Updated', 'success')
            return redirect(url_for('classwork', class_id=class_id))
        else:
            danger_message = 'Please upload a file.'
            return render_template('edit_assignment.html', form=form, danger_message=danger_message, current_file=assignment.get('file_path'), class_id=class_id)
    else:
        # Pre-fill form with current assignment data
        form.name.data = assignment['name']
        form.description.data = assignment['description']
        form.score.data = assignment['score']
        form.due_date.data = assignment['due_date']
    
    return render_template('edit_assignment.html', form=form, current_file=assignment.get('file_path'), class_id=assignment['class_id'])


    ##################(ADD POST(Question)- Instructor Dashboard####################
#question Form Class
class QuestionForm(Form):
    name = StringField('Title', [validators.Length(min=1, max=200)])
    description = TextAreaField('Description')  
    file = FileField('Upload File')
    due_date = DateField('Due Date', format='%Y-%m-%d', validators=(validators.DataRequired(),))
    duration=TimeField('Time',validators=(validators.DataRequired(),))
    score=StringField('Points',[validators.Length(min=1, max=100)])

# questions Route
@app.route('/questions')
@is_logged_in
def questions():
    # Check if user is logged in
    if 'logged_in' in session:
        cursor = mysql.connection.cursor()
        # Fetch materials based on user role
        if session['role'] == 'instructor':
            cursor.execute("SELECT * FROM questions WHERE instructor=%s", [session['email']])
        else:  # For students, show all materials
            cursor.execute("SELECT * FROM questions")
        questions = cursor.fetchall()
        cursor.close()
        questions = cursor.fetchall()
        cursor.execute("SELECT * FROM questions WHERE id=%s", [session.get('class_id')])
        class_data = cursor.fetchone()
        # Render template with materials
        if question:
            return render_template('question.html', question=question,class_data=class_data)
        else:
            message = 'No Materials Found'
            return render_template('question.html', message=message,class_data=class_data)
    else:
        flash('Unauthorized access. Please login', 'danger')
        return redirect(url_for('login'))

# Single question
@app.route('/question/<string:id>/')
@is_logged_in
def question(id):
    # Create cursor
    cursor = mysql.connection.cursor()
    # Get question (Single question from database)
    result = cursor.execute("SELECT * FROM questions WHERE id=%s", [id])
    question = cursor.fetchone()
    # Fetch class data
    cursor.execute("SELECT * FROM classes WHERE id=%s", [session.get('class_id')])
    class_data = cursor.fetchone()
    cursor.close()
    return render_template('question.html', question=question, instructor_first_name=session.get('first_name'), instructor_last_name=session.get('last_name'),class_data=class_data )

# Add question Route
@app.route('/add_question', methods=['GET', 'POST'])
@is_logged_in
def add_question():
    class_id = request.form.get('class_id')  # Get class_id from form in POST request
    class_data = request.args.get('class_data')  # Fetch class_data from request args
    form = QuestionForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        description = form.description.data
        due_date = form.due_date.data
        duration = form.duration.data
        score = form.score.data
        topic_id = request.form['topic']  # Get the selected topic ID from the form

        # Save question to the database
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO questions(name, description, instructor, score, due_date, duration, class_id, topic_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                       (name, description, session['email'], score, due_date, duration, class_id, topic_id))
        mysql.connection.commit()
        cursor.close()
        flash('Question Created', 'success')
        return redirect(url_for('classwork', class_id=class_id))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, name FROM topic WHERE class_id = %s", (class_id,))
    topics = cursor.fetchall()
    cursor.close()
    flash('Invalid form data', 'error')
    return render_template('add_question.html', form=form, class_id=class_id, class_data=class_data, topics=topics)


#Delete Question
@app.route('/delete_question/<string:question_id>', methods=['POST'])
@is_logged_in
def delete_question(question_id):
    if request.method == 'POST':
        # Create cursor
        cursor = mysql.connection.cursor()

        # Execute deletion query
        cursor.execute("DELETE FROM questions WHERE id=%s", [question_id])

        # Commit to DB
        mysql.connection.commit()

        # Close cursor
        cursor.close()
        class_id = request.args.get('class_id')  # Get class_id from URL parameters
        flash('Question Deleted', 'success')
        return redirect(url_for('classwork', class_id=class_id))
#Edit Question
@app.route('/edit_question/<int:id>', methods=['GET', 'POST'])
@is_logged_in
def edit_question(id):
    # Fetch question from the database
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM questions WHERE id = %s", (id,))
    question = cursor.fetchone()
    
    # Create form and populate with question data
    form = QuestionForm(request.form)
    
    if request.method == 'POST' and form.validate():
        name = form.name.data
        description = form.description.data
        due_date = form.due_date.data
        duration = form.duration.data
        score = form.score.data
        class_id = question['class_id']
    
        # Update question in the database
        cursor.execute("UPDATE questions SET name=%s, description=%s, due_date=%s, duration=%s, score=%s WHERE id=%s",
                       (name, description, due_date, duration, score, id))
        cursor.execute("UPDATE questions SET name=%s, description=%s, due_date=%s, duration=%s, score=%s WHERE id=%s",
                       (name, description, due_date, duration, score, id))
        mysql.connection.commit()
        cursor.close()
        
        flash('Question updated', 'success')
        return redirect(url_for('classwork', class_id=class_id))
    
    return render_template('edit_question.html', form=form, current_file=question.get('file_path'))

#############################################################################################3
@app.route('/class_content/<string:class_id>')
@is_logged_in
def class_content(class_id):
    session['class_id'] = class_id
    # Determine the user's role
    user_role = session.get('role')  # Assuming you store the user's role in the session
    image_id = request.args.get('image_id')
    # Fetch user details
    user_email = session.get('email')  # Assuming you store the user's email in the session
    user_cursor = mysql.connection.cursor()
    user_first_name = ""
    user_last_name = ""
    instructor_first_name = ""
    instructor_last_name = ""
    if user_role == 'student':
        user_cursor.execute("SELECT first_name, last_name FROM student WHERE email=%s", [user_email])
    elif user_role == 'instructor':
        user_cursor.execute("SELECT first_name, last_name FROM instructor WHERE email=%s", [user_email])
    user_data = user_cursor.fetchone()
    if user_data:
        user_first_name = user_data['first_name']
        user_last_name = user_data['last_name']
    user_cursor.close()
    # Fetch class details including instructor's name and banner filename
    class_cursor = mysql.connection.cursor()
    class_cursor.execute("SELECT c.*, i.first_name AS instructor_first_name, i.last_name AS instructor_last_name, c.banners_filename FROM classes c JOIN instructor i ON c.instructor_email = i.email WHERE c.id=%s", [class_id])
    class_data = class_cursor.fetchone()
    if class_data:
        instructor_first_name = class_data['instructor_first_name']
        instructor_last_name = class_data['instructor_last_name']
        banners_filename = class_data['banners_filename']  # Retrieve the banners filename
    class_cursor.close()
    # Check if class exists
    if not class_data:
        flash("Class not found", "danger")
        return redirect(url_for('classes'))  # Redirect to a suitable page if the class doesn't exist
  # Fetch materials based on class_id
    material_cursor = mysql.connection.cursor()
    material_cursor.execute("""
        SELECT 
            m.*, 
            COALESCE(s.first_name, i.first_name) AS user_first_name, 
            COALESCE(s.last_name, i.last_name) AS user_last_name, 
            CASE WHEN s.email IS NOT NULL THEN 'student' ELSE 'instructor' END AS user_role,
            'material' AS type 
        FROM 
            materials m 
        LEFT JOIN 
            student s ON m.student = s.email 
        LEFT JOIN 
            instructor i ON m.instructor = i.email 
        WHERE 
            m.class_id = %s 
        ORDER BY 
            m.id DESC
    """, [class_id])
    materials_data = material_cursor.fetchall()
    material_cursor.close()
    material_comment_cursor = mysql.connection.cursor()
    material_comment_cursor.execute("""
        SELECT mc.*, 
            s.first_name AS student_first_name, 
            s.last_name AS student_last_name, 
            i.first_name AS instructor_first_name, 
            i.last_name AS instructor_last_name,
            'material_comment' AS type 
        FROM material_comment mc
        LEFT JOIN student s ON mc.student_email = s.email 
        LEFT JOIN instructor i ON mc.instructor_email = i.email 
        WHERE mc.class_id = %s 
        ORDER BY mc.id DESC
    """, [class_id])
    material_comments_data = material_comment_cursor.fetchall()
    # Convert the cursor result to a list of dictionaries
    material_comments_data = [dict(row) for row in material_comments_data]
    material_comment_cursor.close()
   # Fetch comments for this class
    comment_cursor = mysql.connection.cursor()
    comment_cursor.execute("""
        SELECT c.*, 
            s.first_name AS student_first_name, 
            s.last_name AS student_last_name, 
            i.first_name AS instructor_first_name, 
            i.last_name AS instructor_last_name,
            'comment' AS type 
        FROM comment c 
        LEFT JOIN student s ON c.student_email = s.email 
        LEFT JOIN instructor i ON c.instructor_email = i.email 
        WHERE class_id=%s 
        ORDER BY c.id DESC
    """, [class_id])
    comments_data = comment_cursor.fetchall()
    # Convert the cursor result to a list of dictionaries
    comments_data = [dict(row) for row in comments_data]
    for comment in comments_data:
        if user_role == 'student':
            comment['user_first_name'] = user_first_name
            comment['user_last_name'] = user_last_name
        elif user_role == 'instructor':
            comment['instructor_first_name'] = instructor_first_name
            comment['instructor_last_name'] = instructor_last_name
    for comment in comments_data:
        # Fetch replies for each comment
        reply_cursor = mysql.connection.cursor()
        reply_cursor.execute("""
        SELECT r.*, 
            CASE
                WHEN r.instructor_email IS NOT NULL THEN SUBSTRING_INDEX(r.instructor_email, '@', 1)
                ELSE SUBSTRING_INDEX(r.student_email, '@', 1)
            END AS user_first_name,
            CASE
                WHEN r.instructor_email IS NULL THEN ''
                ELSE SUBSTRING_INDEX(SUBSTRING_INDEX(r.student_email, '@', 1), '.', -1)
            END AS user_last_name
        FROM reply AS r
        WHERE r.comment_id = %s
        ORDER BY r.create_date DESC
    """, [comment['id']])
        comment['replies'] = reply_cursor.fetchall()
        # Convert the cursor result to a list of dictionaries
        comment['replies'] = [dict(row) for row in comment['replies']]
        reply_cursor.close()
        # Update reply data with user/instructor names
        '''for reply in comment['replies']:
            if user_role == 'student':
                reply['user_first_name'] = user_first_name
                reply['user_last_name'] = user_last_name
            elif user_role == 'instructor':
                reply['instructor_first_name'] = instructor_first_name
                reply['instructor_last_name'] = instructor_last_name'''
    comment_cursor.close()
    # Fetch uploaded images for this class
    image_cursor = mysql.connection.cursor()
    image_cursor.execute("""
        SELECT 
            *, 
            'image' AS type,
            CASE 
                WHEN instructor_email IS NOT NULL THEN SUBSTRING_INDEX(instructor_email, '@', 1)
                ELSE SUBSTRING_INDEX(student_email, '@', 1)
            END AS first_name,
            CASE 
                WHEN instructor_email IS NULL THEN ''
                ELSE SUBSTRING_INDEX(SUBSTRING_INDEX(student_email, '@', 1), '.', -1)
            END AS last_name
        FROM 
            uploaded_images 
        WHERE 
            class_id = %s 
        ORDER BY 
            id DESC
    """, [class_id])
    images_data = image_cursor.fetchall()
    image_cursor.close()
     # Fetch comments for images in this class
    image_comment_cursor = mysql.connection.cursor()
    image_comment_cursor.execute("""
        SELECT ic.*, 
            s.first_name AS student_first_name, 
            s.last_name AS student_last_name, 
            i.first_name AS instructor_first_name, 
            i.last_name AS instructor_last_name,
            'image_comment' AS type 
        FROM image_comment ic
        LEFT JOIN student s ON ic.student_email = s.email 
        LEFT JOIN instructor i ON ic.instructor_email = i.email 
        WHERE ic.class_id = %s 
        ORDER BY ic.id DESC
    """, [class_id])
    image_comments_data = image_comment_cursor.fetchall()
    # Convert the cursor result to a list of dictionaries
    image_comments_data = [dict(row) for row in image_comments_data]
    for comment in image_comments_data:
        if user_role == 'student':
            comment['user_first_name'] = user_first_name
            comment['user_last_name'] = user_last_name
        elif user_role == 'instructor':
            comment['instructor_first_name'] = instructor_first_name
            comment['instructor_last_name'] = instructor_last_name
    image_comment_cursor.close()
    # Extract class name and section from class data
    classname = class_data['classname']
    classsection = class_data['classsection']
    return render_template('class_content.html', class_data=class_data, materials=materials_data, material_comments_data=material_comments_data,
                        comments=comments_data, images=images_data, image_comments_data=image_comments_data,
                        user_first_name=user_first_name,
                        user_last_name=user_last_name,
                        instructor_first_name=instructor_first_name,
                        instructor_last_name=instructor_last_name,
                        classname=classname, classsection=classsection,
                        user_role=user_role, image_id=image_id,banner_filename=banners_filename)

@app.route('/save_banner', methods=['POST'])
@is_logged_in
def save_banner():
    if request.method == 'POST':
        selected_banner = request.form.get('selected_banner')
        class_id = session.get('class_id')
        # Update the database with the selected banner filename
        cursor = mysql.connection.cursor()
        try:
            cursor.execute("UPDATE classes SET banners_filename = %s WHERE id = %s", (selected_banner, class_id))
            mysql.connection.commit()
            cursor.close()
            return jsonify({'success': True, 'message': 'Selected banner saved successfully'})
        except Exception as e:
            mysql.connection.rollback()
            cursor.close()
            return jsonify({'success': False, 'message': 'Error saving selected banner: {}'.format(str(e))})
    else:
        return jsonify({'success': False, 'message': 'Method not allowed'})

# Add a route for posting comments
@app.route('/post_comment', methods=['POST'])
@is_logged_in
def post_comment():
    if request.method == 'POST':
        comment = request.form['comment']
        #instructor_email = session['email']
        #student_email = session['email']
        class_id = request.form.get('class_id')  # Retrieve class_id from the form
        user_role = session.get('role')
        user_email = session.get('email')
         # Insert comment into the database
        cursor = mysql.connection.cursor()
        if user_role == 'instructor':
            cursor.execute("INSERT INTO comment (comment, instructor_email, class_id) VALUES ( %s, %s,%s)", (comment, user_email, class_id))
        elif user_role == 'student':
            cursor.execute("INSERT INTO comment (comment, student_email, class_id) VALUES ( %s, %s,%s)", (comment, user_email, class_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Posted', 'success')
        return redirect(url_for('class_content', class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add a route for posting replies to comments
@app.route('/post_reply/<int:comment_id>', methods=['POST'])
@is_logged_in
def post_reply(comment_id):
    if request.method == 'POST':
        reply = request.form['reply']
        #instructor_email = session['email']
        class_id = request.form.get('class_id')  # Retrieve class_id from the form
        user_role = session.get('role')
        user_email = session.get('email')
        # Fetch instructor information from the database
        '''cursor = mysql.connection.cursor()
        cursor.execute("SELECT first_name, last_name FROM instructor WHERE email = %s", (instructor_email,))
        instructor_info = cursor.fetchone()  # Assuming there's only one instructor with the given email
        cursor.close()'''
        # Insert the reply into the database
        cursor = mysql.connection.cursor()
        if user_role == 'instructor':
            cursor.execute("INSERT INTO reply (reply, instructor_email, comment_id,class_id) VALUES (%s, %s, %s,%s)",
                       (reply, user_email, comment_id,class_id))
        elif user_role == 'student':
            cursor.execute("INSERT INTO reply (reply, student_email, comment_id,class_id) VALUES (%s, %s, %s,%s)",
                       (reply, user_email, comment_id,class_id))
        mysql.connection.commit()
        cursor.close()
        flash('Reply Posted', 'success')
        return redirect(url_for('class_content',class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

@app.route('/edit_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def edit_comment(comment_id):
    if request.method == 'POST':
        comment_content = request.form['comment_content']
        # Update comment in the database based on comment_id
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE comment SET comment = %s WHERE id = %s", (comment_content, comment_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Edited', 'success')
        # Retrieve class_id from the form or the comment itself
        class_id = request.form.get('class_id')  # Assuming the class_id is present in the form
        return redirect(url_for('class_content', class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add route for deleting comments
@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def delete_comment(comment_id):
    if request.method == 'POST':
        # Delete comment from the database based on comment_id
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM comment WHERE id = %s", [comment_id])
        mysql.connection.commit()
        cursor.close()
        class_id = request.form.get('class_id')  # Get class_id from form
        flash('Comment Deleted', 'success')
        return redirect(url_for('class_content',class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add route for editing replies
@app.route('/edit_reply/<int:reply_id>', methods=['POST'])
@is_logged_in
def edit_reply(reply_id):
    if request.method == 'POST':
        reply_content = request.form['reply_content']
        # Update reply in the database based on reply_id
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE reply SET reply = %s WHERE id = %s", (reply_content, reply_id))
        mysql.connection.commit()
        cursor.close()
        class_id = request.form.get('class_id')  # Get class_id from form
        flash('Reply Edited', 'success')
        return redirect(url_for('class_content', class_id=class_id))  # Pass class_id to redirect
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add route for deleting replies
@app.route('/delete_reply/<int:reply_id>', methods=['POST'])
@is_logged_in
def delete_reply(reply_id):
    if request.method == 'POST':
        # Delete reply from the database based on reply_id
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM reply WHERE id = %s", [reply_id])
        mysql.connection.commit()
        cursor.close()
        
        class_id = request.json.get('class_id')  # Get class_id from JSON data
        flash('Reply Deleted', 'success')
        return redirect(url_for('class_content', class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add a route for posting comments on images
@app.route('/post_image_comment/<int:image_id>', methods=['POST'])
@is_logged_in
def post_image_comment(image_id):
    if request.method == 'POST':
        comment = request.form['comment']
        user_role = session.get('role')
        user_email = session.get('email')
        class_id = request.form.get('class_id')
        cursor = mysql.connection.cursor()
        if user_role == 'instructor':
            cursor.execute("INSERT INTO image_comment (comment, instructor_email, image_id, class_id) VALUES (%s, %s, %s, %s)",
                           (comment, user_email, image_id, class_id))
        elif user_role == 'student':
            cursor.execute("INSERT INTO image_comment (comment, student_email, image_id, class_id) VALUES (%s, %s, %s, %s)",
                           (comment, user_email, image_id, class_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Posted', 'success')
        # Fetch comments for the image
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM image_comment WHERE image_id = %s", (image_id,))
        image_comments = cursor.fetchall()
        cursor.close()
        return redirect(url_for('class_content', class_id=class_id))  # Redirect to the appropriate route
    return redirect(url_for('class_content', class_id=class_id))  # Redirect to the appropriate route

# Add route for deleting image comments
@app.route('/delete_image_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def delete_image_comment(comment_id):
    if request.method == 'POST':
        # Delete comment from the database based on comment_id
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM image_comment WHERE id = %s", [comment_id])
        mysql.connection.commit()
        cursor.close()
        flash('Comment Deleted', 'success')
        return redirect(url_for('class_content'))  # Redirect to the appropriate route

# Add route for editing comments
@app.route('/edit_image_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def edit_image_comment(comment_id):
    if request.method == 'POST':
        edited_comment = request.json.get('edited_comment')  # Get edited comment from JSON request
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE image_comment SET comment = %s WHERE id = %s", (edited_comment, comment_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Edited', 'success')
        return redirect((request.referrer))  # Redirect back to the previous page
    return redirect(url_for('class_content'))  # Redirect to the appropriate route

# Add a route for posting comments on materials
@app.route('/post_material_comment/<int:material_id>', methods=['POST'])
@is_logged_in
def post_material_comment(material_id):
    if request.method == 'POST':
        comment_text = request.form['comment']  # Retrieve the comment text from the form
        user_role = session.get('role')
        user_email = session.get('email')
        class_id = request.form.get('class_id')  # Retrieve class_id from the form
        cursor = mysql.connection.cursor()
        if user_role == 'instructor':
            cursor.execute("INSERT INTO material_comment (comment, instructor_email, material_id, class_id) VALUES (%s, %s, %s, %s)",
                           (comment_text, user_email, material_id, class_id))
        elif user_role == 'student':
            cursor.execute("INSERT INTO material_comment (comment, student_email, material_id, class_id) VALUES (%s, %s, %s, %s)",
                           (comment_text, user_email, material_id, class_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Posted', 'success')
        # Fetch comments for the material
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM material_comment WHERE material_id = %s", (material_id,))
        image_comments = cursor.fetchall()
        cursor.close()
        return redirect(url_for('class_content', class_id=class_id))
    return redirect(url_for('class_content', class_id=class_id))

# Add route for deleting material comments
@app.route('/delete_material_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def delete_material_comment(comment_id):
    if request.method == 'POST':
        # Delete comment from the database based on comment_id
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM material_comment WHERE id = %s", [comment_id])
        mysql.connection.commit()
        cursor.close()
        flash('Comment Deleted', 'success')
        class_id = request.form.get('class_id')
        return redirect(url_for('class_content',class_id=class_id))  # Redirect to the appropriate route
    return redirect(url_for('class_content'))  # Redirect in case of GET request or any error

# Add route for editing material comments
@app.route('/edit_material_comment/<int:comment_id>', methods=['POST'])
@is_logged_in
def edit_material_comment(comment_id):
    if request.method == 'POST':
        edited_comment = request.json['edited_comment']  # Get edited comment from JSON payload
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE material_comment SET comment = %s WHERE id = %s", (edited_comment, comment_id))
        mysql.connection.commit()
        cursor.close()
        flash('Comment Edited', 'success')
        # Redirect back to the appropriate class content route
        class_id = request.form.get('class_id')
        return redirect(url_for('class_content', class_id=class_id))
    return redirect(url_for('class_content'))  # Redirect to the appropriate route in case of GET request or any error

#Upload Image in Class_Content page
@app.route('/upload_image', methods=['POST', 'GET'])
@is_logged_in
def upload_image():
    if request.method == 'POST':
        # Check if an image was uploaded
        if 'image' in request.files:
            image = request.files['image']
            if image.filename != '':
                # Retrieve class_id from the form or request parameters
                class_id = request.form.get('class_id')
                description = request.form.get('description')
                new_filename = request.form.get('new_filename')  # Retrieve new filename

                if class_id is None:
                    flash('Class ID not provided', 'error')
                    return redirect(url_for('class_content'))

                # Ensure filename has both small and capital letters
                filename = secure_filename(image.filename).lower()

                # Ensure the class directory exists within the UPLOAD_FOLDER
                class_folder = os.path.join(app.config['UPLOAD_FOLDER'], f'class_{class_id}')
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                image_path = os.path.join(class_folder, filename)
                image.save(image_path)

                # Store image information in the database along with class_id
                cursor = mysql.connection.cursor()

                # Use the user's role to determine whether it's an instructor or student uploading the image
                if session['role'] == 'instructor':
                    cursor.execute("INSERT INTO uploaded_images (description, filename, filepath, instructor_email, class_id) VALUES (%s, %s, %s, %s, %s)",
                                   (description, filename, image_path, session['email'], class_id))
                else:
                    cursor.execute("INSERT INTO uploaded_images (description, filename, filepath, student_email, class_id) VALUES (%s, %s, %s, %s, %s)",
                                   (description, filename, image_path, session['email'], class_id))

                mysql.connection.commit()
                cursor.close()

                # Update image filename if new_filename is provided
                if new_filename:
                    cursor = mysql.connection.cursor()
                    cursor.execute("UPDATE uploaded_images SET filename = %s WHERE filepath = %s", (new_filename, filename))
                    mysql.connection.commit()
                    cursor.close()

                flash('Image uploaded successfully', 'success')
                return redirect(url_for('class_content', class_id=class_id))

    # Redirect to class_content route if the request method is not POST or if no image was uploaded
    flash('Image upload failed', 'error')
    return redirect(url_for('class_content'))

#Edit Image
@app.route('/edit_image/<int:image_id>', methods=['POST','GET'])
@is_logged_in
def edit_image(image_id):
    if request.method == 'POST':
        description = request.form['description']
        class_id = request.form.get('class_id')

        # Fetch the image data from the database based on image_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM uploaded_images WHERE id = %s", (image_id,))
        image = cursor.fetchone()

        if 'image' in request.files:
            uploaded_file = request.files['image']
            if uploaded_file.filename != '':
                filename = secure_filename(uploaded_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'class_{class_id}', filename)
                uploaded_file.save(filepath)
                # Delete the old image from the system folder
                if os.path.exists(image['filepath']):
                    os.remove(image['filepath'])
            else:
                filename = request.form.get('new_filename', '')  # Get new filename if provided
                if filename:
                    # Update filepath with new filename
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'class_{class_id}', filename)
                    # Move the file to the new path
                    os.rename(image['filepath'], filepath)
                else:
                    filepath = image['filepath']  # Keep the existing filepath if no new filename provided
        else:
            filename = None
            filepath = None

        # Update image in the database based on image_id
        cursor.execute("UPDATE uploaded_images SET filename = %s, description = %s, filepath = %s, class_id = %s WHERE id = %s",
                       (filename, description, filepath, class_id, image_id))
        mysql.connection.commit()
        cursor.close()
        flash('Image Edited', 'success')
        return redirect(url_for('class_content', class_id=class_id))

    # Fetch the image data from the database based on image_id
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM uploaded_images WHERE id = %s", (image_id,))
    image = cursor.fetchone()
    cursor.close()
    # Pass the image data to the template when rendering
    return render_template('edit_image.html', image=image)

# Add route for deleting images
@app.route('/delete_image/<int:image_id>', methods=['POST'])
@is_logged_in
def delete_image(image_id):
    if request.method == 'POST':
        # Fetch image data from the database based on image_id
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT filename, class_id, filepath FROM uploaded_images WHERE id = %s", (image_id,))
        image_data = cursor.fetchone()
        if image_data:
            image_filename = image_data['filename']
            class_id = image_data['class_id']
            image_filepath = image_data['filepath']
            if class_id is not None:
                # Delete image file from the uploads directory
                if os.path.exists(image_filepath):
                    os.remove(image_filepath)
                else:
                    flash('Image file not found in system folder', 'error')

                # Delete image record from the database
                cursor.execute("DELETE FROM uploaded_images WHERE id = %s", (image_id,))
                mysql.connection.commit()
                cursor.close()
                flash('Image Deleted', 'success')
                return redirect(url_for('class_content', class_id=class_id))
            else:
                flash('Class ID not found', 'error')
    return redirect(url_for('class_content'))


#form builder
@app.route('/formbuilder')
@is_logged_in
def formbuilder():
    return render_template('formbuilder.html')

@app.route('/preview')
@is_logged_in
def preview():
    return render_template('preview.html')

@app.route('/submit_form', methods=['POST'])
@is_logged_in
def submit_form():
    # Handle form submission here
    if request.method == 'POST':
        # Access form data using request.form
        text_field = request.form.get('text_field')
        email_field = request.form.get('email_field')
        # For demonstration purposes, just printing the form data
        print(f'Text Field: {text_field}')
        print(f'Email Field: {email_field}')
        return 'Form submitted successfully!'
    else:
        return 'Invalid request method'
    
@app.route('/classwork/<string:class_id>')
@is_logged_in
def classwork(class_id):
    # Fetch class details
    class_cursor = mysql.connection.cursor()
    class_cursor.execute("SELECT * FROM classes WHERE id=%s", [class_id])
    class_data = class_cursor.fetchone()
    class_cursor.close()
    # Fetch materials based on user role and class_id
    material_cursor = mysql.connection.cursor()
    if session['role'] == 'instructor':
        material_cursor.execute("SELECT *, 'materialc' AS type FROM materialsc WHERE instructor=%s AND class_id=%s ORDER BY create_date DESC", [session['email'], class_id])
    else:  # For students, show materials related to their class
        material_cursor.execute("SELECT *, 'materialc' AS type FROM materialsc WHERE class_id=%s ORDER BY create_date DESC", [class_id])
    materialsc_data = material_cursor.fetchall()
    material_cursor.close()
     # Fetch assignments related to the class
    assignment_cursor = mysql.connection.cursor()
    assignment_cursor.execute("SELECT *, 'assignment' AS type FROM assignments WHERE class_id=%s ORDER BY create_date DESC", [class_id])
    assignment_data = assignment_cursor.fetchall()
    assignment_cursor.close()
    # Fetch questions related to the class
    question_cursor = mysql.connection.cursor()
    question_cursor.execute("SELECT *, 'question' AS type FROM questions WHERE class_id=%s ORDER BY create_date DESC", [class_id])
    question_data =  question_cursor.fetchall()
    question_cursor.close()
    # Fetch uploaded topics related to the class
    topic_cursor = mysql.connection.cursor()
    topic_cursor.execute("SELECT * FROM topic WHERE class_id=%s", [class_id])
    topic_data = topic_cursor.fetchall()
    topic_cursor.close()
    instructor_first_name=session.get('first_name')
    instructor_last_name=session.get('last_name')
    # Combine all data into a single list
    all_posts = materialsc_data + assignment_data + question_data

    return render_template('classwork.html',posts=all_posts, class_data=class_data, materials=materialsc_data, questions=question_data,
                           assignments=assignment_data, topics=topic_data, instructor_first_name=instructor_first_name,
                           instructor_last_name=instructor_last_name,
                           is_student=session['role'] == 'student')

# Add Topic route
@app.route('/add_topic', methods=['GET', 'POST'])
@is_logged_in
def add_topic():
    form = TopicForm(request.form)
    if request.method == 'POST' and form.validate():
        name = form.name.data
        class_id = request.form.get('class_id')  # Get class_id from form
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO topic(name,class_id) VALUES (%s,%s)", (name,class_id))  # Pass both name and class_id
        mysql.connection.commit()
        cursor.close()
        flash('Topic added successfully', 'success')
        return redirect(url_for('classwork',class_id=class_id))
    return render_template('add_topic.html', form=form,class_id=class_id)

# Single Assignment
@app.route('/topic/<string:id>/')
@is_logged_in
def topic(id):
    # Create cursor
    cursor = mysql.connection.cursor()
    # Get Assignment (Single Assignment from database)
    result = cursor.execute("SELECT * FROM topic WHERE id=%s", [id])
    assignment = cursor.fetchone()
    # Close connection
    cursor.close()
    return render_template('topic.html', assignment=assignment, instructor_first_name=session.get('first_name'), instructor_last_name=session.get('last_name') )

# Add route for deleting comments
@app.route('/delete_topic/<int:topic_id>', methods=['POST'])
@is_logged_in
def delete_topic(topic_id):
    if request.method == 'POST':
        global post_positions
        # Delete topic from the database based on topic_id
        cursor = mysql.connection.cursor()
        cursor.execute("DELETE FROM topic WHERE id = %s", [topic_id])
        mysql.connection.commit()
        cursor.close()
        class_id = request.form.get('class_id')  # Get class_id from form
        # Update post positions after deleting the topic
        post_positions = {post_id: t_id for post_id, t_id in post_positions.items() if t_id != topic_id}
        flash('Topic Deleted', 'success')
        return redirect(url_for('classwork', class_id=class_id))
    else:
        return redirect(url_for('classwork', class_id=class_id))  # Redirect in case of GET request or any error

@app.route('/edit_topic/<int:topic_id>', methods=['POST'])
@is_logged_in
def edit_topic(topic_id):
    if request.method == 'POST':
        topic_name = request.form['topic_name']
        class_id = request.form.get('class_id')
        # Update the topic name in the database
        cursor = mysql.connection.cursor()
        cursor.execute("UPDATE topic SET name = %s WHERE id = %s", (topic_name, topic_id))
        mysql.connection.commit()
        cursor.close()
        flash('Topic Edited', 'success')
        return redirect(url_for('classwork', class_id=class_id))
    else:
        # Redirect to the classwork page in case of GET request or any error
        return redirect(url_for('classwork', class_id=class_id))
    

# Dummy data to simulate post positions
post_positions = {}
@app.route('/update_post_position', methods=['POST'])
@is_logged_in
def update_post_position():
    if request.method == 'POST':
        post_id = request.form.get('post_id')
        topic_id = request.form.get('topic_id')
        # Update post position in the dictionary
        post_positions[post_id] = topic_id
        return jsonify({'message': 'Post position updated successfully'}), 200

@app.route('/get_post_positions', methods=['GET'])
@is_logged_in
def get_post_positions():
    return jsonify(post_positions)
# Enroll Route
@app.route('/enroll', methods=['GET', 'POST'])
@is_logged_in  # Require user to be logged in
def enroll():
    if request.method == 'POST':
        if session.get('role') == 'student':  # Safely access session data
            classcode = request.form.get('classcode')  # Safely get the class code from the form
            student_email = session.get('email')  # Safely get the email of the logged-in student
            if not classcode or not student_email:  # Check if classcode or student_email is missing
                flash("Incomplete enrollment data", "danger")
                return redirect(url_for('enroll'))
            cursor = mysql.connection.cursor()
            # Check if the class code exists in the classes table and is not archived
            cursor.execute("SELECT id, classname, classsection, instructor_email, banners_filename \
                           FROM classes WHERE classcode=%s AND archived = 0", [classcode])
            class_data = cursor.fetchone()
            if class_data:
                # Check if the student is already enrolled in this class
                cursor.execute("SELECT * FROM enrollments WHERE class_id = %s AND student_email = %s", (class_data['id'], student_email))
                already_enrolled = cursor.fetchone()
                if already_enrolled:
                    flash("You are already enrolled in this class", "warning")
                    cursor.close()
                    return redirect(url_for('enroll'))
                # Insert enrollment data into the enrollments table
                cursor.execute("INSERT INTO enrollments (class_id, classcode, classsection, student_email) VALUES (%s, %s, %s, %s)", (class_data['id'], classcode, class_data['classsection'], student_email))
                mysql.connection.commit()
                flash("Enrolled in class successfully!", "success")
                cursor.close()
                return redirect(url_for('enroll'))
            else:
                flash("Invalid class code or the class is archived", "danger")
                cursor.close()
                return redirect(url_for('classes'))
        else:
            flash("Only students can enroll in classes", "danger")
            return redirect(url_for('enroll'))
    else:
        # Handle GET request to show the enrollment form or information
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT classes.id, classes.classcode, classes.classname, classes.classsection, instructor.first_name AS instructor_first_name, instructor.last_name AS instructor_last_name, classes.banners_filename FROM classes JOIN enrollments ON classes.id = enrollments.class_id JOIN instructor ON classes.instructor_email = instructor.email WHERE enrollments.student_email = %s AND classes.archived = 0", (session.get('email'),))
        enrolled_classes = cursor.fetchall()

        # Fetch active classes taught by the logged-in instructor
        cursor.execute("SELECT * FROM classes WHERE archived = 0 AND instructor_email = %s", (session['email'],))
        active_classes = cursor.fetchall()
        cursor.close()

        return render_template('enroll.html', enrolled_classes=enrolled_classes, active_classes=active_classes)

# Unenrol Class Route
@app.route('/unenroll/<string:id>', methods=['GET', 'POST'])
@is_logged_in  # Require user to be logged in
def unenroll(id):
    cursor = mysql.connection.cursor()
            
    # Check if the logged-in user is a student
    if session.get('role') == 'student':
        student_email = session.get('email')  # Safely get the email of the logged-in student
        
        # Check if the student is enrolled in the class
        cursor.execute("SELECT * FROM enrollments WHERE class_id=%s AND student_email=%s", (id, student_email))
        enrollment_data = cursor.fetchone()
        
        if not enrollment_data:
            flash("You are not enrolled in this class", "warning")
            cursor.close()
            return redirect(url_for('classes'))
        
        # Delete the enrollment record
        cursor.execute("DELETE FROM enrollments WHERE class_id=%s AND student_email=%s", (id, student_email))
        mysql.connection.commit()
        flash("Unenrolled from class successfully!", "success")
        cursor.close()
        return redirect(url_for('enroll'))
    
    else:
        flash("Only students can unenroll from classes", "danger")
        cursor.close()
        return redirect(url_for('classes'))

# Function to get list of image filenames in the banners directory
def get_banner_images():
    banner_dir = os.path.join(app.static_folder, 'banners')
    print("Banner directory:", banner_dir)  # Debugging output
    if os.path.exists(banner_dir):
        images = [f for f in os.listdir(banner_dir) if os.path.isfile(os.path.join(banner_dir, f))]
        print("Images found:", images)  # Debugging output
        return images
    else:
        print("Banner directory does not exist")  # Debugging output
        return []
    
# Route to serve the banners modal with images
@app.route('/banners')
@is_logged_in
def banners():
    # Get list of image filenames
    images = get_banner_images()
    # Print URLs for debugging
    return jsonify(images)

if __name__ == '__main__':
    #start_pdf_monitoring()
    app.secret_key = 'YasAmar'
    app.run(debug=True)
