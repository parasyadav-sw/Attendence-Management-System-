import os
import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('ParaScan')
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'attendance-system-secret-key-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///attendance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
FACES_DIR = os.path.join('static', 'faces')
os.makedirs(FACES_DIR, exist_ok=True)


class Admin(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)


class Faculty(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    faculty_id = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100))
    email = db.Column(db.String(120))   
    mobile = db.Column(db.String(20))
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    is_active_account = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Classroom(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    division = db.Column(db.String(20))
    department = db.Column(db.String(100))
    academic_year = db.Column(db.String(20))
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    faculty = db.relationship('Faculty', backref=db.backref('classrooms', lazy=True))
    students = db.relationship('Student', backref='classroom', lazy=True)


class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    student_id = db.Column(db.String(50), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    division = db.Column(db.String(20))
    face_image = db.Column(db.String(200))
    face_histogram = db.Column(db.Text)
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'))
    classroom_id = db.Column(db.Integer, db.ForeignKey('classroom.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    faculty = db.relationship('Faculty', backref=db.backref('students', lazy=True))


class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    status = db.Column(db.String(20), default='Present')
    faculty_id = db.Column(db.Integer, db.ForeignKey('faculty.id'))
    classroom_id = db.Column(db.Integer, db.ForeignKey('classroom.id'))
    student = db.relationship('Student', backref=db.backref('attendances', lazy=True))
    faculty = db.relationship('Faculty', backref=db.backref('attendances', lazy=True))
    classroom = db.relationship('Classroom', backref=db.backref('attendances', lazy=True))


class UserSession:
    ROLE_ADMIN = 'admin'
    ROLE_FACULTY = 'faculty'


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or session.get('user_role') != UserSession.ROLE_ADMIN:
            flash('Access denied. Admin login required.', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function


def faculty_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or session.get('user_role') != UserSession.ROLE_FACULTY:
            flash('Access denied. Faculty login required.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@login_manager.user_loader
def load_user(user_id):
    user_id = int(user_id)
    admin = Admin.query.get(user_id)
    if admin:
        return admin
    return Faculty.query.get(user_id)


def compute_histogram(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.flatten().tolist()


def compare_faces(hist1_list, hist2_list, threshold=0.6):
    if hist1_list is None or hist2_list is None:
        return False, 0.0
    hist1 = np.array(hist1_list, dtype=np.float32).flatten()
    hist2 = np.array(hist2_list, dtype=np.float32).flatten()
    if hist1.shape != hist2.shape:
        logger.warning(f"compare_faces: shape mismatch {hist1.shape} vs {hist2.shape}")
        return False, 0.0
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score > threshold, score


def flatten_histogram(hist_list):
    result = []
    def _flatten(item):
        if isinstance(item, list):
            for sub in item:
                _flatten(sub)
        else:
            result.append(item)
    _flatten(hist_list)
    return result


def match_student(face_img, threshold=0.55, classroom_id=None):
    query = Student.query.filter(Student.face_histogram.isnot(None))
    if classroom_id:
        query = query.filter_by(classroom_id=classroom_id)
    students = query.all()
    logger.debug(f"match_student: {len(students)} registered student(s), classroom_id={classroom_id}")
    if not students:
        return None, 0
    try:
        current_hist = compute_histogram(face_img)
    except Exception as e:
        logger.error(f"match_student: histogram computation failed: {e}")
        return None, 0
    best_match = None
    best_score = -1
    for student in students:
        try:
            hist_list = json.loads(student.face_histogram)
            hist_flat = flatten_histogram(hist_list)
            match, score = compare_faces(current_hist, hist_flat, threshold)
            logger.debug(f"match_student: {student.name} score={score:.4f} match={match}")
            if score > best_score:
                best_score = score
                if match:
                    best_match = student
        except Exception as e:
            logger.error(f"match_student: compare failed for {student.name}: {e}")
            continue
    logger.debug(f"match_student: best={best_match.name if best_match else 'None'} score={best_score:.4f}")
    return best_match, best_score


@app.route('/')
def index():
    if current_user.is_authenticated:
        if session.get('user_role') == UserSession.ROLE_ADMIN:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        faculty = Faculty.query.filter_by(username=username).first()
        if faculty and check_password_hash(faculty.password, password):
            if not faculty.is_active_account:
                flash('Your account has been deactivated. Contact admin.', 'error')
                return redirect(url_for('login'))
            login_user(faculty)
            session['user_role'] = UserSession.ROLE_FACULTY
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/dashboard')
@faculty_required
def dashboard():
    faculty_id = current_user.id
    classes = Classroom.query.filter_by(faculty_id=faculty_id).all()
    total_students = Student.query.filter_by(faculty_id=faculty_id).count()
    today = datetime.now().date()
    today_attendance = db.session.query(Attendance).join(Student).filter(
        Student.faculty_id == faculty_id, Attendance.date == today
    ).count()
    absent_students = total_students - today_attendance
    if absent_students < 0:
        absent_students = 0
    recent_students = Student.query.filter_by(faculty_id=faculty_id).order_by(Student.created_at.desc()).limit(5).all()
    recent_attendance = db.session.query(Attendance, Student).join(Student).filter(
        Student.faculty_id == faculty_id
    ).order_by(Attendance.id.desc()).limit(5).all()
    return render_template('dashboard.html',
                           total_students=total_students,
                           today_attendance=today_attendance,
                           absent_students=absent_students,
                           recent_students=recent_students,
                           recent_attendance=recent_attendance,
                           classes=classes,
                           current_date=datetime.now().strftime('%A, %B %d, %Y'),
                           current_time=datetime.now().strftime('%I:%M %p'))


# ==================== CLASS MANAGEMENT ====================

@app.route('/classes')
@faculty_required
def class_list():
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.created_at.desc()).all()
    return render_template('class_list.html', classes=classes)


@app.route('/classes/add', methods=['GET', 'POST'])
@faculty_required
def class_add():
    if request.method == 'POST':
        name = request.form.get('name')
        division = request.form.get('division')
        department = request.form.get('department')
        academic_year = request.form.get('academic_year')
        if not name:
            flash('Class name is required!', 'error')
            return redirect(url_for('class_add'))
        classroom = Classroom(
            name=name,
            division=division,
            department=department,
            academic_year=academic_year,
            faculty_id=current_user.id
        )
        db.session.add(classroom)
        db.session.commit()
        flash('Class created successfully!', 'success')
        return redirect(url_for('class_list'))
    return render_template('class_add.html')


@app.route('/classes/edit/<int:class_id>', methods=['GET', 'POST'])
@faculty_required
def class_edit(class_id):
    classroom = Classroom.query.get_or_404(class_id)
    if classroom.faculty_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('class_list'))
    if request.method == 'POST':
        classroom.name = request.form.get('name')
        classroom.division = request.form.get('division')
        classroom.department = request.form.get('department')
        classroom.academic_year = request.form.get('academic_year')
        db.session.commit()
        flash('Class updated successfully!', 'success')
        return redirect(url_for('class_list'))
    return render_template('class_edit.html', classroom=classroom)


@app.route('/classes/delete/<int:class_id>', methods=['POST'])
@faculty_required
def class_delete(class_id):
    classroom = Classroom.query.get_or_404(class_id)
    if classroom.faculty_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('class_list'))
    Student.query.filter_by(classroom_id=class_id).update({'classroom_id': None})
    db.session.delete(classroom)
    db.session.commit()
    flash('Class deleted successfully!', 'success')
    return redirect(url_for('class_list'))


@app.route('/get-class-students/<int:class_id>')
@faculty_required
def get_class_students(class_id):
    classroom = Classroom.query.get_or_404(class_id)
    if classroom.faculty_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    students = Student.query.filter_by(classroom_id=class_id).all()
    data = [{'id': s.id, 'name': s.name, 'student_id': s.student_id, 'has_face': bool(s.face_histogram)} for s in students]
    return jsonify(data)


# ==================== STUDENT REGISTRATION ====================

@app.route('/register-student', methods=['GET', 'POST'])
@faculty_required
def register_student():
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.name).all()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        student_id = request.form.get('student_id', '').strip()
        class_name = request.form.get('class_name')
        division = request.form.get('division')
        classroom_id = request.form.get('classroom_id')
        logger.info(f"register_student: checking student_id='{student_id}' for faculty={current_user.id}")
        existing = Student.query.filter_by(student_id=student_id, faculty_id=current_user.id).first()
        if existing:
            logger.warning(f"register_student: DUPLICATE student_id='{student_id}' already exists for this faculty (db_id={existing.id})")
            flash('Student ID already exists!', 'error')
            return redirect(url_for('register_student'))
        student = Student(
            name=name,
            student_id=student_id,
            class_name=class_name,
            division=division,
            classroom_id=int(classroom_id) if classroom_id else None,
            faculty_id=current_user.id
        )
        db.session.add(student)
        db.session.commit()
        logger.info(f"register_student: created student '{name}' id='{student_id}' db_id={student.id}")
        flash('Student registered! Now capture the face.', 'success')
        return redirect(url_for('capture_face', student_id=student.id))
    return render_template('register_student.html', classes=classes)


@app.route('/capture-face/<int:student_id>')
@faculty_required
def capture_face(student_id):
    student = Student.query.get_or_404(student_id)
    if student.faculty_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('students'))
    return render_template('capture_face.html', student=student)


@app.route('/save-face', methods=['POST'])
@faculty_required
def save_face():
    student_id = request.form.get('student_id')
    face_data = request.form.get('face_data')
    student = Student.query.get(int(student_id))
    if not student or student.faculty_id != current_user.id:
        return jsonify({'error': 'Student not found'}), 404
    try:
        img_data = base64.b64decode(face_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            face_filename = f"student_{student.id}.jpg"
            face_path = os.path.join(FACES_DIR, face_filename)
            cv2.imwrite(face_path, face_roi)
            student.face_image = face_filename
            histogram = compute_histogram(face_roi)
            student.face_histogram = json.dumps(histogram)
            db.session.commit()
            return jsonify({'success': True, 'message': 'Face captured successfully!'})
        else:
            return jsonify({'error': 'No face detected. Please try again.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/debug/face-data')
@faculty_required
def debug_face_data():
    students = Student.query.filter_by(faculty_id=current_user.id).all()
    result = []
    for s in students:
        has_hist = s.face_histogram is not None
        hist_shape = None
        if has_hist:
            try:
                h = json.loads(s.face_histogram)
                h_flat = flatten_histogram(h)
                hist_shape = f"raw={type(h).__name__} flat_len={len(h_flat)}"
            except Exception as e:
                hist_shape = f"ERROR: {e}"
        result.append({
            'name': s.name,
            'student_id': s.student_id,
            'classroom_id': s.classroom_id,
            'has_face_image': s.face_image is not None,
            'has_histogram': has_hist,
            'histogram_info': hist_shape,
            'face_file_exists': os.path.exists(os.path.join(FACES_DIR, s.face_image)) if s.face_image else False
        })
    return jsonify(result)# ==================== ATTENDANCE ====================

@app.route('/attendance')
@faculty_required
def attendance():
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.name).all()
    return render_template('attendance.html', classes=classes)


@app.route('/students')
@faculty_required
def students():
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.name).all()
    class_id = request.args.get('class_id', '')
    query = Student.query.filter_by(faculty_id=current_user.id)
    if class_id:
        query = query.filter_by(classroom_id=int(class_id))
    all_students = query.order_by(Student.created_at.desc()).all()
    return render_template('students.html', students=all_students, classes=classes, selected_class_id=class_id)


@app.route('/student-edit/<int:student_id>', methods=['GET', 'POST'])
@faculty_required
def student_edit(student_id):
    student = Student.query.get_or_404(student_id)
    if student.faculty_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('students'))
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.name).all()
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        student_id_val = request.form.get('student_id', '').strip()
        classroom_id = request.form.get('classroom_id')
        if not name:
            flash('Student name is required!', 'error')
            return redirect(url_for('student_edit', student_id=student.id))
        if not student_id_val:
            flash('Student ID is required!', 'error')
            return redirect(url_for('student_edit', student_id=student.id))
        existing = Student.query.filter(Student.student_id == student_id_val, Student.faculty_id == current_user.id, Student.id != student.id).first()
        if existing:
            flash('Student ID already exists!', 'error')
            return redirect(url_for('student_edit', student_id=student.id))
        classroom = Classroom.query.get(int(classroom_id)) if classroom_id else None
        student.name = name
        student.student_id = student_id_val
        student.class_name = classroom.name if classroom else request.form.get('class_name', '')
        student.division = classroom.division if classroom else request.form.get('division', '')
        student.classroom_id = int(classroom_id) if classroom_id else None
        db.session.commit()
        flash('Student updated successfully!', 'success')
        return redirect(url_for('students'))
    return render_template('student_edit.html', student=student, classes=classes)


@app.route('/student-delete/<int:student_id>', methods=['POST'])
@faculty_required
def student_delete(student_id):
    student = Student.query.get_or_404(student_id)
    if student.faculty_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('students'))
    name = student.name
    sid = student.student_id
    logger.info(f"student_delete: deleting student '{name}' (id={sid}, db_id={student.id})")
    if student.face_image:
        face_path = os.path.join(FACES_DIR, student.face_image)
        if os.path.exists(face_path):
            os.remove(face_path)
            logger.info(f"student_delete: removed face image {face_path}")
    deleted_att = Attendance.query.filter_by(student_id=student.id).delete()
    logger.info(f"student_delete: deleted {deleted_att} attendance record(s)")
    db.session.delete(student)
    db.session.commit()
    verify = Student.query.filter_by(student_id=sid).first()
    if verify:
        logger.error(f"student_delete: FAILED — student_id='{sid}' still exists after commit!")
    else:
        logger.info(f"student_delete: confirmed student_id='{sid}' removed from database")
    flash(f'Student "{name}" deleted successfully!', 'success')
    return redirect(url_for('students'))


@app.route('/video-feed')
def video_feed():
    def generate_frames():
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_roi = frame[y:y+h, x:x+w]
                student, score = match_student(face_roi)
                name = student.name if student else "Unknown"
                if student:
                    today = datetime.now().date()
                    now = datetime.now().time()
                    existing = Attendance.query.filter_by(student_id=student.id, date=today).first()
                    if not existing:
                        att = Attendance(student_id=student.id, date=today, time=now, status='Present', faculty_id=student.faculty_id, classroom_id=student.classroom_id)
                        db.session.add(att)
                        db.session.commit()
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), color, cv2.FILLED)
                cv2.putText(frame, name, (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/attendance-records')
@faculty_required
def attendance_records():
    date = request.args.get('date', datetime.now().date().isoformat())
    class_id = request.args.get('class_id', '')
    selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    query = db.session.query(Attendance, Student).join(Student).filter(
        Student.faculty_id == current_user.id, Attendance.date == selected_date
    )
    if class_id:
        query = query.filter(Student.classroom_id == int(class_id))
    records = query.all()
    classes = Classroom.query.filter_by(faculty_id=current_user.id).order_by(Classroom.name).all()
    return render_template('attendance_records.html', records=records, selected_date=selected_date, classes=classes, selected_class_id=class_id)


@app.route('/get-attendance-data')
@faculty_required
def get_attendance_data():
    date = request.args.get('date', datetime.now().date().isoformat())
    classroom_id = request.args.get('classroom_id', '')
    selected_date = datetime.strptime(date, '%Y-%m-%d').date()
    query = db.session.query(Attendance, Student).join(Student).filter(
        Student.faculty_id == current_user.id, Attendance.date == selected_date
    )
    if classroom_id:
        query = query.filter(Student.classroom_id == int(classroom_id))
    records = query.all()
    data = [{'name': s.name, 'student_id': s.student_id, 'time': a.time.strftime('%H:%M:%S'), 'status': a.status} for a, s in records]
    return jsonify(data)


@app.route('/mark-attendance', methods=['POST'])
@faculty_required
def mark_attendance():
    data = request.get_json()
    student_id = data.get('student_id')
    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400
    student = Student.query.get(int(student_id))
    if not student or student.faculty_id != current_user.id:
        return jsonify({'error': 'Student not found'}), 404
    today = datetime.now().date()
    now = datetime.now().time()
    existing = Attendance.query.filter_by(student_id=student.id, date=today).first()
    if existing:
        return jsonify({'message': 'Attendance already marked', 'student_name': student.name})
    att = Attendance(student_id=student.id, date=today, time=now, status='Present', faculty_id=current_user.id, classroom_id=student.classroom_id)
    db.session.add(att)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Attendance marked for {student.name}', 'student_name': student.name})


@app.route('/detect-face', methods=['POST'])
@faculty_required
def detect_face():
    data = request.get_json()
    image_data = data.get('image')
    classroom_id = data.get('classroom_id')
    client_width = data.get('clientWidth', 640)
    client_height = data.get('clientHeight', 480)
    if not image_data:
        logger.warning("detect_face: no image data received")
        return jsonify({'error': 'No image data'}), 400
    try:
        img_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("detect_face: cv2.imdecode returned None")
            return jsonify({'success': False, 'message': 'Failed to decode image', 'faces': []})
        logger.debug(f"detect_face: decoded image shape={img.shape}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_boxes = []
        for (fx, fy, fw, fh) in faces:
            face_boxes.append({
                'x': round(float(fx) / img.shape[1] * client_width),
                'y': round(float(fy) / img.shape[0] * client_height),
                'w': round(float(fw) / img.shape[1] * client_width),
                'h': round(float(fh) / img.shape[0] * client_height)
            })
        logger.info(f"detect_face: image={img.shape}, faces={len(faces)}, classroom_id={classroom_id}")
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            cid = int(classroom_id) if classroom_id else None
            student, score = match_student(face_roi, classroom_id=cid)
            if student is None:
                logger.info(f"detect_face: no match (best_score={score:.4f})")
                return jsonify({'success': False, 'message': 'Student not registered', 'score': round(score, 4), 'faces': face_boxes})
            if student.faculty_id != current_user.id:
                logger.warning(f"detect_face: faculty_id mismatch for {student.name}")
                return jsonify({'success': False, 'message': 'Student not registered', 'faces': face_boxes})
            today = datetime.now().date()
            now = datetime.now().time()
            existing = Attendance.query.filter_by(student_id=student.id, date=today).first()
            if existing:
                logger.info(f"detect_face: already marked for {student.name}")
                return jsonify({'success': True, 'student_name': student.name, 'student_id': student.student_id, 'marked': False, 'message': 'Attendance already marked', 'faces': face_boxes, 'score': round(score, 4)})
            att = Attendance(student_id=student.id, date=today, time=now, status='Present', faculty_id=current_user.id, classroom_id=cid)
            db.session.add(att)
            db.session.commit()
            logger.info(f"detect_face: MARKED {student.name} at {now.strftime('%H:%M:%S')} (score={score:.4f})")
            return jsonify({'success': True, 'student_name': student.name, 'student_id': student.student_id, 'time': now.strftime('%H:%M:%S'), 'marked': True, 'faces': face_boxes, 'score': round(score, 4)})
        logger.debug("detect_face: no faces in frame")
        return jsonify({'success': False, 'message': 'No face detected', 'faces': face_boxes})
    except Exception as e:
        logger.error(f"detect_face error: {e}", exc_info=True)
        return jsonify({'error': str(e), 'faces': []}), 500


@app.route('/detect-face-batch', methods=['POST'])
@faculty_required
def detect_face_batch():
    data = request.get_json()
    images = data.get('images', [])
    classroom_id = data.get('classroom_id')
    client_width = data.get('clientWidth', 640)
    client_height = data.get('clientHeight', 480)
    logger.info(f"[BATCH] Received {len(images)} frames, classroom_id={classroom_id}")
    if not images:
        return jsonify({'error': 'No images provided'}), 400
    cid = int(classroom_id) if classroom_id else None
    query = Student.query.filter(Student.face_histogram.isnot(None))
    if cid:
        query = query.filter_by(classroom_id=cid)
    students = query.all()
    logger.info(f"[BATCH] {len(students)} registered student(s) in classroom")
    if not students:
        return jsonify({'success': False, 'message': 'No registered students', 'faces': [], 'debug': {'framesReceived': len(images), 'registeredStudents': 0}})
    all_face_histograms = []
    face_box = None
    for idx, img_data in enumerate(images):
        try:
            clean = img_data.split(',')[1] if ',' in img_data else img_data
            img_bytes = base64.b64decode(clean)
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning(f"[BATCH] Frame {idx}: failed to decode")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            logger.debug(f"[BATCH] Frame {idx}: shape={img.shape}, faces={len(faces)}")
            for (fx, fy, fw, fh) in faces:
                face_roi = img[fy:fy+fh, fx:fx+fw]
                face_roi = cv2.resize(face_roi, (200, 200))
                hist = compute_histogram(face_roi)
                all_face_histograms.append(hist)
                if face_box is None:
                    face_box = {'x': round(float(fx) / img.shape[1] * client_width), 'y': round(float(fy) / img.shape[0] * client_height), 'w': round(float(fw) / img.shape[1] * client_width), 'h': round(float(fh) / img.shape[0] * client_height)}
        except Exception as e:
            logger.error(f"[BATCH] Frame {idx} error: {e}")
    logger.info(f"[BATCH] Extracted {len(all_face_histograms)} face histogram(s) from {len(images)} frame(s)")
    if not all_face_histograms:
        return jsonify({'success': False, 'message': 'No face detected', 'faces': [], 'debug': {'framesReceived': len(images), 'facesFound': 0, 'registeredStudents': len(students), 'reason': 'no_face_in_any_frame'}})
    best_student = None
    best_score = -1
    scores = {}
    for student in students:
        try:
            reg_hist = flatten_histogram(json.loads(student.face_histogram))
            student_best = -1
            for face_hist in all_face_histograms:
                _, score = compare_faces(face_hist, reg_hist, threshold=0.55)
                if score > student_best:
                    student_best = score
            scores[student.name] = round(student_best, 4)
            logger.debug(f"[BATCH] {student.name}: best_score={student_best:.4f}")
            if student_best > best_score:
                best_score = student_best
                best_student = student
        except Exception as e:
            logger.error(f"[BATCH] Match error for {student.name}: {e}")
    logger.info(f"[BATCH] Best match: {best_student.name if best_student else 'None'}, score={best_score:.4f}, all_scores={scores}")
    faces_out = [face_box] if face_box else []
    if best_student is None or best_score < 0.55:
        reason = 'score_below_threshold' if best_score >= 0 else 'no_match'
        logger.info(f"[BATCH] FAILED: reason={reason}, best_score={best_score:.4f}")
        return jsonify({'success': False, 'message': 'Student not registered', 'faces': faces_out, 'score': round(max(best_score, 0), 4), 'debug': {'framesReceived': len(images), 'facesFound': len(all_face_histograms), 'threshold': 0.55, 'bestScore': round(max(best_score, 0), 4), 'scores': scores, 'reason': reason}})
    today = datetime.now().date()
    now = datetime.now().time()
    existing = Attendance.query.filter_by(student_id=best_student.id, date=today).first()
    if existing:
        logger.info(f"[BATCH] {best_student.name} already marked at {existing.time}")
        return jsonify({'success': True, 'student_name': best_student.name, 'student_id': best_student.student_id, 'marked': False, 'message': 'Already marked today', 'faces': faces_out, 'score': round(best_score, 4), 'debug': {'framesReceived': len(images), 'facesFound': len(all_face_histograms), 'matchedStudent': best_student.name, 'bestScore': round(best_score, 4), 'threshold': 0.55, 'scores': scores, 'reason': 'duplicate'}})
    att = Attendance(student_id=best_student.id, date=today, time=now, status='Present', faculty_id=current_user.id, classroom_id=cid)
    db.session.add(att)
    db.session.commit()
    logger.info(f"[BATCH] MARKED {best_student.name} at {now.strftime('%H:%M:%S')} score={best_score:.4f}")
    return jsonify({'success': True, 'student_name': best_student.name, 'student_id': best_student.student_id, 'time': now.strftime('%H:%M:%S'), 'marked': True, 'faces': faces_out, 'score': round(best_score, 4), 'debug': {'framesReceived': len(images), 'facesFound': len(all_face_histograms), 'matchedStudent': best_student.name, 'bestScore': round(best_score, 4), 'threshold': 0.55, 'scores': scores, 'reason': 'marked'}})


@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('user_role', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ==================== ADMIN ROUTES ====================

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated and session.get('user_role') == UserSession.ROLE_ADMIN:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            login_user(admin)
            session['user_role'] = UserSession.ROLE_ADMIN
            flash('Admin login successful!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Invalid admin credentials', 'error')
    return render_template('admin_login.html')


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    total_faculty = Faculty.query.count()
    total_students = Student.query.count()
    total_classes = Classroom.query.count()
    today = datetime.now().date()
    today_attendance = Attendance.query.filter_by(date=today).count()
    total_attendance = Attendance.query.count()
    return render_template('admin_dashboard.html',
                           total_faculty=total_faculty,
                           total_students=total_students,
                           total_classes=total_classes,
                           today_attendance=today_attendance,
                           total_attendance=total_attendance,
                           current_date=datetime.now().strftime('%A, %B %d, %Y'),
                           current_time=datetime.now().strftime('%I:%M %p'))


@app.route('/admin/faculty')
@admin_required
def admin_faculty_list():
    all_faculty = Faculty.query.order_by(Faculty.created_at.desc()).all()
    return render_template('admin_faculty_list.html', faculty_list=all_faculty)


@app.route('/admin/faculty/add', methods=['GET', 'POST'])
@admin_required
def admin_faculty_add():
    if request.method == 'POST':
        name = request.form.get('name')
        faculty_id = request.form.get('faculty_id')
        department = request.form.get('department')
        email = request.form.get('email')
        mobile = request.form.get('mobile')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not all([name, faculty_id, username, password, confirm_password]):
            flash('Please fill all required fields.', 'error')
            return redirect(url_for('admin_faculty_add'))

        if password != confirm_password:
            flash('Passwords do not match!', 'error')
            return redirect(url_for('admin_faculty_add'))

        if Faculty.query.filter_by(username=username).first():
            flash('Username already exists!', 'error')
            return redirect(url_for('admin_faculty_add'))

        if Faculty.query.filter_by(faculty_id=faculty_id).first():
            flash('Faculty ID already exists!', 'error')
            return redirect(url_for('admin_faculty_add'))

        new_faculty = Faculty(
            name=name, faculty_id=faculty_id, department=department,
            email=email, mobile=mobile, username=username,
            password=generate_password_hash(password)
        )
        db.session.add(new_faculty)
        db.session.commit()
        flash('Faculty account created successfully!', 'success')
        return redirect(url_for('admin_faculty_list'))

    return render_template('admin_faculty_add.html')


@app.route('/admin/faculty/edit/<int:faculty_id>', methods=['GET', 'POST'])
@admin_required
def admin_faculty_edit(faculty_id):
    faculty = Faculty.query.get_or_404(faculty_id)
    if request.method == 'POST':
        faculty.name = request.form.get('name')
        faculty.faculty_id = request.form.get('faculty_id')
        faculty.department = request.form.get('department')
        faculty.email = request.form.get('email')
        faculty.mobile = request.form.get('mobile')
        faculty.username = request.form.get('username')
        faculty.is_active_account = request.form.get('is_active') == 'on'

        new_password = request.form.get('password')
        if new_password:
            faculty.password = generate_password_hash(new_password)

        db.session.commit()
        flash('Faculty updated successfully!', 'success')
        return redirect(url_for('admin_faculty_list'))

    return render_template('admin_faculty_edit.html', faculty=faculty)


@app.route('/admin/faculty/delete/<int:faculty_id>', methods=['POST'])
@admin_required
def admin_faculty_delete(faculty_id):
    faculty = Faculty.query.get_or_404(faculty_id)
    db.session.delete(faculty)
    db.session.commit()
    flash('Faculty deleted successfully!', 'success')
    return redirect(url_for('admin_faculty_list'))


@app.route('/admin/faculty/toggle/<int:faculty_id>', methods=['POST'])
@admin_required
def admin_faculty_toggle(faculty_id):
    faculty = Faculty.query.get_or_404(faculty_id)
    faculty.is_active_account = not faculty.is_active_account
    db.session.commit()
    status = 'activated' if faculty.is_active_account else 'deactivated'
    flash(f'Faculty {status} successfully!', 'success')
    return redirect(url_for('admin_faculty_list'))


@app.route('/admin/students')
@admin_required
def admin_students():
    all_students = db.session.query(Student, Faculty).join(Faculty, Student.faculty_id == Faculty.id).order_by(Student.created_at.desc()).all()
    return render_template('admin_students.html', student_list=all_students)


@app.route('/admin/attendance')
@admin_required
def admin_attendance():
    date = request.args.get('date', datetime.now().date().isoformat())
    faculty_filter = request.args.get('faculty_id', '')
    class_filter = request.args.get('class_id', '')
    selected_date = datetime.strptime(date, '%Y-%m-%d').date()

    query = db.session.query(Attendance, Student, Faculty).join(Student, Attendance.student_id == Student.id).join(Faculty, Attendance.faculty_id == Faculty.id).filter(Attendance.date == selected_date)

    if faculty_filter:
        query = query.filter(Attendance.faculty_id == int(faculty_filter))
    if class_filter:
        query = query.filter(Student.classroom_id == int(class_filter))

    records = query.all()
    all_faculty = Faculty.query.order_by(Faculty.name).all()
    all_classes = Classroom.query.order_by(Classroom.name).all()
    return render_template('admin_attendance.html', records=records, selected_date=selected_date, all_faculty=all_faculty, faculty_filter=faculty_filter, all_classes=all_classes, class_filter=class_filter)


@app.route('/admin/logout')
@admin_required
def admin_logout():
    logout_user()
    session.pop('user_role', None)
    flash('Admin logged out.', 'info')
    return redirect(url_for('admin_login'))


with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username='admin').first():
        default_admin = Admin(username='admin', password=generate_password_hash('admin123'), name='System Administrator')
        db.session.add(default_admin)
        db.session.commit()
    if not Faculty.query.filter((Faculty.username=='faculty') | (Faculty.faculty_id=='FAC001')).first():
        default_faculty = Faculty(name='Admin Faculty', faculty_id='FAC001', department='General', username='faculty', password=generate_password_hash('faculty123'))
        db.session.add(default_faculty)
        db.session.commit()

if __name__ == '__main__':
    print("\n========================================")
    print("  ParaScan - Face Recognition Attendance")
    print("  Starting server...")
    print("  Open browser: http://localhost:5000")
    print("  Admin: admin / admin123")
    print("  Faculty: faculty / faculty123")
    print("========================================\n")
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
