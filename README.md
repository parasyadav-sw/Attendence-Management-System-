# Attendance Management System Using Face Recognition

A responsive web application for schools and colleges to automate attendance using face recognition technology.

## Features

- **Faculty Login** - Secure authentication for faculty members
- **Student Registration** - Add students with personal information
- **Face Capture** - Capture and store facial data during registration
- **Face Recognition Attendance** - Automatic attendance marking using face recognition
- **Attendance Records** - View and manage attendance history
- **Responsive Design** - Works on desktop and laptop browsers

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam/Camera for face capture
- Modern web browser (Chrome, Firefox, Edge)

## Installation

1. **Clone or download the project**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python app.py
   ```

4. **Access the application:**
   Open your browser and go to: `http://localhost:5000`

## Default Login Credentials

- **Username:** faculty
- **Password:** faculty123

## Usage Guide

### 1. Faculty Login
- Open the application in your browser
- Enter the faculty credentials
- Click "Login"

### 2. Student Registration
- Click "Register Student" from the dashboard
- Fill in the student information:
  - Student Name
  - Student ID / Roll Number
  - Class
  - Division (optional)
- Click "Continue to Face Capture"
- Position the student's face in front of the camera
- Click "Capture Face" to save the facial data

### 3. Taking Attendance
- Click "Take Attendance" from the dashboard
- The camera will start automatically
- Students stand in front of the camera one by one
- The system will recognize registered students and mark attendance automatically
- Attendance is saved with date and time

### 4. Viewing Records
- Click "Records" from the navigation menu
- Select a date using the date picker
- View all attendance records for that date

## Project Structure

```
AMSuFR/
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css      # Styling
│   ├── js/
│   │   └── main.js        # Frontend JavaScript
│   └── faces/              # Stored face images
└── templates/
    ├── base.html           # Base template
    ├── login.html          # Login page
    ├── dashboard.html      # Dashboard
    ├── register_student.html # Registration form
    ├── capture_face.html   # Face capture page
    ├── attendance.html     # Attendance page
    └── attendance_records.html # Records page
```

## Database

The system uses SQLite database (attendance.db) which is automatically created when the application runs for the first time. The database contains:

- **Faculty** - Faculty login credentials
- **Student** - Registered student information and face encodings
- **Attendance** - Daily attendance records

## Technical Details

- **Backend:** Python Flask
- **Database:** SQLite with SQLAlchemy ORM
- **Face Recognition:** face_recognition library (based on dlib)
- **Frontend:** HTML5, CSS3, JavaScript
- **Video Processing:** OpenCV for camera access

## Important Notes

1. **Camera Permissions:** Ensure your browser has permission to access the camera
2. **Lighting:** Good lighting improves face recognition accuracy
3. **First Run:** The database is created automatically on first run
4. **Face Data:** Facial data is stored securely in the database

## Troubleshooting

- **Camera not working:** Check browser camera permissions
- **Face not detected:** Ensure good lighting and face is clearly visible
- **Application not starting:** Ensure all dependencies are installed correctly

## License

This project is for educational purposes.
