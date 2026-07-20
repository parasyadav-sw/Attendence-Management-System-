// Face Capture Functionality
let videoStream = null;
let canvas = null;
let context = null;

function startCamera(videoElement, canvasElement) {
    canvas = canvasElement;
    context = canvas.getContext('2d');
    
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            videoStream = stream;
            videoElement.srcObject = stream;
            videoElement.play();
        })
        .catch(err => {
            console.error('Error accessing camera:', err);
            alert('Unable to access camera. Please ensure camera permissions are granted.');
        });
}

function captureFrame(videoElement) {
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg');
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
    }
}

// Save Face to Server
function saveFace(studentId, imageData) {
    return fetch('/save-face', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `student_id=${studentId}&face_data=${encodeURIComponent(imageData)}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            return { success: true, message: data.message };
        } else {
            return { success: false, error: data.error };
        }
    })
    .catch(error => {
        return { success: false, error: 'Network error occurred' };
    });
}

// Mark Attendance
function markAttendance(studentId) {
    return fetch('/mark-attendance', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ student_id: studentId })
    })
    .then(response => response.json())
    .then(data => {
        return data;
    })
    .catch(error => {
        return { error: 'Network error occurred' };
    });
}

// Show Flash Message
function showFlashMessage(message, type) {
    const flashContainer = document.querySelector('.flash-messages');
    if (!flashContainer) {
        const container = document.createElement('div');
        container.className = 'flash-messages';
        document.body.appendChild(container);
    }
    
    const flash = document.createElement('div');
    flash.className = `flash ${type}`;
    flash.textContent = message;
    
    document.querySelector('.flash-messages').appendChild(flash);
    
    setTimeout(() => {
        flash.remove();
    }, 5000);
}

// Attendance Page Functionality
let attendanceInterval = null;

function startAttendanceMode() {
    const video = document.getElementById('attendance-video');
    if (!video) return;
    
    startCamera(video, document.createElement('canvas'));
    
    // Check for faces every 2 seconds
    attendanceInterval = setInterval(() => {
        const imageData = captureFrame(video);
        
        // Send frame to server for face recognition
        fetch('/video-feed-check', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData })
        })
        .then(response => response.json())
        .then(data => {
            if (data.student_id) {
                markAttendance(data.student_id).then(result => {
                    if (result.success) {
                        showFlashMessage(result.message, 'success');
                        updateAttendanceList(result.student_name);
                    }
                });
            }
        })
        .catch(err => console.log('Face detection error:', err));
    }, 2000);
}

function stopAttendanceMode() {
    if (attendanceInterval) {
        clearInterval(attendanceInterval);
    }
    stopCamera();
}

function updateAttendanceList(studentName) {
    const list = document.getElementById('attendance-list');
    if (!list) return;
    
    const li = document.createElement('li');
    const time = new Date().toLocaleTimeString();
    
    li.innerHTML = `
        <span>${studentName}</span>
        <span class="status">Present</span>
        <span>${time}</span>
    `;
    
    list.insertBefore(li, list.firstChild);
}

// Form Validation
function validateRegistrationForm() {
    const name = document.getElementById('name').value.trim();
    const studentId = document.getElementById('student_id').value.trim();
    const className = document.getElementById('class_name').value.trim();
    
    if (!name || !studentId || !className) {
        showFlashMessage('Please fill in all required fields', 'error');
        return false;
    }
    
    return true;
}

// Initialize Camera on Page Load
document.addEventListener('DOMContentLoaded', function() {
    const cameraVideo = document.getElementById('camera-video');
    const captureCanvas = document.getElementById('capture-canvas');
    
    if (cameraVideo && captureCanvas) {
        startCamera(cameraVideo, captureCanvas);
    }
    
    // Auto-hide flash messages
    const flashMessages = document.querySelectorAll('.flash');
    flashMessages.forEach(flash => {
        setTimeout(() => {
            flash.style.animation = 'slideOut 0.3s ease forwards';
            setTimeout(() => flash.remove(), 300);
        }, 5000);
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopCamera();
    stopAttendanceMode();
});
