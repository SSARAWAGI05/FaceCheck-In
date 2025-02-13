# AttendanceFrontend

## Problem Statement
Managing attendance manually can be time-consuming and prone to errors. Traditional methods of attendance tracking, such as roll calls and sign-in sheets, lack efficiency and accuracy. Additionally, monitoring unauthorized access to restricted areas is a challenge. Our solution automates attendance tracking using AI-powered facial recognition to improve efficiency, security, and reliability.

## AttendanceFrontend (Solution)
AttendanceFrontend is a facial recognition-based attendance system that automates the tracking and monitoring of individuals in an organization. It captures and processes live video streams to detect and recognize faces, marking attendance seamlessly. Unrecognized individuals are flagged and stored in a separate directory for review. The platform ensures an accurate, secure, and hassle-free way of managing attendance records.

## Installation
### Cloning the repository
```bash
git clone https://github.com/SSARAWAGI05/AttendanceFrontend.git
```

### Run AttendanceFrontend frontend with Flask
```bash
pip install -r requirements.txt
python app.py
```

## Features of AttendanceFrontend
- **Automated Face Recognition**: Detects and marks attendance of registered users in real-time.
- **Unknown Faces Detection**: Captures and stores images of unrecognized faces for review.
- **Live Video Processing**: Continuously monitors and updates attendance records using webcam feeds.
- **User-Friendly Web Interface**: Provides an easy-to-use dashboard for administrators to view attendance records.

## Technology Used
### Backend
- Flask (Python)

### Frontend
- HTML, CSS
- JavaScript

### AI/ML
- OpenCV for real-time face detection
- Deep Learning models for facial recognition

## Acknowledgements
We extend our sincere gratitude to the open-source community for providing valuable resources that made this project possible.

## Contributor
Shubam Sarawagi

