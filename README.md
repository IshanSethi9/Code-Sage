# 🔢 Code-Sage – Exploring the Golden Ratio with AI & Computer Vision

**Code-Sage** is an interactive web application that visualizes and explores the concept of the **Golden Ratio (𝜙 ≈ 1.618)** through real-time computer vision and engaging UI components. Developed as part of a university project at **SRM University**, it helps users discover how the Golden Ratio appears in nature, architecture, the human body, and even musical instruments.

---

## 🎯 Key Features

### 🧮 Golden Ratio Detector
- Uses **computer vision** to detect and measure objects via webcam.
- Calculates the ratio of dimensions and highlights those that approximate the Golden Ratio.

### 🎸 Live Guitar Player Game
- Includes a game where users must identify guitars designed using the Golden Ratio.
- Upon selecting the correct guitar, users are redirected to an **interactive guitar simulator**.

### 🧍 Human Body Mapping
- Demonstrates how various proportions of the human body align with the Golden Ratio.
- Great for anatomy-based learning or mathematical visualization.

### 📚 Interactive Tabs
- Sections include: **Nature**, **Art**, **Architecture**, **Finance**, and **Human Anatomy**.
- Provides educational content alongside visual demonstrations.

---

## 🛠️ Tech Stack

- **HTML / CSS / JavaScript**
- **Python (for CV components)**
- **OpenCV**
- **MediaPipe / Dlib** *(optional for live detection)*
- **Flask** *(optional for web integration)*

---

## 📁 Folder Structure

Code-Sage/
├── golden_ratio_detector/ # CV scripts for detecting golden ratio in objects
├── live_guitar_game/ # Guitar game based on object recognition
├── human_body_demo/ # Golden ratio visual mapping in human anatomy
├── static/ # CSS, images, JavaScript
├── templates/ # HTML pages
├── app.py # Flask app (if applicable)
├── README.md
└── requirements.txt # Dependencies


---

## 💻 Setup Instructions

1. Clone the repository
```
git clone https://github.com/IshanSethi9/Code-Sage.git
cd Code-Sage
```
2. (Optional) Set up virtual environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run the application
```
python app.py
```
5. Open your browser and navigate to: http://127.0.0.1:5000/

🌟 Project Highlights
Combines math + art + computer vision in a fun and educational way.

Demonstrates the aesthetic and structural importance of the Golden Ratio across multiple domains.

Designed to promote interactive and visual learning.

🎯 Future Improvements
Integrate with mobile cameras for object detection on the go.

Add audio feedback for detection results.

Deploy as a live web app using Streamlit or Heroku.

Extend detection to real-world photos and design models.

📄 License
This project is licensed under the MIT License.

## 🔗 Connect with Me

- **LinkedIn:** [Ishan Sethi](https://www.linkedin.com/in/ishansethi09/)
- **GitHub:** [IshanSethi9](https://github.com/IshanSethi9)


