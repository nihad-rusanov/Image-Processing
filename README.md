### 📂 Project Structure

```
OpenCv_Project/
│
├── dataset/                        # Stores captured face images (User.ID.jpg)
│   ├── User.1.5.jpg
│   ├── User.1.6.jpg
│   └── ...
│
├── trainer.yml                     # Trained face recognizer model
├── main.py                         # Face image data collection (20 images/user)
├── train.py                        # Trains LBPH recognizer from dataset
├── face_recognition.py             # Real-time face detection and recognition
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## ✅ Expanded PowerPoint — Slide 5: System Architecture

### **Slide Title**: System Architecture

---

### **Content**:

#### **Step 1: Data Collection**

* Script: `main.py`
* Uses OpenCV and Haar Cascade to detect faces from webcam.
* Captures and saves 20 grayscale face images per user.
* Output: `.jpg` images saved in `/dataset/`.

#### **Step 2: Model Training**

* Script: `train.py`
* Loads all images from `/dataset/`, extracts user IDs from filenames.
* Trains an LBPH model (`cv2.face.LBPHFaceRecognizer_create()`).
* Saves model as `trainer.yml`.

#### **Step 3: Real-Time Recognition**

* Script: `face_recognition.py`
* Uses webcam to capture live video.
* Detects faces and matches them against `trainer.yml`.
* Labels detected faces as “User X” or “Unknown”.

---

### Diagram (if included in the slide):

```plaintext
[main.py] ──► Captures Face Images ──► [dataset/]
       │
       ▼
[train.py] ──► Trains LBPH Recognizer ──► [trainer.yml]
       │
       ▼
[face_recognition.py] ──► Real-Time Webcam Recognition
```

---


