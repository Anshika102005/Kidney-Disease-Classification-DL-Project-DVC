# 🩺 Kidney Disease Classification using Deep Learning

A deep learning-based web application that automatically classifies **kidney CT scan images** into four categories: **Normal, Cyst, Stone, and Tumor**.

The project follows an end-to-end **MLOps workflow** using **TensorFlow, MLflow, DVC, and DagsHub**, with a Flask backend and a modern React-based frontend.

---

## 📌 Project Overview

Kidney diseases can be challenging to identify manually from CT scan images. This project uses a **Convolutional Neural Network (CNN)** to classify kidney CT scan images into four categories:

* 🟢 **Normal**
* 🟡 **Cyst**
* 🟠 **Stone**
* 🔴 **Tumor**

Users can upload a kidney CT scan through the web interface and receive a predicted class from the trained deep learning model.

> **Note:** This project is intended for educational and research purposes and should not be used as a substitute for professional medical diagnosis.

---

## ✨ Features

* 🧠 Deep Learning-based kidney CT scan classification
* 🔬 Four-class classification: Normal, Cyst, Stone, Tumor
* 📊 Experiment tracking with MLflow
* 🔄 Data and pipeline versioning using DVC
* 🌐 DagsHub integration
* ⚡ Flask-based prediction API
* 💻 Interactive React frontend
* 🎨 Tailwind CSS-based UI
* 🧩 Modular project architecture
* 🐳 Docker support
* 🔁 Reproducible ML pipeline
* 📦 Git/GitHub version control

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* CNN

### MLOps

* MLflow
* DVC
* DagsHub

### Backend

* Flask
* Flask-CORS

### Frontend

* React
* TypeScript
* Vite
* Tailwind CSS
* Framer Motion
* React Three Fiber

### Data & Image Processing

* NumPy
* Pandas
* OpenCV
* Pillow

### Development & Deployment

* Git
* GitHub
* Docker
* WSL

---

## 📂 Project Structure

```text
Kidney-Disease-Classification-DL-Project-DVC/
│
├── config/
│
├── src/
│   ├── components/
│   ├── config/
│   ├── entity/
│   ├── pipeline/
│   └── utils/
│
├── Frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── artifacts/
│
├── research/
│
├── config.yaml
├── params.yaml
├── dvc.yaml
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## 🔄 ML Workflow

The project follows a modular and reproducible machine learning workflow:

```text
Dataset
   ↓
Data Ingestion
   ↓
Data Preparation
   ↓
Base Model Preparation
   ↓
Model Training
   ↓
Model Evaluation
   ↓
MLflow Experiment Tracking
   ↓
DVC Pipeline
   ↓
Trained Model
   ↓
Flask Prediction API
   ↓
React Frontend
   ↓
Prediction
```

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC.git

cd Kidney-Disease-Classification-DL-Project-DVC
```

---

## 2️⃣ Create a Virtual Environment

### Using Conda

```bash
conda create -n cnncls python=3.11 -y
```

Activate the environment:

```bash
conda activate cnncls
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Train the Model

To execute the complete training pipeline:

```bash
python main.py
```

The pipeline performs the required stages for data preparation, model preparation, training, and evaluation.

The trained model is saved inside the `artifacts/` directory.

---

# 📊 MLflow Experiment Tracking

Start the MLflow tracking UI:

```bash
mlflow ui --port 5001
```

Open:

```text
http://127.0.0.1:5001
```

MLflow can be used to track:

* Training metrics
* Validation metrics
* Loss
* Accuracy
* Model experiments
* Parameters

---

# 🌐 DagsHub Integration

The project can use DagsHub for remote experiment tracking and MLflow integration.

```python
import dagshub

dagshub.init(
    repo_owner="Anshika102005",
    repo_name="Kidney-Disease-Classification-DL-Project-DVC",
    mlflow=True
)
```

### Set Environment Variables

#### Linux / WSL

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC.mlflow"

export MLFLOW_TRACKING_USERNAME="Anshika102005"

export MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"
```

#### Windows PowerShell

```powershell
$env:MLFLOW_TRACKING_URI="https://dagshub.com/Anshika102005/Kidney-Disease-Classification-DL-Project-DVC.mlflow"

$env:MLFLOW_TRACKING_USERNAME="Anshika102005"

$env:MLFLOW_TRACKING_PASSWORD="<YOUR_DAGSHUB_TOKEN>"
```

> 🔐 Never commit your DagsHub token or other credentials to GitHub.

---

# 📦 DVC

DVC is used to manage the machine learning pipeline and maintain reproducibility.

### Initialize DVC

```bash
dvc init
```

### Run the Pipeline

```bash
dvc repro
```

### Check Pipeline Status

```bash
dvc status
```

### Visualize Pipeline

```bash
dvc dag
```

---

# 🔬 Model Evaluation

The trained model can be evaluated using the evaluation pipeline.

Evaluation metrics include:

* Accuracy
* Loss

The evaluation results are stored in the project artifacts and can also be tracked through MLflow.

---

# 🖥️ Run Flask Backend

Start the Flask backend:

```bash
python app.py
```

The backend will run on:

```text
http://localhost:5000
```

The Flask application loads the trained model and provides the prediction functionality for uploaded kidney CT scan images.

---

# 💻 Run Frontend

Open a new terminal and navigate to the frontend:

```bash
cd Frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will generally be available at:

```text
http://localhost:5173
```

---

# 🐳 Run with Docker

Docker can be used to package and run the Flask backend in a reproducible environment.

## Build Docker Image

From the project root:

```bash
docker build -t kidney-ml-backend .
```

## Check Docker Image

```bash
docker images
```

You should see:

```text
kidney-ml-backend
```

## Run Docker Container

```bash
docker run -p 5000:5000 kidney-ml-backend
```

The Flask backend will be available at:

```text
http://localhost:5000
```

### Stop the Container

First check running containers:

```bash
docker ps
```

Then:

```bash
docker stop <CONTAINER_ID>
```

---

# 🧠 Model Classes

| Class     | Description   |
| --------- | ------------- |
| 🟢 Normal | Normal kidney |
| 🟡 Cyst   | Kidney cyst   |
| 🟠 Stone  | Kidney stone  |
| 🔴 Tumor  | Kidney tumor  |

---

# 📈 MLOps Components

| Tool                 | Purpose                            |
| -------------------- | ---------------------------------- |
| **TensorFlow/Keras** | Deep Learning model                |
| **MLflow**           | Experiment tracking                |
| **DVC**              | Data & pipeline versioning         |
| **DagsHub**          | Remote ML collaboration & tracking |
| **Git/GitHub**       | Source code version control        |
| **Docker**           | Application containerization       |
| **Flask**            | Backend API                        |
| **React**            | Frontend interface                 |

---

# 🔮 Future Improvements

* ☁️ AWS EC2 deployment
* 🔄 CI/CD using GitHub Actions
* 📊 Model monitoring
* ☸️ Kubernetes deployment
* 🔐 User authentication
* 📜 Prediction history
* 🚀 REST API documentation
* 📈 Improved model performance
* 🧪 Automated model testing

---

# 👩‍💻 Author

### Anshika Sahu

B.Tech CSE — Artificial Intelligence & Machine Learning

**GitHub:** [Anshika102005](https://github.com/Anshika102005)

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ **Star** on GitHub.
