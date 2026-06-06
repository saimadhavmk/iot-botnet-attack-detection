# IoT Botnet Attack Detection

A deep learning-based cybersecurity project designed to detect botnet attacks in IoT (Internet of Things) environments using multiple neural network models including CNN, LSTM, RNN, and a hybrid ACLR model.

## Project Overview

With the rapid growth of IoT devices, cyber threats such as botnet attacks have become increasingly common. This project aims to identify malicious botnet traffic using deep learning techniques and improve detection performance through a hybrid model approach.

The system preprocesses network traffic data, trains multiple deep learning models, compares their performance, and predicts whether traffic is normal or malicious.

---

## Features

- Botnet attack detection in IoT networks
- Data preprocessing and normalization
- Deep learning model training:
  - CNN
  - LSTM
  - RNN
  - Hybrid ACLR Model
- Performance comparison using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confusion Matrix Visualization
- ROC Curve & Precision-Recall Curve
- GUI-based implementation using Tkinter

---

## Tech Stack

- Python
- TensorFlow / Keras
- Scikit-Learn
- Pandas
- NumPy
- Tkinter
- Matplotlib
- Seaborn

---

## Dataset

This project uses the **UNSW-NB15 Dataset** for botnet attack detection in IoT environments.

Dataset source:  
https://research.unsw.edu.au/projects/unsw-nb15-dataset

---

## Project Workflow

1. Upload dataset  
2. Preprocess network traffic data  
3. Train CNN, LSTM, and RNN models  
4. Train Hybrid ACLR model  
5. Compare model performance  
6. Predict botnet attacks on test data

---

## Installation

Clone the repository:

```bash
git clone https://github.com/saimadhavmk1/iot-botnet-attack-detection.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## Screenshots

(Add screenshots of model outputs and GUI here)

---

## Future Improvements

- Real-time network traffic detection
- Cloud deployment
- Improved hybrid model optimization
- Better visualization dashboard

---

## Author

**Sai Madhav Mokirala**
