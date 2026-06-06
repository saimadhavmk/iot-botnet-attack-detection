import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import scrolledtext
from tkinter import font as tkfont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, SimpleRNN, Dropout, Flatten, MaxPooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

global text_widget
global X_train, X_test, y_train, y_test, df
accuracy = []
precision = []
recall = []
fscore = []  
algorithm = []

def upload_dataset():
    global df
    text_widget.delete(1.0,  END)
    file_path = filedialog.askopenfilename(initialdir = 'Dataset', filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        text_widget.insert(END, f"{file_path}\nDataset Uploaded\n{df.head()}")

def preprocess_dataset():
    global X_train, X_test, y_train, y_test, df
    text_widget.delete(1.0,  END)
    if df is None:
        text_widget.insert(END, "No dataset uploaded")
        return
    text_widget.insert(END, "Preprocessing dataset...\n")

    for col in ['proto', 'service', 'state']:
        df[col] = df[col].astype('category').cat.codes
    df['attack_cat'] = df['attack_cat'].astype('category')
    
    X = df.drop(columns = ['attack_cat', 'label', 'id'])
    y = df['label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = pd.DataFrame(X, columns=['dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes',
           'dbytes', 'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss',
           'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
           'tcprtt', 'synack', 'ackdat', 'smean', 'dmean', 'trans_depth',
           'response_body_len', 'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm',
           'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
           'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
           'ct_srv_dst', 'is_sm_ips_ports'])
    text_widget.insert(END, X.head())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    text_widget.insert(END, "\nDataset has been split into training and testing sets!\n")
    text_widget.insert(END, f"Training set size: {X_train.shape[0]}\n")
    text_widget.insert(END, f"Test set size: {X_test.shape[0]}\n")
    X_train = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))
    text_widget.insert(END, "Preprocessing Complete\n")
    
def calculate_metrics(algorithm_name, predict, testY):
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100
    
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    algorithm.append(algorithm_name)
    
    text_widget.delete(1.0, END)
    text_widget.insert(END, f"\n{algorithm_name} Evaluation:\n")
    text_widget.insert(END, f"{algorithm_name} Accuracy: {a:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} Precision: {p:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} Recall: {r:.2f}%\n")
    text_widget.insert(END, f"{algorithm_name} F-Measure: {f:.2f}%\n")
    
    labels = np.unique(testY)
    label_map = {0: 'Normal', 1: 'Attacked'}
    label_names = [label_map[label] for label in labels]

    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    ax = sns.heatmap(conf_matrix, xticklabels=label_names, yticklabels=label_names, annot=True, cmap="viridis", fmt="g")
    ax.set_ylim([0, len(label_names)])
    plt.title(f"{algorithm_name} Confusion Matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def train_cnn():
    global X_train, X_test, y_train, y_test
    model_file = 'Models/cnn.hdf5'
    text_widget.delete(1.0,  END)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        text_widget.insert(END, "Training new model...")
        model = Sequential()
        model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(128, 3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64)
        model.save("Models/cnn.hdf5")       
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    model.summary()
    calculate_metrics("CNN Algorithm", y_pred, y_test)

def train_lstm():
    global X_train, X_test, y_train, y_test
    model_file = 'Models/LSTM_Model.hdf5'
    text_widget.delete(1.0,  END)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        text_widget.insert(END, "Training new model...")
        model = Sequential()
        model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64, return_sequences=True))
        model.add(Dropout(0.5))
        model.add(LSTM(64))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64)
        model.save("Models/LSTM_Model.hdf5")
    model.summary()
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)  
    calculate_metrics("LSTM Algorithm", y_pred, y_test)
    

def train_rnn():
    global X_train, X_test, y_train, y_test
    model_file = 'Models/RNN_Model.hdf5'
    text_widget.delete(1.0,  END)
    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        text_widget.insert(END, "Training new model...")
        model = Sequential()
        model.add(SimpleRNN(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(32))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=64)
        model.save("Models/RNN_Model.hdf5")
    model.summary()
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)  
    calculate_metrics("RNN Algorithm", y_pred, y_test)
    

def plot_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='blue')
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f'{model_name} - ROC & PR Curves')
    plt.xlabel('False Positive Rate / Recall')
    plt.ylabel('True Positive Rate / Precision')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return roc_auc, pr_auc
 
    
def train_aclr():
    global X_train, X_test, y_train, y_test

    model_file = 'Models/aclr_hybrid_model.hdf5'
    text_widget.delete(1.0, END)

    if os.path.exists(model_file):
        model = load_model(model_file)
    else:
        text_widget.insert(END, "Training new hybrid model...\n")
        model = Sequential()
        model.add(Conv1D(128, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(128, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(SimpleRNN(64))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=30, batch_size=32,callbacks=[early_stop], class_weight='balanced')
        model.save(model_file)
    model.summary()

    # Predict probabilities first
    y_pred_proba = model.predict(X_test)
    # Adjusted threshold for better recall or precision tradeoff
    optimal_threshold = 0.4
    y_pred = (y_pred_proba > optimal_threshold)

    a = accuracy_score(y_test, y_pred) * 100
    p = precision_score(y_test, y_pred, average='macro') * 100
    r = recall_score(y_test, y_pred, average='macro') * 100
    f = f1_score(y_test, y_pred, average='macro') * 100
    roc_auc = roc_auc_score(y_test, y_pred_proba) * 100
    pr_auc = average_precision_score(y_test, y_pred_proba) * 100
    algorithm_name = "Hybrid ACLR Algorithm"
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    algorithm.append(algorithm_name)

    text_widget.insert(END, f"\n{algorithm_name} Evaluation:\n")
    text_widget.insert(END, f"{algorithm_name} Accuracy: {a:.2f}%\n")
    text_widget.insert(END, f'{algorithm_name} ROC-AUC: {roc_auc:.2f}\n')
    text_widget.insert(END, f'{algorithm_name} PR-AUC: {pr_auc:.2f}\n')
    plot_curve(y_test, y_pred_proba, algorithm_name)


def train_aclrKfold():
    k_values = [3, 5, 7, 10]
    global X_train, X_test, y_train, y_test
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    model_file = 'Models/aclrhybrid_kfold.hdf5'
    text_widget.delete(1.0,  END)
    if os.path.exists(model_file):
        text_widget.insert(END, "Loading pre-trained hybrid model...\n")
        model = load_model(model_file)
        print("Loaded pre-trained model:")
        model.summary()  # Print to console
    else:
        text_widget.insert(END, "Training new hybrid model...\n")
        for k in k_values:
            text_widget.insert(END, f"Training model with K={k}...\n")
            fold_accuracies = []
            fold_roc_auc = []
            fold_pr_auc = []
            for train_index, val_index in kfold.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                model = Sequential()
                model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train_fold.shape[1], 1)))
                model.add(Dropout(0.5))
                model.add(MaxPooling1D(pool_size=2))
                model.add(LSTM(64, return_sequences=True))
                model.add(Dropout(0.5))
                model.add(SimpleRNN(64))
                model.add(Dropout(0.5))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
                model.fit(X_train_fold, y_train_fold, epochs=30, batch_size=32, verbose=1)
                model.summary()
                y_val_pred = model.predict(X_val_fold)
                y_val_pred = (y_val_pred > 0.5)
                accuracy = accuracy_score(y_val_fold, y_val_pred)
                roc_auc = roc_auc_score(y_val_fold, y_val_pred)
                pr_auc = average_precision_score(y_val_fold, y_val_pred)
                fold_accuracies.append(accuracy)
                fold_roc_auc.append(roc_auc)
                fold_pr_auc.append(pr_auc)
            avg_accuracy = np.mean(fold_accuracies)
            avg_roc_auc = np.mean(fold_roc_auc)
            avg_pr_auc = np.mean(fold_pr_auc)
            results.append((k, avg_accuracy, avg_roc_auc, avg_pr_auc))
            text_widget.insert(END,f"K={k} - Avg Accuracy: {avg_accuracy:.4f}, Avg ROC-AUC: {avg_roc_auc:.4f}, Avg PR-AUC: {avg_pr_auc:.4f}\n")        
        model.save(model_file)
        print("Final model architecture after training:")
        model.summary() 
    best_k = max(results, key=lambda x: x[1])
    text_widget.insert(END, f"Best K value: {best_k[0]} with Accuracy: {best_k[1]:.4f}\n")

def compare_graph():

    text_widget.delete(1.0, END)
    if not algorithm:
        text_widget.insert(END, "\nNo model results to compare. Please train Models first.\n")
        return
    metrics = {'Accuracy': accuracy,'Precision': precision,'Recall': recall,'F1-Score': fscore}
    df = pd.DataFrame(metrics, index=algorithm)

    # Display table in console or text_widget (optional)
    text_widget.insert(END, "\nModel Metrics Table:\n")
    text_widget.insert(END, df.to_string())
    text_widget.insert(END, "\n\n")
    
    x = np.arange(len(algorithm))  # positions for x-axis
    width = 0.2  # width of each bar
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric_name, values) in enumerate(metrics.items()):
        bar = ax.bar(x + i*width, values, width=width, label=metric_name)
        for rect in bar:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),xytext=(0, 3), textcoords="offset points",ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Percentage')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(algorithm, rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.show()

def prediction():
    global df
    model_file = 'Models/aclrhybrid_kfold.hdf5'
    if not os.path.exists(model_file):
        text_widget.insert(END, "Model not found. Please train the ACLR K-Fold model first.\n")
        return

    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        text_widget.insert(END, "No file selected for prediction.\n")
        return

    text_widget.delete(1.0, END)
    text_widget.insert(END, f"Loading data from: {file_path}\n")

    test_data = pd.read_csv(file_path)

    for col in ['proto', 'service', 'state']:
        if col in test_data.columns:
            test_data[col] = test_data[col].astype('category').cat.codes

    for col in ['attack_cat', 'label', 'id']:
        if col in test_data.columns:
            test_data.drop(col, axis=1, inplace=True)

    scaler = StandardScaler()
    test_data_scaled = scaler.fit_transform(test_data)
    test_data_scaled = test_data_scaled.reshape((test_data_scaled.shape[0], test_data_scaled.shape[1], 1))

    model = load_model(model_file)
    y_pred_proba = model.predict(test_data_scaled)
    y_pred = (y_pred_proba > 0.4).astype(int)

    label_map = {0: "Normal", 1: "Attack"}
    new_data = pd.read_csv(file_path)
    text_widget.insert(END, "\nPrediction Results:\n")
    for i in range(len(y_pred)):
        record_features = new_data.iloc[i]
        prediction_label = label_map[y_pred[i][0]]
        text_widget.insert(END, f"Record: {record_features.values} ===> Predicted as {prediction_label}\n\n")


root = tkinter.Tk()
root.geometry('1200x600')
root.title("Botnet Attack Detection in IoT Environment")
root.configure(bg="#f0f4f7")

heading = Label(root, text="Hybrid Machine Learning Model for Efficient Botnet Attack Detection in IoT Environment", bg="#f0f4f7", fg="#1f3a93")
heading.config(height=3, width=100)
heading.config(font=("Helvetica", 14, "bold"))
heading.place(x=0, y=10)

text_widget = scrolledtext.ScrolledText(root, wrap=WORD, font=("Courier New", 12, 'bold'), bg='black', fg='white')
text_widget.place(x=300, y=100)

font1 = (("Courier New", 12, 'bold'))
upload = Button(root, text="Upload Dataset", command=upload_dataset,bg="#2e86de", fg="white", font = font1)
upload.place(x=30, y=100)

preprocess = Button(root, text="Preprocess Dataset", command=preprocess_dataset,bg="#2e86de", fg="white", font = font1)
preprocess.place(x=30, y=150)

cnn = Button(root, text="Train Existing CNN", command=train_cnn,bg="#2e86de", fg="white", font = font1)
cnn.place(x=30, y=200)

lstm = Button(root, text="Train Existing LSTM", command=train_lstm,bg="#2e86de", fg="white", font = font1)
lstm.place(x=30, y=250)

rnn = Button(root, text="Train Existing RNN", command=train_rnn,bg="#2e86de", fg="white", font = font1)
rnn.place(x=30, y=300)

aclrHybrid = Button(root, text="Train Proposed ACLR", command=train_aclr,bg="#2e86de", fg="white", font = font1)
aclrHybrid.place(x=30, y=350)

compareGraph = Button(root, text="Comparison Graph", command=compare_graph,bg="#2e86de", fg="white", font = font1)
compareGraph.place(x=30, y=400)

predict = Button(root, text="Botnet Attack Prediction", command=prediction,bg="#2e86de", fg="white", font = font1)
predict.place(x=30, y=450)


# Start GUI loop
root.mainloop()
