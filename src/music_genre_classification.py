import librosa
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Function to extract features (MFCC, chroma, tempo) from an audio file
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return np.hstack((mfccs, chroma, tempo))

# Main function to run the entire process
def main():
    features_list = []
    genres = ['blues', 'classical', 'jazz', 'rock']

    for genre in genres:
        folder_path = f'../data/genres_original/{genre}'
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                try:
                    features = extract_features(file_path)
                    features_list.append(np.append(features, genre))
                except Exception as e:
                    print(f"Error processing")

    # Create DataFrame with the extracted features
    columns = [f'mfcc{i}' for i in range(13)] + [f'chroma{i}' for i in range(12)] + ['tempo', 'genre']
    df = pd.DataFrame(features_list, columns=columns)

    # Split the dataset into features and labels
    X = df.drop('genre', axis=1)
    y = df['genre']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate the model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()