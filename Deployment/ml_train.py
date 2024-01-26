import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import os

# Load the labeled dataset
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..'))
labeled_file = os.path.join(parent_directory, 'Data', 'labeled_db.csv')

df_labeled = pd.read_csv(labeled_file, parse_dates=['Timestamp'])

# Encode categorical labels using LabelEncoder
le = LabelEncoder()
df_labeled['Attack-type'] = le.fit_transform(df_labeled['Attack-type'])

# Separate features and target variable
X = df_labeled.drop(['Timestamp', 'Attack-type'], axis=1)
y = df_labeled['Attack-type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Common ML classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'k-Nearest Neighbors': KNeighborsClassifier()
}

# Store metrics in a dictionary
metrics_data = {'Classifier': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'Support': []}

# Train common ML classifiers
for clf_name, clf in classifiers.items():
    print(f"\nTraining Classifier: {clf_name}")

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the classifier
    report = classification_report(y_test, y_pred, output_dict=True)

    # Store metrics in the dictionary
    metrics_data['Classifier'].append(clf_name)
    metrics_data['Accuracy'].append(f"{report['accuracy']*100:.3f}")
    metrics_data['Precision'].append(f"{report['weighted avg']['precision']*100:.3f}")
    metrics_data['Recall'].append(f"{report['weighted avg']['recall']*100:.3f}")
    metrics_data['F1 Score'].append(f"{report['weighted avg']['f1-score']*100:.3f}")
    metrics_data['Support'].append(report['weighted avg']['support'])

# Deep Learning model
print("\nTraining Deep Learning Model")
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(df_labeled['Attack-type'].unique()), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the Deep Learning model
print("\nDeep Learning Model Results:")
y_pred_nn = model.predict(X_test)
y_pred_nn_classes = y_pred_nn.argmax(axis=-1)

# Evaluate the classifier
report_nn = classification_report(y_test, y_pred_nn_classes, output_dict=True)

# Store metrics in the dictionary
metrics_data['Classifier'].append('Deep Learning Model')
metrics_data['Accuracy'].append(f"{report_nn['accuracy']*100:.3f}")
metrics_data['Precision'].append(f"{report_nn['weighted avg']['precision']*100:.3f}")
metrics_data['Recall'].append(f"{report_nn['weighted avg']['recall']*100:.3f}")
metrics_data['F1 Score'].append(f"{report_nn['weighted avg']['f1-score']*100:.3f}")
metrics_data['Support'].append(report_nn['weighted avg']['support'])

# Create a DataFrame from the metrics data
metrics_df = pd.DataFrame(metrics_data)

# Print the metrics table
print("\nMetrics Table:")
print(metrics_df.to_markdown(index=False))

# Export the RF model
filename = 'random_forest_classifier.joblib'
random_forest_classifier = classifiers['Random Forest']
joblib.dump(random_forest_classifier, filename)

#  LabelEncoder object
label_encoder_filename = 'label_encoder.joblib'
joblib.dump(le, label_encoder_filename)
