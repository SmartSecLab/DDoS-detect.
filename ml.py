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

# Load the labeled dataset
labeled_file = 'labeled_db.csv'
df_labeled = pd.read_csv(labeled_file, parse_dates=['Timestamp'])

# Encode categorical labels using LabelEncoder
le = LabelEncoder()
df_labeled['Attack_Type'] = le.fit_transform(df_labeled['Attack_Type'])

# Separate features and target variable
X = df_labeled.drop(['Timestamp', 'Attack_Type'], axis=1)
y = df_labeled['Attack_Type']

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

# Train common ML classifiers
for clf_name, clf in classifiers.items():
    print(f"\nTraining Classifier: {clf_name}")
    
    # Train the classifier
    clf.fit(X_train, y_train)

# Improved Deep Learning model
print("\nTraining Deep Learning Model")
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(df_labeled['Attack_Type'].unique()), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=128, validation_data=(X_test, y_test))

# Evaluate all models
print("\nEvaluation Results:")
for clf_name, clf in classifiers.items():
    # Make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # Evaluate the classifier
    print(f"\n{clf_name} Results:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Evaluate the Deep Learning model
print("\nDeep Learning Model Results:")
y_pred_nn = model.predict(X_test)
y_pred_nn_classes = y_pred_nn.argmax(axis=-1)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nn_classes))
print("Classification Report:\n", classification_report(y_test, y_pred_nn_classes))

