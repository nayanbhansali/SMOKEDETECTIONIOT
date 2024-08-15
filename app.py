from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load your data
df = pd.read_csv('smoke_detection_iot.csv')

# Encode and preprocess data
df['Fire Alarm'] = df['Fire Alarm'].astype('category').cat.codes
X = df.drop(columns=['Fire Alarm', 'UTC'])  # Remove UTC column
y = df['Fire Alarm']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression()
}

model_accuracies = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy

# Save the best model
best_model_name = max(model_accuracies, key=model_accuracies.get)
best_model = models[best_model_name]

with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Load scaler and model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    form_data = [float(x) for x in request.form.values()]
    features = np.array(form_data).reshape(1, -1)
    
    # Standardize the features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = best_model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
