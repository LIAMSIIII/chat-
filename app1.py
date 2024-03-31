from flask import Flask, render_template, request, jsonify
import random
import json
import numpy as np
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import pickle
from keras.preprocessing import image
import os
import tensorflow as tf
from keras.applications.vgg16 import VGG16
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UPLOAD_FOLDER = r'C:\Users\L I A M S I\Desktop\chatbot\uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}





# Load intents data
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the neural network model
FILE = "data.pth"
data = torch.load(FILE, map_location=torch.device('cpu'))

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, img_size):
    # Load and resize the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    
    # Normalize pixel values
    img = img / 255.0
    
    return img

def load_model(model_path):
    return tf.keras.models.load_model(model_path)



def reset_heart_disease_assessment():
    global heart_disease_assessment, current_question_index, user_responses
    heart_disease_assessment = False
    current_question_index = 0
    user_responses = {}

class_labels = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']

# Define questions for heart disease assessment
heart_disease_questions = [
    "How old are you?",
    "What is your gender?",
    "Please specify the type of chest pain. (0,1,2,3)",
    "What is your resting blood pressure? (90-200)",
    "What is your serum cholesterol level? (120-600)",
    "Is your fasting blood sugar level greater than 120mg/dl? (Yes/No)",
    "What is your resting electrocardiographic result? (0,2)",
    "What is your maximum heart rate achieved? (70-210)",
    "Do you experience exercise-induced angina? (Yes/No)",
    "What is your ST depression induced by exercise relative to rest? (0-6.5)",
    "What is the slope of the peak exercise ST segment? (0-2)",
    "How many major vessels (0-3) colored by fluoroscopy do you have?",
    "What is your thalassemia type? (0 = normal; 1 = fixed defect; 2 = reversible defect)"
]

# Initialize variables to keep track of heart disease assessment state
heart_disease_assessment = False
current_question_index = 0
user_responses = {}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/get", methods=["POST"])
def get_response():
    global heart_disease_assessment, current_question_index

    user_message = request.form['msg']
    response = ""

    if "medical" in user_message.lower() or "heart disease" in user_message.lower():
        heart_disease_assessment = True
        current_question_index = 0
        response = heart_disease_questions[current_question_index]
    elif "brain tumor" in user_message.lower():
        response = "Please upload a photo of your brain for tumor prediction."
        return jsonify({'msg': response, 'showUpload': True})  # Trigger file upload section
    elif heart_disease_assessment:
        if current_question_index < len(heart_disease_questions):
            user_responses[heart_disease_questions[current_question_index]] = user_message
            current_question_index += 1
            if current_question_index < len(heart_disease_questions):
                response = heart_disease_questions[current_question_index]
            else:
                response = make_heart_disease_prediction()
    else:
        response = process_user_message(user_message)

    return jsonify({'msg': response})

def process_user_message(message):
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])

    return "I do not understand..."


def convert_response_to_numeric(question, response):
    if "gender" in question.lower():
        return 1 if response.lower() == "male" else 0
    elif "blood sugar" in question.lower():
        return 1 if response.lower() == "yes" else 0
    elif "angina" in question.lower():
        return 1 if response.lower() == "yes" else 0
    else:
        try:
            return float(response)
        except ValueError:
            return None

def make_heart_disease_prediction():
    feature_vector = []
    for question in heart_disease_questions:
        response = user_responses.get(question, "")
        if response:
            numeric_response = convert_response_to_numeric(question, response)
            if numeric_response is not None:
                feature_vector.append(numeric_response)
        else:
            feature_vector.append("")

    with open('heartdisease_model.pkl', 'rb') as file:
        heart_disease_model = pickle.load(file)

    user_input_numeric = np.array(feature_vector, dtype=float)
    prediction = heart_disease_model.predict(user_input_numeric.reshape(1, -1))

    reset_heart_disease_assessment()

    if prediction == 1:
        return "Our model predicts that you might have a heart disease."
    else:
        return "Our model predicts that you do not have a heart disease."
    import os
from werkzeug.utils import secure_filename



app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'msg': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'msg': 'No selected file'})
    
    if file and allowed_file(file.filename):
        # Generate a secure filename to avoid conflicts
        filename = secure_filename(file.filename)
        
        brain_tumor_model = load_model("brain_tum.h5")
        class_labels = ['no_tumor', 'glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']
        # Save the file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)
        #img = image.load_img(file_path, target_size=(300, 300))
        #img = np.asarray(img)
        #img = np.expand_dims(img, axis=0)
        #predictions = brain_tumor_model.predict(img)
        #predicted_class_index = np.argmax(predictions[0])
        #predicted_class = class_labels[predicted_class_index]
        # Make brain tumor prediction using the real file path
        predicted_class,confidence=predict_single_image(brain_tumor_model, file_path, (300,300), class_labels)
        return jsonify({'msg': f"Our model predicts that you have a {predicted_class} with a confidence of {confidence}."})
    
    return jsonify({'msg': 'Invalid file format'})





def predict_single_image(model, image_path, img_size, class_labels):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path, img_size)
    
    # Expand dimensions to match model input shape
    input_image = np.expand_dims(preprocessed_image, axis=0)
    
    # Make predictions
    predictions = model.predict(input_image)
    
    # Get predicted class index and confidence score
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get predicted class label
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class, confidence


if __name__ == "__main__":
    app.run(debug=True)

