from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib

app = Flask(__name__)

MODEL_FILE = 'data/crop_recommendation_model.pkl'

def get_crop_name(predicted_crop):
    crop_names = [
        'Apple', 'Banana', 'Blackgram', 'Chickpea', 'Coconut', 'Coffee', 'Cotton', 'Grapes',
        'Jute', 'Kidneybeans', 'Lentil', 'Maize', 'Mango', 'Mothbeans', 'Mungbeans',
        'Muskmelon', 'Orange', 'Papaya', 'Pigeonpeas', 'Pomegranate', 'Rice', 'Watermelon'
    ]
    return crop_names[predicted_crop]

def get_level(value, low, high, labels):
    for i in range(len(labels)):
        if low <= value <= high:
            return labels[i]
        low, high = high, high * 2
    return labels[-1]

def train_model():
    data = pd.read_csv('data/Crop_recommendation.csv')
    data['label'] = LabelEncoder().fit_transform(data['label'])
    X = data.drop(['label'], axis=1)
    Y = data.label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    return classifier

def save_model(model):
    try:
        joblib.dump(model, MODEL_FILE)
    except Exception as e:
        print(f"Error while saving the model: {e}")

def load_model():
    try:
        return joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print("Model file not found. Training and saving a new model.")
        model = train_model()
        save_model(model)
        return model
    except Exception as e:
        print(f"Error while loading the model: {e}")
        return None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    # You can return a custom response or simply a 204 No Content status code
    return make_response("", 204)

@app.route("/home", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            N = float(request.form['Nitrogen'])
            P = float(request.form['Phosphorous'])
            K = float(request.form['Potassium'])
            Temperature = float(request.form['Temperature'])
            Humidity = float(request.form['Humidity'])
            PH = float(request.form['PH'])
            Rainfall = float(request.form['Rainfall'])

            model = load_model()

            if model is not None:
                predict1 = model.predict([[N, P, K, Temperature, Humidity, PH, Rainfall]])
                crop_name = get_crop_name(predict1[0])

                humidity_level = get_level(Humidity, 1, 33, ['Low Humid', 'Medium Humid', 'High Humid'])
                temperature_level = get_level(Temperature, 0, 6, ['Cool', 'Warm', 'Hot'])
                rainfall_level = get_level(Rainfall, 1, 100, ['Less', 'Moderate', 'Heavy Rain'])
                N_level = get_level(N, 1, 50, ['Less', 'Not too less and Not to High', 'High'])
                P_level = get_level(P, 1, 50, ['Less', 'Not too less and Not to High', 'High'])
                K_level = get_level(K, 1, 50, ['Less', 'Not too less and Not to High', 'High'])
                ph_level = get_level(PH, 0, 5, ['Acidic', 'Neutral', 'Alkaline'])

                return render_template("Display.html", cont=[N_level, P_level, K_level, humidity_level,
                                                             temperature_level, rainfall_level, ph_level],
                                       values=[N, P, K, Humidity, Temperature, Rainfall, PH],
                                       cropName=crop_name)
            else:
                return render_template("index.html", error="Model loading failed.")
        except Exception as e:
            return render_template("index.html", error=f"Error occurred: {e}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug= True)
