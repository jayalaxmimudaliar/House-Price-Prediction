from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    bedrooms = sorted(data['BHK'].unique())
    bathrooms = sorted(data['bath'].unique())
    sizes = sorted(data['total_sqft'].unique())
    location = sorted(data['location'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, location=location)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('BHK')
    bathrooms = request.form.get('bath')
    size = request.form.get('total_sqft')
    location = request.form.get('location')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, location]],columns=['BHK', 'bath', 'total_sqft', 'location'])

   
    
    prediction = pipe.predict(input_data)[0]

    return str(prediction)
if __name__ == "__main__":
    app.run(debug=True, port=5001)
