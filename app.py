from fastapi import FastAPI, Request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.initializers import Orthogonal  # Import the initializer
import pickle

# Define custom objects for loading the model
custom_objects = {"Orthogonal": Orthogonal}

# Load the saved model with custom_objects
model = load_model("exhaustion_model.h5", custom_objects=custom_objects)

# Load the tokenizer (if used during training)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define the FastAPI app
app = FastAPI()

# Define the prediction endpoint


@app.post("/predict")
async def predict(request: Request):
    # Get the input text from the request
    data = await request.json()
    text = data.get("text")

    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(
        sequence, maxlen=50)  # Adjust maxlen as needed

    # Make the prediction
    prediction = model.predict(padded_sequence)
    # Rescale to original range (if normalized)
    exhaustion_score = prediction[0][0] * 10

    # Return the prediction as JSON
    return {"exhaustion_score": float(exhaustion_score)}

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
