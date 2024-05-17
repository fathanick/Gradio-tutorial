import gradio as gr
import joblib

# Load the model
model = joblib.load('student_score_model.pkl')

def predict_score(hours):
    hours = float(hours)
    score_prediction = round(model.predict([[hours]])[0], 2)

    return score_prediction

# Create a Gradio interface
iface = gr.Interface(fn=predict_score, 
             inputs="text", 
             outputs=gr.Textbox(label="Predicted Score"))

iface.launch()