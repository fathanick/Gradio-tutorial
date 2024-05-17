import gradio as gr
import joblib

def predict_salary(experience):
    model = joblib.load('salary_model.pkl')
    prediction = model.predict([[experience]])[0]
    return float(prediction)  # Convert numpy.float64 to Python float

demo = gr.Interface(
    fn=predict_salary,
    inputs="number",
    outputs="number",
    title="Salary Prediction",
    description="Enter the number of years of experience to predict the salary."
)

demo.launch()