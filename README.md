
# Profit Prediction

📈 Profit Prediction Web App

This project is a Machine Learning-powered web application that predicts the profit of a startup based on its R&D Spend, Administration Spend, Marketing Spend, and State.
The backend is built with Flask (Python) and a Multiple Linear Regression model, while the frontend is styled using HTML + TailwindCSS for a clean, modern UI.

🔥 Key Features

📊 Machine Learning Model: Multiple Linear Regression trained on startup dataset

⚡ Real-Time Predictions: Enter financial details to instantly calculate expected profit

🎨 Beautiful UI: TailwindCSS + responsive design for smooth experience

🔗 Flask REST API: Backend endpoints serve predictions via JSON

🖥️ Seamless Integration: Frontend form communicates with backend using Fetch API

🛠️ Tech Stack

Backend: Python, Flask, Flask-CORS, NumPy, Joblib

Frontend: HTML5, TailwindCSS, JavaScript (Fetch API)

Model: Multiple Linear Regression

📂 Project Structure

profit-prediction-app/

│── app.py             # Flask backend with prediction API

│── MLR_Model.pkl      # Pre-trained regression model

│── index.html         # Frontend UI with TailwindCSS

│── README.md          # Project documentation

🚀 Getting Started

1️⃣ Clone Repository

git clone https://github.com/your-username/profit-prediction-app.git
cd profit-prediction-app

2️⃣ Create Virtual Environment (recommended)

python -m venv venv

source venv/bin/activate   # On Mac/Linux

venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies

pip install flask flask-cors joblib numpy

4️⃣ Run the Flask Server

python app.py

➡ Server starts at http://127.0.0.1:5000/

5️⃣ Open Frontend

Open index.html directly in your browser

Fill out the form and click Predict Profit

🧪 Example Input & Output

Input:

R&D Spend: 160259.07

Administration Spend: 136897.80

Marketing Spend: 471784.10

State: California

Output:

{

  "predicted_profit": 192261.83

}

📸 Screenshots

Input Form

Prediction Result

⚡ Future Enhancements

✅ Deploy on Heroku/Render/AWS

✅ Add data visualization for financial insights

✅ Expand model with more features (e.g., company size, industry)

✅ Save prediction history in a database

👨‍💻 Developer

Hitesh Bachale

📌 Passionate about Data Science, Machine Learning, Artificial Intelligence, and Web Development — with a keen interest in building predictive models, deploying AI-powered applications, and solving real-world problems using data-driven insights. Enthusiastic about Deep Learning, Natural Language Processing (NLP), and integrating intelligent systems into modern web solutions.
