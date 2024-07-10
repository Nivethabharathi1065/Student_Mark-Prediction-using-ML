from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import plotly.graph_objs as go
from sklearn.metrics import r2_score
import numpy as np

app = Flask(__name__)

# Load the dataset
df = pd.read_csv("stu1.csv")

# Features and target variables
X = df.drop(['CGPA'], axis=1)  # Assuming 'CGPA' is the target variable
y = df['CGPA']  # Assuming 'CGPA' is the target variable

# Define categorical features and numeric features
cat_features = ['Group', 'Parental_Level_of_Education', 'Standard/Assisted_Achievement_Program', 'Online_Course_Preparation', 'Student_Degree']
num_features = ['Academic_Score', 'Attendance_Score', 'Hackathon_Score']

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Preprocess data
X_preprocessed = preprocessor.fit_transform(X)

# Train models for each target
lr_models = {}
knn_models = {}
target_features = ['Academic_Score', 'Attendance_Score', 'Hackathon_Score']

for feature in target_features:
    lr_model = LinearRegression()
    knn_model = KNeighborsRegressor()
    lr_model.fit(X_preprocessed, df[feature])
    knn_model.fit(X_preprocessed, df[feature])
    lr_models[feature] = lr_model
    knn_models[feature] = knn_model

# Train CGPA models
lr_cgpa_model = LinearRegression()
knn_cgpa_model = KNeighborsRegressor()
lr_cgpa_model.fit(X_preprocessed, y)
knn_cgpa_model.fit(X_preprocessed, y)

# Calculate R² score for the Linear Regression model using the entire dataset
lr_pred_cgpa = lr_cgpa_model.predict(X_preprocessed)
lr_r2_score = r2_score(y, lr_pred_cgpa)

# Calculate R² score for the KNN model using the entire dataset
knn_pred_cgpa = knn_cgpa_model.predict(X_preprocessed)
knn_r2_score = r2_score(y, knn_pred_cgpa)

def predict_scores(input_data):
    input_preprocessed = preprocessor.transform(input_data)
    predictions = {'lr': {}, 'knn': {}}
    for feature in target_features:
        predictions['lr'][feature] = lr_models[feature].predict(input_preprocessed)[0]
        predictions['knn'][feature] = knn_models[feature].predict(input_preprocessed)[0]
    predictions['lr']['CGPA'] = lr_cgpa_model.predict(input_preprocessed)[0]
    predictions['knn']['CGPA'] = knn_cgpa_model.predict(input_preprocessed)[0]
    
    return predictions

def get_feedback(cgpa):
    if cgpa >= 8.5:
        return "Congratulations on your excellent CGPA! Keep up the good work and continue to excel in your studies."
    elif 7.0 <= cgpa < 8.5:
        return "You're doing well, but there's always room for improvement. Consider joining extracurricular activities or seeking additional academic support to enhance your skills."
    else:
        return "Don't be discouraged by your current CGPA. Identify areas where you can improve, such as attending study groups, seeking help from professors, or participating in academic workshops."



@app.route('/')
def index():
    return render_template('predictor_form.html')

@app.route('/result', methods=['POST'])
def result():
    try:
        group = request.form['group']
        education = request.form['education']
        program = request.form['program']
        course_prep = request.form['course_prep']
        academic_score = float(request.form['academic_score'])
        attendance_score = float(request.form['attendance_score'])
        hackathon_score = float(request.form['hackathon_score'])
        student_degree = request.form['student_degree']
        
        input_data = pd.DataFrame({
            'Group': [group],
            'Parental_Level_of_Education': [education],
            'Standard/Assisted_Achievement_Program': [program],
            'Online_Course_Preparation': [course_prep],
            'Student_Degree': [student_degree],
            'Academic_Score': [academic_score],
            'Attendance_Score': [attendance_score],
            'Hackathon_Score': [hackathon_score]
        })
        
        predictions = predict_scores(input_data)

        # Generate feedback based on predicted CGPA
        lr_feedback = get_feedback(predictions['lr']['CGPA'])
        knn_feedback = get_feedback(predictions['knn']['CGPA'])

        # Generate Plotly graph for predicted CGPA values from LR and KNN
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['CGPA'], y=[predictions['lr']['CGPA']], name='LR Predicted CGPA'))
        fig.add_trace(go.Bar(x=['CGPA'], y=[predictions['knn']['CGPA']], name='KNN Predicted CGPA'))

        fig.update_layout(title='Predicted CGPA from LR and KNN Models', barmode='group', xaxis_title='Model', yaxis_title='Predicted CGPA')

        graph_html = fig.to_html(full_html=False)

        return render_template('result_form.html',
                               group=group,
                               education=education,
                               program=program,
                               course_prep=course_prep,
                               academic_score=academic_score,
                               attendance_score=attendance_score,
                               hackathon_score=hackathon_score,
                               student_degree=student_degree,
                               academic_lr=predictions['lr']['Academic_Score'],
                               attendance_lr=predictions['lr']['Attendance_Score'],
                               hackathon_lr=predictions['lr']['Hackathon_Score'],
                               cgpa_lr=predictions['lr']['CGPA'],
                               academic_knn=predictions['knn']['Academic_Score'],
                               attendance_knn=predictions['knn']['Attendance_Score'],
                               hackathon_knn=predictions['knn']['Hackathon_Score'],
                               cgpa_knn=predictions['knn']['CGPA'],
                               lr_r2_score=lr_r2_score,
                               knn_r2_score=knn_r2_score,
                               lr_feedback=lr_feedback,  # Pass LR feedback to the template
                               knn_feedback=knn_feedback,  # Pass KNN feedback to the template
                               graph_html=graph_html)
    except Exception as e:
        print("Error:", e)
        return render_template('result_form.html', error_message="An error occurred while processing the request. Please try again.")


@app.route('/visualization')
def visualization():
    try:
        # Calculate the count of each degree in the dataset
        degree_counts = df['Student_Degree'].value_counts()

        # Generate plotly figure
        fig = go.Figure(data=[go.Bar(x=degree_counts.index, y=degree_counts.values)])
        fig.update_layout(title='Presence of Students by Degree', xaxis_title='Student Degree', yaxis_title='Count')

        # Render the plotly figure to HTML
        graph_html = fig.to_html(full_html=False)

        return render_template('visualization.html', graph_html=graph_html)
    except Exception as e:
        print("Error:", e)
        return render_template('visualization.html', error_message="An error occurred while processing the request. Please try again.")

@app.route('/marks_vs_cgpa')
def marks_vs_cgpa():
    try:
        # Create scatter plots for Academic_Score, Attendance_Score, and Hackathon_Score vs CGPA
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['Academic_Score'], y=df['CGPA'], mode='markers', name='Academic Score vs CGPA'))
        fig.add_trace(go.Scatter(x=df['Attendance_Score'], y=df['CGPA'], mode='markers', name='Attendance Score vs CGPA'))
        fig.add_trace(go.Scatter(x=df['Hackathon_Score'], y=df['CGPA'], mode='markers', name='Hackathon Score vs CGPA'))

        fig.update_layout(title='Scores vs CGPA', xaxis_title='Scores', yaxis_title='CGPA')

        # Render the plotly figure to HTML
        graph_html = fig.to_html(full_html=False)

        return render_template('marks_vs_cgpa.html', graph_html=graph_html)
    except Exception as e:
        print("Error:", e)
        return render_template('marks_vs_cgpa.html', error_message="An error occurred while processing the request. Please try again.")

@app.route('/marks_and_attendance')
def marks_and_attendance():
    try:
        # Create scatter plot for Academic_Score vs Attendance_Score
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['Academic_Score'], y=df['Attendance_Score'], mode='markers', name='Academic Score vs Attendance Score'))

        fig.update_layout(title='Academic Score vs Attendance Score', xaxis_title='Academic Score', yaxis_title='Attendance Score')

        # Render the plotly figure to HTML
        graph_html = fig.to_html(full_html=False)

        return render_template('marks_and_attendance.html', graph_html=graph_html)
    except Exception as e:
        print("Error:", e)
        return render_template('marks_and_attendance.html', error_message="An error occurred while processing the request. Please try again.")

if __name__ == '__main__':
    app.run(debug=False)  # Set debug to False
