import csv
from PyPDF2 import PdfReader
from flask import Flask, redirect, render_template, request, url_for
import os
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Set the directory for storing temporary files
TEMP_DIR = "temp"

if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Load the trained model for categorization
model1 = pickle.load(open('resume_pipeline.pkl', 'rb'))

# Define job categories
job_categories = ['Advocate', 'Arts', 'Automation Testing', 'Blockchain', 'Business Analyst', 'Civil Engineer',
                  'Data Science', 'Database', 'DevOps Engineer', 'DotNet Developer', 'ETL Developer',
                  'Electrical Engineering', 'HR', 'Hadoop', 'Health and fitness', 'Java Developer',
                  'Mechanical Engineer', 'Network Security Engineer', 'Operations Manager', 'PMO',
                  'Python Developer', 'SAP Developer', 'Sales', 'Testing', 'Web Designing']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/post', methods=['GET', 'POST'])
def index2():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']

        # Save the uploaded file to a temporary location
        pdf_file_path = os.path.join(TEMP_DIR, pdf_file.filename)
        pdf_file.save(pdf_file_path)

        with open(pdf_file_path, "rb") as file:
            pdf_reader = PdfReader(file)

            # Write the text to a CSV file
            csv_file_path = "resumes.csv"
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                for i in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[i]
                    text = page.extract_text()
                    writer.writerow([text])

    # Redirect to the screen route after uploading resumes
    return redirect(url_for('screen'))

@app.route('/innerpage', methods=['GET', 'POST'])
def innerpage():
    return render_template('inner-page.html')

@app.route('/screen', methods=['GET', 'POST'])
def screen():
    if request.method == 'POST':
        # Assuming 'category' is passed as form data
        desired_category = request.form.get('category')

        # Fetch resumes related to the desired category
        cat_df = pd.read_csv('resumes.csv')  # Load resumes from CSV

        # Select resumes related to the desired category
        cat_df = cat_df[cat_df['Predicted_Category'] == desired_category]

        # Fetch the resumes text
        resumes = cat_df['Resume'].tolist()
        # Predict probabilities of input belonging to each category
        proba = model1.pr(resumes)

        # Get the probability of the desired category
        desired_proba = proba[:, job_categories.index(desired_category)]
        
        # Sort the indices of the input by their probability of belonging to the desired category
        ranked_indices = np.argsort(desired_proba)[::-1]

        # Initialize a list to hold ranked resumes
        ranked_resumes = []

        # Print the ranked indices and their corresponding probabilities
        for i, idx in enumerate(ranked_indices):
            ranked_resumes.append((resumes[idx], desired_proba[idx]))

        return render_template('ranked-resumes.html', ranked_resumes=ranked_resumes)

    else:
        data = pd.read_csv('resumes.csv')

        predicted_categories = []
        for index, row in data.iterrows():
            output1 = model1.predict(row)
            predicted_category = job_categories[output1[0]]
            predicted_categories.append(predicted_category)

        # Count the frequency of each job category
        category_counts = pd.Series(predicted_categories).value_counts()

        # Sort the job categories by frequency in descending order
        ranked_categories = category_counts.index.tolist()

        return render_template('ranked-categories.html', ranked_categories=ranked_categories)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
