import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('mlp_classifier_model.pkl')

# Define the feature names and their corresponding questions
feature_questions = {
    'Login Frequency': 'How often do you log into Canvas this summer?',
    'Time Spent on Canvas': 'How much time do you spend on Canvas weekly for internship-related activities?',
    'Accessed Optional Module': 'Have you accessed the Optional Continuing Education module in the Canvas Open Learning page before?',
    'Check for Emails': 'How often do you check for emails and announcements?',
    'Timely Submission': 'How often do you submit the Canvas activities on time?',
    'Time on Client Project': 'How much time do you spend weekly on your client project?',
    'Time on SMART Goals': 'How much time do you spend weekly working on the three SMART goals set at the beginning of the internship?',
    'Communication with Coordinator': 'How often do you communicate with your ATLAS coordinator?',
    'Communication with Client': 'How often do you communicate with your client?',
    'Communication with Interns': 'How often do you communicate and collaborate with your coworkers or other interns on projects?',
    'Internship Meaningful': 'How often do you feel that your internship tasks are meaningful and impactful?',
    'Internship Difficulty': 'How often do you feel challenged by your internship tasks?',
    'Learned From Internship': 'On a scale of 1 to 5, how much have you learned from your internship so far?',
    'Satisfaction With Support': 'On a scale of 1 to 5, how satisfied are you with the level of support and resources provided during your internship?'
}

# Define the scales for each feature
feature_scales = {
    'Login Frequency': (1, 5),
    'Time Spent on Canvas': (1, 5),
    'Accessed Optional Module': (0, 1),
    'Check for Emails': (1, 5),
    'Timely Submission': (1, 4),
    'Time on Client Project': (1, 5),
    'Time on SMART Goals': (1, 5),
    'Communication with Coordinator': (1, 5),
    'Communication with Client': (1, 5),
    'Communication with Interns': (1, 5),
    'Internship Meaningful': (1, 4),
    'Internship Difficulty': (1, 4),
    'Learned From Internship': (1, 5),
    'Satisfaction With Support': (1, 5)
}

# Define the main function for the app
def main():
    st.title('Student Engagement Level Prediction')
    st.write('Enter the features to get an evaluation of your engagement level.')

    # Create input fields for each feature
    feature_values = []
    for feature, question in feature_questions.items():
        min_val, max_val = feature_scales[feature]
        value = st.slider(question, min_value=min_val, max_value=max_val, step=1)
        feature_values.append(value)

    # Convert the input to a DataFrame
    input_data = pd.DataFrame([feature_values], columns=feature_questions.keys())

    # Predict the engagement level
    if st.button('Predict'):
        prediction = model.predict(input_data)
        st.write(f'Predicted Engagement Level: {prediction[0]}')

# Run the app
if __name__ == '__main__':
    main()