import streamlit as st
import joblib

# Load the trained model
model = joblib.load('../model/I1.pkl')

# Streamlit UI
def main():
    st.title('Spam Detector')

    # User input
    user_input = st.text_area('Enter an email text:')
    if st.button('Predict'):
        # Directly use the user input for prediction
        prediction = model.predict([user_input])
        result = 'spam' if prediction[0] == 1 else 'ham'
        st.write(f'The predicted label is: {result}')

if __name__ == '__main__':
    main()
