from dotenv import load_dotenv

load_dotenv() ## load the environment variables from the .env file

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai 

genai.configure(api_key=os.getenv('GENAI_API_KEY')) ## set the API key

## Function to load Gemini Pro Vision
model=genai.GenerativeModel('gemini-pro-vision')   

def get_gemini_response(input, image, prompt):
    response = model.generate_content([input, image[0], prompt])
    return response.text

def input_image_details(uploaded_file):
    if uploaded_file is not None:
        # Read the file into bytes
        
        bytes_data = uploaded_file.getvalue()
        
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded") 

## Initializing the Streamlit app

st.set_page_config(page_title='Multilanguage Invoice Extractor', page_icon='🔮', layout='centered', initial_sidebar_state='expanded')
input = st.text_input("Input prompt:", key=input)
uploaded_file = st.file_uploader("Choose an Image....", type=["jpg", "jpeg", "png"]) 

image = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)   
    
submit = st.button("Tell me about the invoice")

input_prompt = """
You are a expert in understanding invoices. We will upload an invoice image and you will tell us about the invoice. 
And you will have to answer any questions that we ask about the invoice.
"""

# If submit button is clicked
if submit:
    image_data = input_image_details(uploaded_file) 
    response = get_gemini_response(input_prompt, image_data, input)
    st.subheader("The Response is")
    st.write(response) 