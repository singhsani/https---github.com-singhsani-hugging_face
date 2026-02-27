# Hugging Face 


     Hugging Face ek AI company aur open-source community platform hai jo Machine Learning aur NLP (Natural Language Processing) models ke liye bahut popular hai.

## 🔹 Kya karta hai?
            Pretrained Models provide karta hai
            Jaise:
               1. BERT
               2. GPT-2
               3. LLaMA

## Transformers Library : 
    Python library transformers ke through aap directly models load karke use kar sakte ho.
           1. Datasets & Spaces
           2. Ready-made datasets
           3. Demo apps host karne ke liye Spaces


## Model Hub  Lakhon pretrained models free me available hain.
        . Use Cases
            1. Chatbots
            2. Text summarization
            3. Translation
            4. Image generation


## Speech recognition
        PART 1: Hugging Face Account Kaise Banaye
        🔹 Step 1: Website Open kare
        👉 https://huggingface.co
        🔹 Step 2: Sign Up “Sign Up” par click kare Email + Password enter kare Email verify kare

## PART 2: Access Token Generate Kare (Bahut Important)
        Model upload/download ke liye token chahiye hota hai.
        🔹 Steps:
               -> Profile → Settings
               -> Access Tokens
               -> “New Token” → Role: Write
               -> Token copy karke save kare


## PART 4: Model Kaise Download Kare (3 Methods)
    🔹 Method 1: Transformers Pipeline (Sabse Easy)
        Example: Sentiment Analysis using
        distilbert-base-uncased-finetuned-sst-2-english
         
                ''' from transformers import pipeline
                classifier = pipeline("sentiment-analysis")
                print(classifier("I love India")) '''

## Method 2: Specific Model Load Kare

        Example:
        bert-base-uncased

