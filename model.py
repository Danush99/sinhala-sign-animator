from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging
import fasttext
import nltk
import string
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from google.cloud import translate_v2 as translate
from IPython.display import Video, display
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

rf = joblib.load('rf.pkl') 

model = fasttext.load_model('cc.en.300.bin')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'translate-key.json' 

def get_embedding(word):
    return model.get_word_vector(word)


def preprocess_sentence(sentence):
    sentence = sentence.lower()

    words = sentence.split()

    stop_words = set(stopwords.words('english'))

    filtered_words = [word for word in words if word not in stop_words]

    filtered_sentence = ' '.join(filtered_words)

    return filtered_sentence


def move_verb_to_end(sentence):
    words = nltk.word_tokenize(sentence)

    pos_tags = nltk.pos_tag(words)

    verb = None
    verb_index = None
    for i, (word, tag) in enumerate(pos_tags):
        if tag.startswith('VB'):  # Check if the tag is a verb (VB, VBD, VBG, VBN, VBP, VBZ)
            verb = word
            verb_index = i
            break

    if verb is not None:
        words.pop(verb_index)
        words.append(verb)

    modified_sentence = ' '.join(words)

    return modified_sentence

def translate_text(text, target_language='en'):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def move_verb_to_end(sentence):
    words = nltk.word_tokenize(sentence)

    pos_tags = nltk.pos_tag(words)

    verb = None
    verb_index = None
    for i, (word, tag) in enumerate(pos_tags):
        if tag.startswith('VB'):  # Check if the tag is a verb (VB, VBD, VBG, VBN, VBP, VBZ)
            verb = word
            verb_index = i
            break

    if verb is not None:
        words.pop(verb_index)
        words.append(verb)

    modified_sentence = ' '.join(words)

    return modified_sentence

def remove_punctuation(sentence):
    translator = str.maketrans('', '', string.punctuation)

    cleaned_sentence = sentence.translate(translator)

    return cleaned_sentence

def preprocess_and_predict(sentences):
    processed_sentences = []
    for sentence in sentences:
        sentence = preprocess_sentence(sentence)
        sentence = move_verb_to_end(sentence)
        sentence = remove_punctuation(sentence)
        processed_sentences.append(sentence)

    # Split each processed sentence into words and get embeddings
    predictions = []
    for sentence in processed_sentences:
        words = sentence.split()
        word_embeddings = [get_embedding(word) for word in words]

        # Convert embeddings to DataFrame
        embeddings_df = pd.DataFrame(word_embeddings)

        # Predict the sign for each word's embedding
        sentence_predictions = rf.predict(embeddings_df)
        predictions.append(sentence_predictions)
    print(predictions)

    return predictions

video_folder_path = '/Signs/'

def generate_sign_language_video(sign_keys):
    video_files = [video_folder_path +  key + '.mkv' for key in sign_keys]
    valid_clips = []

    for video_file in video_files:
        if os.path.exists(video_file):
            try:
                clip = VideoFileClip(video_file)
                valid_clips.append(clip)
            except Exception as e:
                print(f"Error loading video {video_file}: {e}")
        else:
            print(f"Video file {video_file} does not exist")

    if not valid_clips:
        raise ValueError("No valid video clips found")

    # Concatenate the valid clips
    final_clip = concatenate_videoclips(valid_clips)

    # Define the path for the output video in Google Drive
    output_video_path = '/output/final_output.mp4' 
    
    # Write the result to the file in Google Drive
    final_clip.write_videofile(output_video_path)
    
    # Display the video
    return output_video_path
    


