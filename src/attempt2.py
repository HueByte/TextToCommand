import json
import os
import time
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re

commands = {
    "weather": {
        "syntax": "$weather <location>",
        "description": "Command for fetching weather for a specific location on a specific date",
        "parameters": [
            {"name": "location", "description": "The city or place to get the weather for"},
        ]
    },
    "news": {
        "syntax": "$news <topic>",
        "description": "Command for fetching news on a specific topic",
        "parameters": [
            {"name": "topic", "description": "The topic to fetch news about"}
        ]
    },
    "get": {
        "syntax": "$get <url>",
        "description": "Command for fetching a specific URL from the web",
        "parameters": [
            {"name": "url", "description": "The URL used to download the file from the web"}
        ]
    },
    "line": {
        "syntax": "$line <from> <to> <color>",
        "description": "Command for drawing a line from one point to another with a specific color",
        "parameters": [
            {"name": "from",
                "description": "The starting point of the line, , expressed as a coordinate pair (x, y)"},
            {"name": "to",
                "description": "The ending point of the line, expressed as a coordinate pair (x, y)"},
            {"name": "color", "description": "The color of the line, either in the hex format like #000000 or a color name"}
        ]
    },
    "time": {
        "syntax": "$time <location>",
        "description": "Command for fetching the current time for a specific location",
        "parameters": [
            {"name": "location",
                "description": "The city or place to get the current time for"}
        ]
    },
    "translate": {
        "syntax": "$translate <text> <language>",
        "description": "Command for translating a given text to a specified language",
        "parameters": [
            {"name": "text", "description": "The text to translate"},
            {"name": "language", "description": "The language to translate the text into"}
        ]
    },
    "stock": {
        "syntax": "$stock <symbol>",
        "description": "Command for fetching the current stock price for a specific stock symbol",
        "parameters": [
            {"name": "symbol", "description": "The stock symbol to fetch the price for"}
        ]
    },
    "convert": {
        "syntax": "$convert <amount> <from_currency> <to_currency>",
        "description": "Command for converting a specific amount from one currency to another",
        "parameters": [
            {"name": "amount", "description": "The amount of money to convert"},
            {"name": "from_currency", "description": "The currency to convert from"},
            {"name": "to_currency", "description": "The currency to convert to"}
        ]
    },
    "define": {
        "syntax": "$define <word>",
        "description": "Command for fetching the definition of a specific word",
        "parameters": [
            {"name": "word", "description": "The word to define"}
        ]
    },
    "joke": {
        "syntax": "$joke",
        "description": "Command for fetching a random joke",
        "parameters": []
    },
    "fact": {
        "syntax": "$fact",
        "description": "Command for fetching a random fact",
        "parameters": []
    },
    "reminder": {
        "syntax": "$reminder <time> <message>",
        "description": "Command for setting a reminder with a specific message at a specific time",
        "parameters": [
            {"name": "time", "description": "The time to set the reminder for"},
            {"name": "message", "description": "The message for the reminder"}
        ]
    },
    "todo": {
        "syntax": "$todo <action> <task>",
        "description": "Command for managing a to-do list, with actions like add, remove, and list tasks",
        "parameters": [
            {"name": "action",
                "description": "The action to perform on the to-do list (add, remove, list)"},
            {"name": "task", "description": "The task to add or remove from the to-do list"}
        ]
    }
}


# Load fine-tuned NER model
# model_name = 'tner/roberta-large-ontonotes5'
model_name = 'ner-english-ontonotes-large'

# Load NER model from Hugging Face
ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")

# Load pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_most_relevant_command(user_prompt, commands):
    user_prompt = preprocess(user_prompt)

    command_descriptions = [cmd['description'] for cmd in commands.values()]
    all_texts = [user_prompt] + command_descriptions

    embeddings = sbert_model.encode(all_texts)
    user_embedding = embeddings[0]
    command_embeddings = embeddings[1:]

    similarities = util.cos_sim(user_embedding, command_embeddings)
    most_relevant_idx = np.argmax(similarities)

    command_key = list(commands.keys())[most_relevant_idx]
    return commands[command_key]


def extract_entities(user_prompt):
    entities = ner_pipeline(user_prompt)
    extracted_parameters = {}

    current_entity = None
    current_label = None

    for entity in entities:
        entity_word = entity['word'].strip()
        entity_label = entity['entity_group']

        if entity_label.startswith("B-"):
            if current_entity is not None:
                # Store the previous entity
                if current_label not in extracted_parameters:
                    extracted_parameters[current_label] = []
                extracted_parameters[current_label].append(current_entity)

            # Start a new entity
            current_entity = entity_word
            current_label = entity_label[2:]  # Remove the B- prefix

        elif entity_label.startswith("I-") and current_label == entity_label[2:]:
            # Continue the current entity
            current_entity += " " + entity_word

        else:
            if current_entity is not None:
                # Store the previous entity
                if current_label not in extracted_parameters:
                    extracted_parameters[current_label] = []
                extracted_parameters[current_label].append(current_entity)

            # Reset current entity and label
            current_entity = None
            current_label = None

            if entity_label == "O":
                continue
            else:
                # Start a new entity
                current_entity = entity_word
                current_label = entity_label[2:]  # Remove the I- prefix

    # Store the last entity
    if current_entity is not None and current_label is not None:
        if current_label not in extracted_parameters:
            extracted_parameters[current_label] = []
        extracted_parameters[current_label].append(current_entity)

    return extracted_parameters


def refine_parameter_with_sbert(user_prompt, extracted_parameter, parameter_name, parameter_description):
    # Creating the prompt for refinement
    prompt = f"User prompt: {user_prompt}\nParameter name: {parameter_name}\nParameter description: {parameter_description}\nExtracted parameter: {extracted_parameter}\nIs this parameter correct?"

    prompt_embedding = sbert_model.encode([prompt])
    parameter_embedding = sbert_model.encode([parameter_description])

    similarity = util.cos_sim(prompt_embedding, parameter_embedding)

    # Assuming similarity above a certain threshold indicates a valid parameter
    if similarity[0][0] > 0.5:
        return extracted_parameter
    else:
        return ""  # Return empty if the parameter is not valid


def match_parameters_with_sbert(user_prompt, extracted_parameters, command_metadata):
    matched_parameters = {}

    for parameter in command_metadata['parameters']:
        parameter_name = parameter['name']
        parameter_description = parameter['description']

        for entity_group, entity_values in extracted_parameters.items():
            for entity_value in entity_values:
                refined_parameter = refine_parameter_with_sbert(
                    user_prompt, entity_value, parameter_name, parameter_description)
                if refined_parameter:
                    matched_parameters[parameter_name] = refined_parameter
                    break

    return matched_parameters


def generate_command(user_prompt, commands):
    print(f'User prompt: {user_prompt}')
    relevant_command = get_most_relevant_command(user_prompt, commands)

    command_metadata = {
        'syntax': relevant_command['syntax'],
        'description': relevant_command['description'],
        'parameters': relevant_command['parameters']
    }

    extracted_entities = extract_entities(user_prompt)
    print(f'Extracted entities: {extracted_entities}')

    matched_parameters = match_parameters_with_sbert(
        user_prompt, extracted_entities, command_metadata)

    final_command = command_metadata["syntax"]
    for parameter in command_metadata['parameters']:
        parameter_name = parameter['name']
        refined_parameter = matched_parameters.get(parameter_name, "")
        final_command = final_command.replace(
            f"<{parameter_name}>", refined_parameter)

    return final_command


start = time.time()
prompts = [
    "What is the weather like in New York?",
    "Show me the latest news about technology.",
    "Download the file from https://example.com/document.pdf",
    "Draw a line from (50, 50) to (150, 150) with blue color.",
    "What time is it in Tokyo right now?",
    "Translate 'Hello, how are you?' to Spanish.",
    "Get the current stock price for AAPL.",
    "Convert 100 USD to EUR.",
    "Define the word 'serendipity'.",
    "Tell me a joke.",
    "Give me a random fact.",
    "Set a reminder for 8:00 PM to call mom.",
    "Add 'buy groceries' to my to-do list.",
    "Remove 'buy groceries' from my to-do list.",
    "List all tasks in my to-do list.",
]

for prompt in prompts:
    generated_command = generate_command(prompt, commands)
    print(f"Generated command: {generated_command}\n")

end = time.time()
print(f'{round(end - start, 3)}s')
