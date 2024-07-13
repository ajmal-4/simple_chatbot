import random
import json
import threading # for implementing parallel processing
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from train import retrain_model  # Ensure retrain_model is imported

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def retrain_in_background():
    print("Retraining model in the background...")
    retrain_model()

def handle_feedback(sentence):
    user_response = input("Can you please provide the correct response? (type 'skip' to skip): ")
    if user_response.lower() != 'skip':
        for intent in intents['intents']:
            if intent["tag"] == "unknown":
                intent["patterns"].append(sentence)
                intent["responses"].append(user_response)
                break
        else:
            new_intent = {
                "tag": "unknown",
                "patterns": [sentence],
                "responses": [user_response]
            }
            intents['intents'].append(new_intent)

        with open('intents.json', 'w') as json_data:
            json.dump(intents, json_data, indent=4)
        print(f"{bot_name}: Thank you! I will remember that.")
        
        threading.Thread(target=retrain_in_background).start()

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence_tokenized = tokenize(sentence)
    X = bag_of_words(sentence_tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.90:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
        handle_feedback(sentence)
