import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from bot import telegram_chatbot
import json
from bot import telegram_chatbot

bot = telegram_chatbot("config.cfg")
update_id = None

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

reply = None
from_ = None
wr = None
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def make_reply(msg):
    if msg is not None:
        reply = str(msg)
    return reply


while True:
    updates = bot.get_updates(offset=update_id)
    updates = updates["result"]

    if updates:
        for item in updates:
            update_id = item["update_id"]
            from_ = item["message"]["from"]["id"]
            sentence = str(item["message"]["text"])
            if sentence == "quit":
                break

            sentence = tokenize(sentence)
            X = bag_of_words(sentence, all_words)
            X = X.reshape(1, X.shape[0])
            X = torch.from_numpy(X).to(device)

            output = model(X)
            _, predicted = torch.max(output, dim=1)

            tag = tags[predicted.item()]

            probs = torch.softmax(output, dim=1)
            prob = probs[0][predicted.item()]
            print(prob.item())
            if prob.item() > 0.75:
                for intent in intents['intents']:
                    if tag == intent["tag"]:

                        reply = make_reply(f"{random.choice(intent['responses'])}")

            else:
                reply = make_reply("I do not understand...")
                print(f"{bot_name}: I do not understand...")

            bot.send_message(reply, from_)
