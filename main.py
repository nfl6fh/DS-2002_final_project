import sys
import pymongo
import pandas as pd

# BEGIN: From in-class demo
import nltk
nltk.download('punkt')
from nltk import word_tokenize,sent_tokenize

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np 
import tflearn
import tensorflow as tf
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_x = []
    docs_y = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])


    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
               bag.append(1)
            else:
              bag.append(0)
    
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        
        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)

# END: From in-class demo

def main(args):
    data_ingest()
    if len(args) == 1:
        full_functionality()
    elif len(args) == 2 and args[1] == '--demo':
        demo()
    else:
        print('Usage: python main.py [--demo]')

def demo():
    print()
    print('Demo mode. This will not connect to Discord or OpenAI.')
    print('Enter a message to send to the bot. Enter "exit" to quit.')
    while True:
        message = input('Message: ')
        if message == 'exit':
            print('Goodbye!')
            break
        elif message.startswith('!help'):
            print("I am a discord bot primarily designed to answer questions regarding the past 5 years in a netflix dataset\n"
            "Things you can ask me include:\n"
            "1. \"What is the best show on netflix?\"\n"
            "2. \"What is the best show on netflix in [2017-2021]?\"\n"
            "3. \"What is the best movie on netflix?\"\n"
            "4. \"What is the best movie on netflix in [2017-2021]?\"\n"
            "5. \"What were the top 5 shows on netflix in [2017-2021]?\"\n"
            "6. \"What were the top 5 movies on netflix in [2017-2021]?\"\n"
            "7. \"What is the average rating of the top 5 shows on netflix in [2017-2021]?\"\n"
            "8. \"What is the average rating of the top 5 movies on netflix in [2017-2021]?\"\n"
            "9. \"What is the average rating of the top 5 shows on netflix?\"\n"
            "10. \"What is the average rating of the top 5 movies on netflix?\"\n"
            "\nI can also generate images based on a prompt. "
            "For example, you can ask me \"!createImage a picture of a dog.\"\n"
            "You can also ask me to use OpenAI's gpt3 model to generate a response to a message by prefixing it with \"!gpt3 \".")
        elif message.startswith('!createImage'):
            print(f'Image of "{message[13:]}" generated')
        elif message.startswith('!gpt3'):
            print(f'Response generated for "{message[6:]}"')
        elif message.startswith('!'):
            print('Unknown command. Try !help.')
        else:
            print(f'Response: {respond(message)}')

def data_ingest():
    print('Ingesting data...')
    # Connect to MongoDB
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client['netflix']
    # Ingest data
    shows = pd.read_csv('data/Best Shows Netflix.csv', index_col=0)
    show_yr = pd.read_csv('data/Best Show by Year Netflix.csv', index_col=0)
    movies = pd.read_csv('data/Best Movies Netflix.csv', index_col=0)
    movie_yr = pd.read_csv('data/Best Movie by Year Netflix.csv', index_col=0)
    credits = pd.read_csv('data/raw_credits.csv', index_col=0)
    titles = pd.read_csv('data/raw_titles.csv', index_col=0)
    dfs = [shows, show_yr, movies, movie_yr, credits, titles]

    # remove irrelevant data
    shows = shows.where(shows.RELEASE_YEAR >= 2017).dropna()
    movies = movies.where(movies.RELEASE_YEAR >= 2017).dropna()
    show_yr = show_yr.where(show_yr.RELEASE_YEAR >= 2017).dropna()
    movie_yr = movie_yr.where(movie_yr.RELEASE_YEAR >= 2017).dropna()
    titles = titles.where(titles.release_year >= 2017).dropna()
    return

def full_functionality():
    import openai
    import discord
    import requests
    import os
    from dotenv import load_dotenv
    load_dotenv()

    # Set up OpenAI API key
    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Set up Discord client
    client = discord.Client(intents=discord.Intents.all())

    @client.event
    async def on_message(message):
        # Ignore messages from the bot itself
        if message.author == client.user:
            return
        print(f'message:\n    author: "{message.author}",\n    content: "{message.content}",\n    channel: "{message.channel}",')
        if message.content == "":
            return
        elif message.content.startswith("!help"):
            await message.channel.send("I am a discord bot primarily designed to answer questions regarding a netflix dataset"
            "\nI can also generate images based on a prompt. "
            "For example, you can ask me \"!createImage a picture of a dog.\"\n"
            "You can also ask me to use OpenAI's gpt3 model to generate a response to a message by prefixing it with \"!gpt3 \".")
            return
        elif message.content.startswith("!createImage"):
            try:
                image = openai.Image.create(
                    prompt=message.content[13:],
                    n=1,
                    size="1024x1024"
                )
                # download the image and send it
                image_r = requests.get(image['data'][0]['url'], allow_redirects=True).content
                open('temp_image.png', 'wb').write(image_r)
                await message.channel.send(file=discord.File('temp_image.png'))
                # delete the image
                os.remove('temp_image.png')
                # await message.channel.send(image['data'][0]['url'])
                print('    image: "' + image['data'][0]['url'] + '"')
            except:
                await message.channel.send("I couldn't generate an image for that prompt.")
                print('    image: "None"')
            return
        elif message.content.startswith("!gpt3"):
            # Use GPT-3 to generate a response to the user's message if they ask for it
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"{message.author} said: {message.content[6:]}\nBot response:",
                temperature=0.9,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
            )
            # Send the response to the Discord channel
            reply = response["choices"][0]["text"].strip()
            if reply:
                await message.channel.send(reply)
                print(f'    reply: "{reply}"')
            else:
                await message.channel.send("I don't understand")
                print(f'    reply: "I don\'t understand"')
        elif message.content.startswith("!"):
            await message.channel.send("I don't understand that command. Try !help.")
            return
        else:
            # Use the model to generate a response to the user's message
            reply = respond(message.content)
            await message.channel.send(reply)
            print(f'    reply: "{reply}"')

    @client.event
    async def on_member_join(member):
        await member.create_dm()
        await member.dm_channel.send(
            f'Hello {member.name}, welcome to my Discord server!'
        )
        print(f'{member.name} has joined the server')

    # Start the Discord client
    print('Starting Discord client...')
    client.run(os.getenv('DISCORD_TOKEN'))

def respond(query):
    result = model.predict([bag_of_words(query, words)])[0]
    result_index = np.argmax(result)
    tag = labels[result_index]

    if result[result_index] > 0.7:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        return (random.choice(responses))

    else:
        return ("I didnt get that. Can you explain or try again.")

if __name__ == '__main__':
    main(sys.argv)