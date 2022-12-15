import sys
import pymongo
import pandas as pd

def main(args):
    data_ingest()
    if len(args) == 1:
        full_functionality()
    elif len(args) == 2 and args[1] == '--demo':
        demo()
    else:
        print('Usage: python main.py [--demo]')

def demo():
    print('Demo mode. This will not connect to Discord or OpenAI.')
    print('Enter a message to send to the bot. Enter "exit" to quit.')
    while True:
        message = input('Message: ')
        if message == 'exit':
            break
        elif message.startswith('!help'):
            print("I am a discord bot primarily designed to answer questions regarding a netflix dataset"
            "\nI can also generate images based on a prompt. "
            "For example, you can ask me \"!createImage a picture of a dog.\"\n"
            "You can also ask me to use OpenAI's gpt3 model to generate a response to a message by prefixing it with \"!gpt3 \".")
        elif message.startswith('!createImage'):
            print(f'Image of {message[13:]} generated')
        elif message.startswith('!gpt3'):
            print(f'Response generated for {message[6:]}')
        elif message.startswith('!'):
            print('Unknown command. Try !help.')
        else:
            print('Message sent')

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

    # drop columns if not needed
    shows = shows.drop(columns=[])
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

    @client.event
    async def on_member_join(member):
        await member.create_dm()
        await member.dm_channel.send(
            f'Hello {member.name}, welcome to my Discord server!'
        )
        print(f'{member.name} has joined the server')

    # Start the Discord client
    client.run(os.getenv('DISCORD_TOKEN'))

if __name__ == '__main__':
    main(sys.argv)