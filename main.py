import sys

def main(args):
    if len(args) == 1:
        full_functionality()
    elif len(args) == 2 and args[1] == '--demo':
        demo()
    else:
        print('Usage: python main.py [--demo]')

def demo():
    pass

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
        if message.content.startswith("!help"):
            await message.channel.send("I am a discord bot primarily designed to answer questions regarding a netflix dataset"
            "\nI can also generate images based on a prompt. "
            "For example, you can ask me \"!createImage a picture of a dog.\"\n"
            "You can also ask me to use openai's gpt3 model to generate a response to a message by prefixing it with \"!gpt3 \".")
            return
        if message.content.startswith("!createImage"):
            image = openai.Image.create(
                prompt=message.content[13:],
                n=1,
                size="1024x1024"
            )
            # download the image and send it
            try:
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
                prompt=f"{message.author} said: {message.content}\nBot response:",
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
        if message.content.startswith("!"):
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