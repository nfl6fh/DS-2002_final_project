# DS-2002_final_project
## Nathan Lindley

This is my final project for DS 2002 in the Fall 2022 semester. It is a discord bot that can be asked questions about a netflix dataset and will respond using queries of a MongoDB. It also has the ability to respond to user questions with answers generated by openai's gpt-3 algorithm, as well as generate images from user prompts.

## How to run
1. Clone the repository
2. Create a file called .env in the root directory of the project
3. Add the following to the .env file
```
DISCORD_TOKEN=<your discord bot token>
MONGO_URI=<your mongo uri>
OPENAI_API_TOKEN=<your openai api token>
```
4. Ensure you have the following python packages installed
```
discord.py
pymongo
openai
requests
dotenv
```
5. Run the main.py file with `python main.py`
6. Invite the bot to your server with admin permissions (You can give it less permissions if you want, but it will not work as intended without a change to the intents in main.py)