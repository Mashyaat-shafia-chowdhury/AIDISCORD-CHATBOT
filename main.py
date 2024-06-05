import discord
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
from neuralintents.assistants import BasicAssistant

chatbot = BasicAssistant('intents.json')


import discord
import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0
y_train, y_val, y_test = to_categorical(y_train), to_categorical(y_val), to_categorical(y_test)

# Build a simple neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and log training and validation losses
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=32)

# Plot the training and validation loss graph
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Graph')
plt.legend()
plt.grid(True)
plt.savefig('loss_graph.png')  # Save the plot as an image
plt.close()


# Ensure model is trained before saving
chatbot.fit_model(epochs=10)
chatbot.save_model()

intents = discord.Intents.all()
client = discord.Client(intents=intents)

load_dotenv()
TOKEN = 'MTE5MDkyNTc4NTYyMDU1Nzg1NA.GBo35W.lAovi0N837EwoThXHWOYy7HXfGVeQx-Ctyw7Wk'


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.content.startswith('aibot'):
        response = chatbot.process_input(message.content[6:])

        await message.channel.send(response)


        # Plot the training loss graph


        # Send the training loss graph as an attachment
        # await message.channel.send(file=discord.File('loss_graph.png'))


client.run(TOKEN)


