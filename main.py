import logging
import numpy as np
import pandas as pd
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import nest_asyncio
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Apply nest_asyncio to allow running the async event loop in environments that already have one
nest_asyncio.apply()

# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variables
user_data = {}  # To store each user's sequence history
authorized_users = {7284648472}  # Add user IDs of people allowed to use the bot
admin_user_id = 7284648472  # Replace with your Telegram ID for admin management

# Get Telegram bot token from environment variables
telegram_token = os.getenv("7638208028:AAHchjtDECdiyEaJNCyVvvoxsyvzaDr4Vek")

# Helper Functions for ARIMA, Random Forest, and LSTM

# ARIMA Model Prediction
def arima_prediction(sequence):
    try:
        model = ARIMA(sequence, order=(5, 1, 0))  # ARIMA parameters (p, d, q)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast[0]
    except Exception as e:
        return None

# Random Forest Prediction
def rf_prediction(sequence, rf_model):
    # Features: previous numbers
    features = np.array([[sequence[-1]]])  # last number in sequence
    prediction = rf_model.predict(features)
    return "Big" if prediction[0] > 5 else "Small"

# LSTM Model Prediction
def lstm_prediction(sequence, lstm_model):
    # Reshape sequence for LSTM input
    sequence = np.array(sequence).reshape((len(sequence), 1, 1))  # Reshape for LSTM input
    prediction = lstm_model.predict(sequence[-1:])
    return "Big" if prediction[0] > 5 else "Small"

# Train Random Forest Model
def train_rf_model(sequence):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # Features: previous numbers
    features = np.array([[sequence[i]] for i in range(len(sequence) - 1)])
    labels = np.array(sequence[1:])
    rf_model.fit(features, labels)
    return rf_model

# Train LSTM Model
def train_lstm_model(sequence):
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=False, input_shape=(1, 1)))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    sequence = np.array(sequence).reshape((len(sequence), 1, 1))  # Reshape for LSTM input
    lstm_model.fit(sequence[:-1], sequence[1:], epochs=10, batch_size=1, verbose=0)
    return lstm_model

# Helper Functions for User Data Management
def classify_number(number):
    return "Small (0-4)" if number <= 4 else "Big (5-9)"

def generate_final_prediction(predictions, user_id):
    """Generate final prediction using the majority vote from ARIMA, Random Forest, and LSTM."""
    # Track last 10 results (W/L)
    win_loss_sequence = ''.join(user_data[user_id]['win_loss_sequence'][-10:])
    final_prediction = {"Big": 0, "Small": 0}

    for prediction in predictions:
        final_prediction[prediction] += 1

    prediction_result = "Big" if final_prediction["Big"] > final_prediction["Small"] else "Small"
    
    # Bias final prediction based on W/L sequence (e.g., if the user has more Wins in the last 10 results, bias towards Big)
    if user_data[user_id]['win_loss_sequence'][-1] == 'W':  # If the last result was a win, increase Big bias
        if prediction_result == "Big":
            final_prediction["Big"] += 1
        else:
            final_prediction["Small"] += 1
    
    return f"Final Prediction: {prediction_result}\nLast 10 Results: {win_loss_sequence}"

# Track Win/Loss Sequence
def track_win_loss(user_id, result):
    if user_id not in user_data:
        user_data[user_id] = {'sequence': [], 'results': [], 'win_loss_sequence': []}
    user_data[user_id]['results'].append(result)
    user_data[user_id]['win_loss_sequence'].append(result)

# User Management Command Handlers
async def add_user(update: Update, context):
    user_id = update.message.from_user.id
    if user_id != admin_user_id:
        await update.message.reply_text("You are not authorized to manage users.")
        return
    try:
        new_user_id = int(context.args[0])
        if new_user_id in authorized_users:
            await update.message.reply_text("This user is already authorized.")
        else:
            authorized_users.add(new_user_id)
            await update.message.reply_text(f"User {new_user_id} has been added to the authorized users list.")
    except (IndexError, ValueError):
        await update.message.reply_text("Please provide a valid user ID. Example: /add_user 123456789")

async def remove_user(update: Update, context):
    user_id = update.message.from_user.id
    if user_id != admin_user_id:
        await update.message.reply_text("You are not authorized to manage users.")
        return
    try:
        remove_user_id = int(context.args[0])
        if remove_user_id in authorized_users:
            authorized_users.remove(remove_user_id)
            await update.message.reply_text(f"User {remove_user_id} has been removed from the authorized users list.")
        else:
            await update.message.reply_text("This user is not in the authorized users list.")
    except (IndexError, ValueError):
        await update.message.reply_text("Please provide a valid user ID. Example: /remove_user 123456789")

async def user_list(update: Update, context):
    user_id = update.message.from_user.id
    if user_id != admin_user_id:
        await update.message.reply_text("You are not authorized to manage users.")
        return
    if authorized_users:
        user_list_message = "Authorized Users:\n" + "\n".join(map(str, authorized_users))
    else:
        user_list_message = "No users are currently authorized."
    await update.message.reply_text(user_list_message)

# Bot Commands
async def start(update: Update, context):
    user_id = update.message.from_user.id
    if user_id not in authorized_users:
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    user_data[user_id] = {'sequence': [], 'results': [], 'win_loss_sequence': []}
    await update.message.reply_text("Hello! I am your prediction bot.\nSend a sequence for analysis.")

async def handle_message(update: Update, context):
    user_id = update.message.from_user.id
    if user_id not in authorized_users:
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    text = update.message.text.strip()
    if text.isdigit():
        sequence = list(map(int, text))
        win_loss_history = user_data[user_id]['win_loss_sequence']  # Use win/loss sequence as feature

        # Train models
        rf_model = train_rf_model(sequence)
        lstm_model = train_lstm_model(sequence)

        # Get predictions from ARIMA, Random Forest, and LSTM models
        arima_pred = arima_prediction(sequence)
        rf_pred = rf_prediction(sequence, rf_model)
        lstm_pred = lstm_prediction(sequence, lstm_model)

        # Handle case where ARIMA prediction is None
        arima_result = "Small" if arima_pred is None else "Big" if arima_pred > 5 else "Small"

        predictions = [rf_pred, lstm_pred, arima_result]
        final_prediction = generate_final_prediction(predictions, user_id)

        # Display predictions
        await update.message.reply_text(f"ARIMA Prediction: {arima_pred if arima_pred is not None else 'None'}")
        await update.message.reply_text(f"Random Forest Prediction: {rf_pred}")
        await update.message.reply_text(f"LSTM Prediction: {lstm_pred}")
        await update.message.reply_text(final_prediction)

        await update.message.reply_text("Reply 'W' for Win or 'L' for Loss to update your sequence.")
    elif text.upper() in ['W', 'L']:
        result = 'Win' if text.upper() == 'W' else 'Loss'
        track_win_loss(user_id, result)

        # Final prediction based on win/loss
        final_prediction = generate_final_prediction([rf_prediction(user_data[user_id]['sequence'], rf_model)], user_id)
        await update.message.reply_text(final_prediction)
        await update.message.reply_text("Send the next sequence for analysis.")

# Start the bot
if __name__ == '__main__':
    application = Application.builder().token(telegram_token).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('add_user', add_user))
    application.add_handler(CommandHandler('remove_user', remove_user))
    application.add_handler(CommandHandler('user_list', user_list))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()
