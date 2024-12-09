import re
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for rendering plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
import os
import time
import logging

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/player_projection', methods=['POST'])
def player_projection():
    try:
        logging.info("Request received for player projection.")

        # Parse request JSON
        data = request.json
        player_name = data.get('player_name')
        logging.debug(f"Received player name: {player_name}")

        if not player_name:
            logging.error("Player name is missing in the request.")
            return jsonify({'error': 'Player name is required'}), 400

        # Load and preprocess data
        data_file = 'weekly_player_data.csv'
        if not os.path.exists(data_file):
            logging.error(f"Data file not found: {data_file}")
            return jsonify({'error': 'Data file not found'}), 500

        weekly_data = pd.read_csv(data_file)
        logging.info("Data file loaded successfully.")

        try:
            weekly_data = pd.read_csv(data_file, na_values=['--', 'NA', 'null'])
            required_columns = ['player_name', 'position', 'season', 'week', 'fantasy_points_ppr']
            weekly_data.dropna(subset=required_columns, inplace=True)
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return jsonify({'error': 'Failed to load or preprocess data.'}), 500


        # Drop Week 18 data
        weekly_data = weekly_data[weekly_data['week'] != 18]
        logging.debug(f"Data after removing Week 18: {weekly_data.shape[0]} rows remaining.")

        # Filter player data
        player_data = weekly_data[weekly_data['player_name'] == player_name].sort_values(by=['season', 'week'])
        logging.debug(f"Filtered player data: {player_data.shape[0]} rows.")

        if player_data.empty:
            logging.warning(f"No data found for player: {player_name}")
            return jsonify({'error': f'No data found for player: {player_name}'}), 404

        # Determine position-specific features
        position = player_data['position'].iloc[0]
        logging.debug(f"Player position: {position}")

        if position == 'QB':
            features = ['passing_yards', 'pass_td', 'interception', 'rushing_yards', 'run_td', 'passer_rating']
        elif position == 'WR':
            features = ['receptions', 'receiving_yards', 'reception_td', 'rushing_yards', 'run_td']
        elif position == 'TE':
            features = ['receptions', 'receiving_yards', 'reception_td', 'rushing_yards', 'run_td']
        elif position == 'RB':
            features = ['receptions', 'receiving_yards', 'reception_td', 'rushing_yards', 'run_td']
        else:
            logging.error(f"Unsupported position: {position}")
            return jsonify({'error': f'Unsupported position: {position}'}), 400

        # Ensure the required features and target variable are present
        missing_features = [f for f in features if f not in player_data.columns]
        if 'fantasy_points_ppr' not in player_data.columns or missing_features:
            error_message = f"Missing required columns: {missing_features + ['fantasy_points_ppr'] if 'fantasy_points_ppr' not in player_data.columns else missing_features}"
            logging.error(error_message)
            return jsonify({'error': error_message}), 500

        # Compute weights based on recency
        player_data['game_weight'] = 1 / (player_data['season'] * 100 + player_data['week'])
        player_data['game_weight'] = player_data['game_weight'] / player_data['game_weight'].max()
        logging.debug("Game weights computed.")

        # Prepare sequences for training
        X_sequences, y_sequences = [], []

        for i in range(len(player_data) - 1):
            X_seq = player_data.iloc[i][features].values
            y_value = player_data.iloc[i + 1]['fantasy_points_ppr'] * player_data.iloc[i + 1]['game_weight']
            X_sequences.append(X_seq)
            y_sequences.append(y_value)

        logging.info("Training sequences prepared.")
        X_sequences = np.array(X_sequences).astype(np.float32)
        y_sequences = np.array(y_sequences).astype(np.float32)
        logging.debug(f"X_sequences shape: {X_sequences.shape}, y_sequences shape: {y_sequences.shape}")

        # Split data into training and testing
        train_size = int(0.8 * len(X_sequences))
        X_train, X_test = X_sequences[:train_size], X_sequences[train_size:]
        y_train, y_test = y_sequences[:train_size], y_sequences[train_size:]
        logging.info(f"Training size: {len(X_train)}, Test size: {len(X_test)}")

        # Normalize the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        logging.debug("Data normalized.")

        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Define and compile the LSTM model
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(100, activation='tanh', return_sequences=True),
            Dropout(0.3),
            LSTM(50, activation='tanh', return_sequences=True),
            Dropout(0.3),
            LSTM(25, activation='tanh'),
            Dropout(0.3),
            Dense(1)
        ])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.compile(optimizer=Adam(learning_rate=0.002), loss='mean_squared_error', metrics=[MeanSquaredError(name='mse'), MeanAbsoluteError(name='mae')])
        logging.info("Model defined and compiled.")

        # Train the model
        history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
        logging.info("Model training completed.")

        # Generate graphs and save them
        timestamp = int(time.time())
        output_path_mse = f'static/mse_graph_{timestamp}.png'
        plt.plot(history.history['mse'], label='Training MSE')
        plt.plot(history.history['val_mse'], label='Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig(output_path_mse)
        plt.close()

        output_path_mae = f'static/mae_graph_{timestamp}.png'
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.savefig(output_path_mae)
        plt.close()

        # Make a prediction for the next game
        latest_sequence = X_sequences[-1].reshape(1, 1, -1)
        latest_sequence = scaler.transform(latest_sequence.reshape(-1, latest_sequence.shape[-1])).reshape(latest_sequence.shape)
        projection = model.predict(latest_sequence)[0][0]
        projection = float(projection)

        #Get ESPN Prediction

        # Check if ESPN data exists
        espn_file = 'FantasyFootballWeekly.xlsx'
        if not os.path.exists(espn_file):
            logging.error(f"ESPN data file not found: {espn_file}")
            return jsonify({'error': 'ESPN data file not found'}), 500

        # Load ESPN data
        espn_data = pd.read_excel(espn_file, sheet_name=None)

        # Get the current season and week
        current_season = player_data['season'].iloc[-1]
        current_week = player_data['week'].iloc[-1]

        # Filter ESPN data for the player and current week    
        espn_projection = None
        if current_season == 2023:
            for sheet_name, sheet_data in espn_data.items():
                match = re.search(r'Week(\d+)', sheet_name)
                if match and int(match.group(1)) == current_week:
                    filtered_sheet = sheet_data[sheet_data['PLAYER NAME'] == player_name]
                    if not filtered_sheet.empty:
                        try:
                            espn_proj_value = filtered_sheet['PROJ'].values[0]
                            espn_projection = float(espn_proj_value) if pd.notna(espn_proj_value) and re.match(r'^-?\d+(\.\d+)?$', str(espn_proj_value)) else None
                            if espn_projection is not None:
                                break
                        except (ValueError, TypeError):
                            logging.warning(f"Invalid projection value: {filtered_sheet['PROJ'].values[0]}")
                            espn_projection = None

        if espn_projection is None:
            logging.warning(f"ESPN projection not found for player: {player_name}, season: {current_season}, week: {current_week + 1}")
            espn_projection_message = "ESPN projection not available."
        else:
            logging.info(f"ESPN Projection for {player_name}: {espn_projection:.2f} fantasy points")
            espn_projection_message = f"ESPN Projection for {player_name}: {espn_projection:.2f} fantasy points"

        logging.info(f"Projection for {player_name}: {projection:.2f} fantasy points.")
        return jsonify({
            'message': f'Projection for {player_name}: {projection:.2f} fantasy points for Week {current_week + 1} in {current_season}. \n {espn_projection_message}',
            'projection': projection,
            'espn_projection': espn_projection,
            'mse_graph_url': f'/{output_path_mse}',
            'mae_graph_url': f'/{output_path_mae}'
        })

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
