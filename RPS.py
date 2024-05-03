# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
'''
def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    print(prev_play)
    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import math

# Define a function to preprocess the data
def preprocess_data(data):
    # Convert moves to numerical labels
    label_mapping = {"R": 0, "P": 1, "S": 2}
    return [label_mapping[move] for move in data]

# Train an RNN model on the fly
def train_model_onthefly(historical_moves, next_moves, model, seq_length):
    if len(historical_moves) < seq_length + 1:
        # Not enough data to train the model
        return model
    
    X_train = np.array([preprocess_data(historical_moves[-seq_length:])])
    y_train = np.array([preprocess_data(next_moves[-1])])
    
    model.train_on_batch(X_train, y_train)
    return model

# Predict the next move using the trained model
def player(opponent_last_move, opponent_last_moves=[]):
    if opponent_last_move=='':
        opponent_last_move="R"
    opponent_last_moves.append(opponent_last_move)
    if len(opponent_last_moves) < (seq_length+1):
        # Not enough historical moves, choose randomly
        return np.random.choice(["R", "P", "S"])
    X_train = np.array([preprocess_data(opponent_last_moves[-(seq_length+1):-1])])
    y_trai = np.array(preprocess_data([opponent_last_move]))
    X_pred = np.array([preprocess_data(opponent_last_moves[-seq_length:])])
    model.fit(X_train,y_trai,epochs=5,batch_size=1,verbose=0)
    next_move_idx = model.predict(X_pred,verbose=0)[0]
    next_move_mapping = {0: "R", 1: "P", 2: "S"}
    next_move = next_move_mapping[np.argmax(next_move_idx)]
    return next_move

# Example usage:
# Initialize the model
seq_length = 3  # Sequence length
model:Sequential = Sequential([
    Embedding(input_dim=3, output_dim=8, input_length=seq_length),
    LSTM(64),
    Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
# Play the game
historical_moves = []
next_moves = []
for _ in range(10):  # Play 10 rounds
    # Opponent's move (random for this example)
    opponent_last_move = np.random.choice(["R", "P", "S"])
    
    # Player's move
    next_move = player(opponent_last_moves=historical_moves, model=model, seq_length=seq_length)
    
    # Update historical moves and next moves
    historical_moves.append(opponent_last_move)
    next_moves.append(next_move)
    
    # Train the model on the fly
    model = train_model_onthefly(historical_moves, next_moves, model, seq_length)
    
    print("Opponent's move:", opponent_last_move)
    print("Player's move:", next_move)
'''
