import chess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import json
import dotenv
import os

dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def print_board(board):
    print(str(board))

def rank_moves(board, previous_state_embeddings, previous_move_embeddings, model):
    moves = list(board.legal_moves)
    move_scores = []
    for move in moves:
        board.push(move)
        state_string = str(board)
        move_string = move.uci()
        state_embedding = model.encode([state_string])[0]
        move_embedding = model.encode([move_string])[0]
        state_similarities = cosine_similarity([state_embedding], previous_state_embeddings)[0]
        move_similarities = cosine_similarity([move_embedding], previous_move_embeddings)[0]
        average_state_similarity = np.mean(state_similarities)
        average_move_similarity = np.mean(move_similarities)
        instruction = f'Here is the current board:\n\n{state_string}\n\nThe average cosine similarity of the current state to the previous states is {average_state_similarity} and the average cosine similarity of the current move to the previous moves is {average_move_similarity}.\n\nPlease provide a score for this position. ...'
        score = get_openai_score(instruction)
        move_scores.append((move, score))
        board.pop()
    return move_scores

def get_openai_score(instruction):
    chess_rules = '\n1. The game is played on an 8x8 grid, with alternating white and black squares. \
        \n2. Each player starts with 16 pieces: one king, one queen, two rooks, two knights, two bishops, and eight pawns. \
        \n3. The goal of the game is to checkmate the opponent\'s king. This means the opponent\'s king is in a position to be captured ("in check") and there is no way to move the king out of capture ("checkmate"). \
        \n4. The game can also end by resignation. If a player decides they cannot win, they can choose to resign, ending the game immediately. \
        \n5. The game is drawn if neither player can checkmate the other\'s king. This can occur under several conditions, including insufficient material to checkmate, stalemate, or threefold repetition of a position.'
    prompt = chess_rules + "\n\n" + instruction
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=100)
    return response.choices[0].text.strip()

def get_best_move(move_scores):
    move_scores.sort(key=lambda x: x[1], reverse=True)
    best_move = move_scores[0][0]
    return best_move.uci()

def get_openai_move(board, previous_state_embeddings, previous_move_embeddings, model):
    move_scores = rank_moves(board, previous_state_embeddings, previous_move_embeddings, model)
    best_move = get_best_move(move_scores)
    return best_move

def get_user_move(board):
    while True:
        move = input("Enter your move: ")
        if move.lower() == 'resign':
            print("You've resigned. Game over.")
            return None
        try:
            move_obj = chess.Move.from_uci(move)
            if move_obj in board.legal_moves:
                return move
            else:
                print("Illegal move. Try again.")
        except:
            print("Invalid move. Try again.")

def load_game_data():
    try:
        with open('game_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "responses": [],
            "state_embeddings": [],
            "move_embeddings": []
        }

def play_game():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    board = chess.Board()
    previous_state_embeddings = []
    previous_move_embeddings = []
    while True:
        print_board(board)
        if board.is_checkmate():
            print("Checkmate!")
            break
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            print("It's a draw!")
            break
        elif board.turn:
            move = get_user_move(board)
            if move is None:  # user resigns
                break
        else:
            move = get_openai_move(board, previous_state_embeddings, previous_move_embeddings, model)
        board.push_uci(move)
        state_string = str(board)
        move_string = move
        state_embedding = model.encode([state_string])[0]
        move_embedding = model.encode([move_string])[0]
        game_data["state_embeddings"].append(state_embedding.tolist())
        game_data["move_embeddings"].append(move_embedding.tolist())
        with open('game_data.json', 'w') as f:
            json.dump(game_data, f)
        previous_state_embeddings.append(state_embedding)
        previous_move_embeddings.append(move_embedding)

if __name__ == "__main__":

    game_data = load_game_data()  # Load existing data from "game_data.json" or initialize empty data
    play_game()
