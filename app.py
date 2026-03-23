from flask import Flask, render_template, jsonify, request
from helper import BOARD_SIZE, SHAPES, is_valid_move
from player import BotPlayer

app = Flask(__name__)

game_instance = {}
bot_instance = BotPlayer()

def init_game():
    global game_instance
    game_instance = {
        'board': [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)],
        'inventories': {
            "1": list(SHAPES.keys()),
            "2": list(SHAPES.keys()),
            "3": list(SHAPES.keys()),
            "4": list(SHAPES.keys())
        },
        'current_color': 1,
        'first_moves': {1: True, 2: True, 3: True, 4: True},
        'pass_count': 0, 
        'message': "Game Start! You play Blue (1) and Red (3). Blue goes first in Top-Left.",
        'is_game_over': False,
        'final_scores': None
    }

def calculate_scores():
    # In Blokus, you get -1 point for each unit square you failed to place.
    # We can extract the size of the shape from the first character of its name (e.g., "5_W5" -> 5)
    def get_color_score(color_id):
        inventory = game_instance['inventories'][str(color_id)]
        return sum(-int(shape.split('_')[0]) for shape in inventory)

    scores = {
        1: get_color_score(1),
        2: get_color_score(2),
        3: get_color_score(3),
        4: get_color_score(4)
    }
    
    # Player 1 is Blue(1) + Red(3). Bot is Yellow(2) + Green(4).
    p1_total = scores[1] + scores[3]
    bot_total = scores[2] + scores[4]
    
    game_instance['final_scores'] = {
        'p1': p1_total,
        'bot': bot_total,
        'details': scores
    }
    
    if p1_total > bot_total:
        game_instance['message'] = f"GAME OVER! You Win! (You: {p1_total} | Bot: {bot_total})"
    elif bot_total > p1_total:
        game_instance['message'] = f"GAME OVER! Bot Wins! (You: {p1_total} | Bot: {bot_total})"
    else:
        game_instance['message'] = f"GAME OVER! It's a Tie! ({p1_total} to {bot_total})"

def next_turn():
    game_instance['current_color'] = (game_instance['current_color'] % 4) + 1
    
    # If all 4 colors pass consecutively, the game ends
    if game_instance['pass_count'] >= 4:
        game_instance['is_game_over'] = True
        calculate_scores()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state', methods=['GET'])
def get_state():
    if not game_instance:
        init_game()
    return jsonify(game_instance)

@app.route('/api/play', methods=['POST'])
def play_shape():
    if game_instance['is_game_over']:
        return jsonify({'error': 'Game is over.'}), 400

    color_id = game_instance['current_color']
    
    if color_id not in [1, 3]:
        return jsonify({'error': 'It is the Bot\'s turn.'}), 400

    data = request.json
    shape_name = data.get('shape_name')
    coords = [tuple(c) for c in data.get('coords', [])]
    
    if is_valid_move(game_instance['board'], color_id, coords, game_instance['first_moves'][color_id]):
        for r, c in coords:
            game_instance['board'][r][c] = color_id
            
        game_instance['inventories'][str(color_id)].remove(shape_name)
        game_instance['first_moves'][color_id] = False
        game_instance['pass_count'] = 0 # Reset pass count on a valid move
        
        color_name = "Blue" if color_id == 1 else "Red"
        game_instance['message'] = f"You played a shape as {color_name}."
        
        next_turn()
        return get_state()
        
    return jsonify({'error': 'Invalid Move! You must touch EXACTLY a corner (if first move) or ONLY diagonal corners of your own color (if later).'}), 400

@app.route('/api/pass', methods=['POST'])
def pass_turn():
    if game_instance['is_game_over']: return get_state()
    color_id = game_instance['current_color']
    if color_id not in [1, 3]: return get_state()
    
    game_instance['pass_count'] += 1
    game_instance['message'] = f"{'Blue' if color_id == 1 else 'Red'} Passed."
    next_turn()
    return get_state()

@app.route('/api/bot_turn', methods=['POST'])
def bot_turn():
    if game_instance['is_game_over']: return get_state()
    
    color_id = game_instance['current_color']
    if color_id not in [2, 4]: return get_state()
        
    available_shapes = game_instance['inventories'][str(color_id)]
    is_first = game_instance['first_moves'][color_id]
    
    shape_name, coords = bot_instance.get_play(game_instance['board'], color_id, available_shapes, is_first)
    
    color_name = "Yellow" if color_id == 2 else "Green"

    if not shape_name:
        game_instance['pass_count'] += 1
        game_instance['message'] = f"Bot passed turn for {color_name}."
    else:
        for r, c in coords:
            game_instance['board'][r][c] = color_id
            
        game_instance['inventories'][str(color_id)].remove(shape_name)
        game_instance['first_moves'][color_id] = False
        game_instance['pass_count'] = 0
        game_instance['message'] = f"Bot played a shape as {color_name}."

    next_turn()
    return get_state()

@app.route('/api/reset', methods=['POST'])
def reset():
    init_game()
    return get_state()

if __name__ == '__main__':
    app.run(debug=True)