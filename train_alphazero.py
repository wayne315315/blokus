import numpy as np
import tensorflow as tf
from tf_alphazero_bot import AdvancedBlokusModel, ExpertBlokusBot
from helper import BOARD_SIZE

def generate_expert_game(bot):
    states, policies, players = [], [], []
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    
    # Initialize game variables
    inventories = {i: list(bot.SHAPES.keys()) for i in range(1, 5)} if hasattr(bot, 'SHAPES') else {}
    first_moves = {1: True, 2: True, 3: True, 4: True}
    current_player = 1
    pass_count = 0
    
    while pass_count < 4:
        state_tensor = bot._build_state_tensor(board, current_player, inventories, first_moves)
        legal_moves = bot._get_legal_moves(board, current_player, inventories, first_moves)
        
        if not legal_moves:
            pass_count += 1
            current_player = (current_player % 4) + 1
            continue
            
        pass_count = 0
        
        # Playout Cap Randomization: 80% Fast (50), 20% Deep (400)
        is_fast_playout = np.random.rand() < 0.80 
        action, pi = bot.get_action(state_tensor, legal_moves, is_training=True, fast_playout=is_fast_playout)
        
        # ONLY record high-quality Deep searches for training!
        if not is_fast_playout:
            states.append(state_tensor)
            # Map pi back to full action space shape (e.g. 1600 dims)
            full_pi = np.zeros(BOARD_SIZE * BOARD_SIZE * 4) 
            # full_pi[mapped_indices] = pi
            policies.append(full_pi) 
            players.append(current_player)
            
        # Apply Move
        shape_name, coords = bot._decode_action(action, legal_moves)
        for r, c in coords:
            board[r][c] = current_player
        if shape_name in inventories[current_player]: inventories[current_player].remove(shape_name)
        first_moves[current_player] = False
        current_player = (current_player % 4) + 1
        
    # --- END OF GAME: CALCULATE AUXILIARY TARGETS ---
    def get_score(pid): return sum(int(s.split('_')[0]) for s in inventories.get(pid, []))

    team1_score = get_score(1) + get_score(3)
    team2_score = get_score(2) + get_score(4)
    
    # Ownership Matrix (Teaches spatial territory map)
    # 0 = Empty, 1-4 = Player ID
    ownership_matrix = np.copy(board) 
    
    val_targets, score_targets, own_targets = [], [], []
    for p in players:
        if p in [1, 3]:
            val_targets.append(1 if team1_score < team2_score else -1)
            score_targets.append(team2_score - team1_score) # Point margin
        else:
            val_targets.append(1 if team2_score < team1_score else -1)
            score_targets.append(team1_score - team2_score)
        own_targets.append(ownership_matrix)

    return states, policies, val_targets, score_targets, own_targets

def run_training_pipeline():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    adv_model = AdvancedBlokusModel()
    bot = ExpertBlokusBot(model=adv_model.model)
    
    for iteration in range(50):
        print(f"Iteration {iteration}...")
        S, P, V, SC, O = [], [], [], [], []
        
        # Parallel generation would be used here normally
        for _ in range(10): 
            s, p, v, sc, o = generate_expert_game(bot)
            S.extend(s); P.extend(p); V.extend(v); SC.extend(sc); O.extend(o)
            
        adv_model.model.fit(
            np.array(S), 
            {'policy': np.array(P), 'value': np.array(V), 'score_lead': np.array(SC), 'ownership': np.array(O)},
            batch_size=256,
            epochs=2
        )
        adv_model.model.save(f"blokus_expert_v{iteration}.keras")

if __name__ == "__main__":
    run_training_pipeline()
