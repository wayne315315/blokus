import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision

from helper import BOARD_SIZE, SHAPES, is_valid_move, rotate_shape, flip_shape

mixed_precision.set_global_policy('mixed_float16')

# ==============================================================================
# SPATIAL 2D RESNET ARCHITECTURE (AlphaZero Style)
# ==============================================================================
def conv2d_residual_block(x, filters, kernel_size=3):
    shortcut = x
    
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def create_spatial_resnet(input_shape=(20, 20, 6), num_blocks=5, filters=128, is_policy=False):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    for _ in range(num_blocks):
        x = conv2d_residual_block(x, filters)
        
    if is_policy:
        x = layers.Conv2D(2, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='linear', dtype='float32')(x)
    else:
        x = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1, activation='tanh', dtype='float32')(x)

    return models.Model(inputs=inputs, outputs=x)

def create_value_network():
    return create_spatial_resnet(is_policy=False)

def create_policy_network():
    return create_spatial_resnet(is_policy=True)

# ==============================================================================
# MONTE CARLO TREE SEARCH NODE
# ==============================================================================
class MCTSNode:
    def __init__(self, action=None, prior=0.0, team=0):
        self.action = action
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = []
        self.is_expanded = False
        self.team = team # 0 for P1(Blue/Red), 1 for P2(Yellow/Green)

# ==============================================================================
# ALPHAZERO BLOKUS BOT (MCTS ENABLED)
# ==============================================================================
class BlokusAlphaZeroBot:
    def __init__(self, name="AZ Bot", val_model_path="tf_value_model.keras", 
                 policy_model_path="tf_policy_model.keras", is_training=True, exploration_rate=0.15, pipe=None):
        self.name = name
        self.is_training = is_training
        self.exploration_rate = exploration_rate if is_training else 0.0
        self.pipe = pipe  
        
        self.episode_memory = [] 
        self.policy_memory = []  
        
        if self.pipe is None:
            try:
                self.val_net = tf.keras.models.load_model(val_model_path, compile=False)
                self.policy_net = tf.keras.models.load_model(policy_model_path, compile=False)
            except:
                self.val_net = create_value_network()
                self.policy_net = create_policy_network()

    def clear_memory(self):
        self.episode_memory = []
        self.policy_memory = []

    def _get_legal_actions(self, board, color_id, available_shapes, is_first_move):
        legal_actions = []
        for shape_name in available_shapes:
            base_coords = SHAPES[shape_name]
            current_coords = base_coords
            for flip_state in range(2):
                if flip_state == 1: current_coords = flip_shape(base_coords)
                for rot_state in range(4):
                    current_coords = rotate_shape(current_coords)
                    for r in range(BOARD_SIZE):
                        for c in range(BOARD_SIZE):
                            shifted_coords = [(r + dr, c + dc) for dr, dc in current_coords]
                            if is_valid_move(board, color_id, shifted_coords, is_first_move):
                                legal_actions.append((shape_name, shifted_coords))
        return legal_actions

    def _encode_base_board(self, board, color_id):
        seq = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = board[r][c]
                if cell > 0: seq[r, c, cell - 1] = 1.0
        seq[:, :, 5] = color_id / 4.0 
        return seq

    def _simulate_step(self, board, inventories, first_moves, color_id, pass_count, action):
        n_board = [row[:] for row in board]
        n_inv = {k: v[:] for k, v in inventories.items()}
        n_fm = first_moves.copy()
        
        if action[0] is not None:
            shape_name, coords = action
            for r, c in coords:
                n_board[r][c] = color_id
            k = str(color_id) if str(color_id) in n_inv else color_id
            if shape_name in n_inv[k]:
                n_inv[k].remove(shape_name)
            n_fm[color_id] = False
            n_pass = 0
        else:
            n_pass = pass_count + 1
            
        n_color = (color_id % 4) + 1
        return n_board, n_inv, n_fm, n_color, n_pass

    def get_play(self, board, color_id, inventories, first_moves, pass_count=0):
        # Gracefully handle string vs int JSON keys
        inv_key = str(color_id) if str(color_id) in inventories else color_id
        available_shapes = inventories[inv_key]
        is_first = first_moves[color_id]
        
        legal_actions = self._get_legal_actions(board, color_id, available_shapes, is_first)
        
        if not legal_actions: return None, None
        if len(legal_actions) == 1: return legal_actions[0]

        # --- MCTS INITIALIZATION ---
        root_team = 0 if color_id in [1, 3] else 1
        root = MCTSNode(team=root_team)
        
        base_board = self._encode_base_board(board, color_id)
        batch_inputs = np.empty((len(legal_actions), 20, 20, 6), dtype=np.float32)
        for i, action in enumerate(legal_actions):
            np.copyto(batch_inputs[i], base_board)
            for r, c in action[1]:
                batch_inputs[i, r, c, 4] = 1.0
        
        if self.is_training and self.pipe:
            self.pipe.send((False, batch_inputs.astype(np.float16)))
            values = self.pipe.recv().flatten().tolist()
            self.pipe.send((True, batch_inputs.astype(np.float16)))
            policy_logits = self.pipe.recv().flatten()
        else:
            values = self.val_net(batch_inputs, training=False).numpy().flatten().tolist()
            policy_logits = self.policy_net(batch_inputs, training=False).numpy().flatten()
            
        exp_logits = np.exp(policy_logits - np.max(policy_logits))
        probs = (exp_logits / np.sum(exp_logits)).tolist()
        
        if self.exploration_rate > 0:
            explore_prob = self.exploration_rate / len(legal_actions)
            probs = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probs]
            
        for i, action in enumerate(legal_actions):
            child = MCTSNode(action=action, prior=probs[i], team=root_team)
            child.value_sum = values[i]
            child.visits = 1
            root.children.append(child)
            
        root.is_expanded = True
        root.visits = len(legal_actions)

        # --- MCTS SIMULATION LOOP ---
        num_sims = 15 if self.is_training else 50
        for _ in range(num_sims):
            node = root
            c_board = [row[:] for row in board]
            c_inv = {k: v[:] for k,v in inventories.items()}
            c_fm = first_moves.copy()
            c_color = color_id
            c_pass = pass_count
            
            search_path = [node]
            
            # 1. Selection (PUCT)
            while node.is_expanded and node.children:
                best_score = -float('inf')
                best_child = None
                for child in node.children:
                    q = child.value_sum / child.visits if child.visits > 0 else 0.0
                    u = 1.5 * child.prior * np.sqrt(node.visits) / (1 + child.visits)
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_child = child
                node = best_child
                search_path.append(node)
                
                if node.action[0] is not None:
                    c_board, c_inv, c_fm, c_color, c_pass = self._simulate_step(c_board, c_inv, c_fm, c_color, c_pass, node.action)
                else:
                    c_pass += 1
                    c_color = (c_color % 4) + 1
                    
            # 2. Expansion & Evaluation
            if c_pass < 4 and not node.is_expanded:
                c_inv_key = str(c_color) if str(c_color) in c_inv else c_color
                l_actions = self._get_legal_actions(c_board, c_color, c_inv[c_inv_key], c_fm[c_color])
                if not l_actions: l_actions = [(None, None)]
                    
                b_board = self._encode_base_board(c_board, c_color)
                b_inputs = np.empty((len(l_actions), 20, 20, 6), dtype=np.float32)
                for i, a in enumerate(l_actions):
                    np.copyto(b_inputs[i], b_board)
                    if a[0] is not None:
                        for r, c in a[1]: b_inputs[i, r, c, 4] = 1.0
                        
                if self.is_training and self.pipe:
                    self.pipe.send((False, b_inputs.astype(np.float16)))
                    n_vals = self.pipe.recv().flatten().tolist()
                    self.pipe.send((True, b_inputs.astype(np.float16)))
                    n_pols = self.pipe.recv().flatten()
                else:
                    n_vals = self.val_net(b_inputs, training=False).numpy().flatten().tolist()
                    n_pols = self.policy_net(b_inputs, training=False).numpy().flatten()
                    
                e_pols = np.exp(n_pols - np.max(n_pols))
                n_probs = (e_pols / np.sum(e_pols)).tolist()
                
                node_team = 0 if c_color in [1, 3] else 1
                for i, a in enumerate(l_actions):
                    child = MCTSNode(action=a, prior=n_probs[i], team=node_team)
                    child.value_sum = n_vals[i]
                    child.visits = 1
                    node.children.append(child)
                    
                node.is_expanded = True
                node.visits = len(l_actions)
                
                leaf_value = np.max(n_vals) 
                
                # 3. Backpropagate
                for p in reversed(search_path[:-1]):
                    p.visits += 1
                    if p.team == node_team: p.value_sum += leaf_value
                    else: p.value_sum -= leaf_value
                        
        # --- ACTION SELECTION ---
        if self.is_training:
            # Temperature = 1 (Proportional to MCTS visits)
            visits = [child.visits for child in root.children]
            sum_v = sum(visits)
            probs = [v / sum_v for v in visits]
            chosen_child = random.choices(root.children, weights=probs, k=1)[0]
        else:
            # Temperature = 0 (Argmax)
            chosen_child = max(root.children, key=lambda c: c.visits)
            
        chosen_index = root.children.index(chosen_child)
        mcts_probs = [child.visits / root.visits for child in root.children]
        
        if self.is_training:
            self.episode_memory.append({
                'inputs': batch_inputs, 
                'chosen_index': chosen_index, 
                'value_pred': chosen_child.value_sum / max(1, chosen_child.visits)
            })
            # Policy memory now tracks the improved MCTS visit counts, not the raw policy logic
            for i in range(len(legal_actions)):
                self.policy_memory.append((batch_inputs[i], mcts_probs[i]))
                
        return chosen_child.action
