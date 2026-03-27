import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import math

from helper import BOARD_SIZE, SHAPES, is_valid_move, rotate_shape, flip_shape

class AdvancedBlokusModel:
    def __init__(self, board_size=20, num_blocks=10, filters=128):
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.filters = filters
        self.policy_dim = 21 * 8 * (board_size * board_size) 
        self.model = self.build_unified_model()

    def build_unified_model(self):
        inputs = layers.Input(shape=(self.board_size, self.board_size, 6))

        x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        for _ in range(self.num_blocks):
            shortcut = x
            x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            
            g = layers.GlobalAveragePooling2D()(x)
            g = layers.Dense(self.filters, use_bias=False)(g)
            g = layers.Reshape((1, 1, self.filters))(g)
            x = layers.Add()([x, g]) 
            
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)

        p = layers.Conv2D(4, 1, padding='same', use_bias=False)(x)
        p = layers.BatchNormalization()(p)
        p = layers.Activation('relu')(p)
        p = layers.Flatten()(p)
        policy_out = layers.Dense(self.policy_dim, activation='softmax', name='policy')(p)

        v = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        v = layers.BatchNormalization()(v)
        v = layers.Activation('relu')(v)
        v = layers.Flatten()(v)
        v = layers.Dense(256, activation='relu')(v)
        value_out = layers.Dense(1, activation='tanh', name='value')(v)

        s = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        s = layers.BatchNormalization()(s)
        s = layers.Activation('relu')(s)
        s = layers.Flatten()(s)
        s = layers.Dense(256, activation='relu')(s)
        score_out = layers.Dense(1, name='score_lead')(s)  

        own = layers.Conv2D(5, 1, padding='same', name='ownership_conv')(x)
        ownership_out = layers.Reshape((self.board_size, self.board_size, 5))(own)
        ownership_out = layers.Softmax(axis=-1, name='ownership')(ownership_out)

        model = models.Model(inputs=inputs, outputs=[policy_out, value_out, score_out, ownership_out])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error',
                'score_lead': 'huber',  
                'ownership': 'sparse_categorical_crossentropy' 
            },
            loss_weights={
                'policy': 1.0,
                'value': 1.0,
                'score_lead': 0.05, 
                'ownership': 0.05   
            }
        )
        return model

class ExpertNode:
    def __init__(self, prior, parent=None, action_id=None):
        self.prior = prior
        self.parent = parent
        self.action_id = action_id 
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.score_sum = 0.0
        
        self.virtual_loss = 0.0 
        self.is_evaluating = False 

    def q_value(self): 
        if self.visit_count <= 0: return 0
        return (self.value_sum - self.virtual_loss) / self.visit_count
        
    def q_score(self): 
        return self.score_sum / self.visit_count if self.visit_count > 0 else 0

class ExpertBlokusBot:
    def __init__(self, name="ExpertZero", model=None, pipe=None, shared_data=None, is_training=False):
        self.name = name
        self.model = model
        self.pipe = pipe
        self.shared_data = shared_data
        self.is_training = is_training
        self.c_puct = 1.5
        self.score_utility_weight = 0.02 
        
        self.policy_dim = 21 * 8 * (BOARD_SIZE * BOARD_SIZE) 
        self.shape_keys = list(SHAPES.keys())
        self.shape_to_id = {name: idx for idx, name in enumerate(self.shape_keys)}

    def get_play(self, board, color, inventories, first_moves, pass_count):
        state_tensor = self._build_state_tensor(board, color, inventories, first_moves)
        legal_moves = self._get_legal_moves(board, color, inventories, first_moves)
        if not legal_moves: return None, []
        action, _ = self.get_action(state_tensor, legal_moves, self.is_training)
        return self._decode_action(action, legal_moves)

    def get_action(self, state_tensor, legal_moves, is_training=True, fast_playout=False):
        num_simulations = 20 if fast_playout else 200
        VIRTUAL_THREADS = 16 
        
        root = ExpertNode(1.0)
        
        if is_training:
            noise = np.random.dirichlet([0.03] * len(legal_moves))
            for action, n in zip(legal_moves, noise):
                prob = 0.75 * (1.0/len(legal_moves)) + 0.25 * n
                root.children[action] = ExpertNode(prior=prob, parent=root, action_id=action[0])
        else:
            for action in legal_moves:
                root.children[action] = ExpertNode(prior=1.0/len(legal_moves), parent=root, action_id=action[0])

        sims_completed = 0
        while sims_completed < num_simulations:
            paths_to_eval = []
            states_to_eval = []
            
            for _ in range(VIRTUAL_THREADS):
                if sims_completed >= num_simulations: break
                
                node = root
                path = [node]
                
                while len(node.children) > 0:
                    node.virtual_loss += 1.0 
                    node.visit_count += 1
                    
                    best_score = -float('inf')
                    best_child = None
                    
                    for action, child in node.children.items():
                        q = child.q_value() + (self.score_utility_weight * child.q_score())
                        
                        # Added math.sqrt safety
                        u = self.c_puct * child.prior * math.sqrt(max(0, node.visit_count)) / (1 + child.visit_count)
                        if q + u > best_score:
                            best_score, best_child = q + u, child
                    
                    node = best_child
                    path.append(node)
                    
                if node.is_evaluating:
                    # 🚀 CRITICAL BUG FIX: Roll back ONLY path[:-1] to prevent negative visits on leaves
                    for n in path[:-1]:
                        n.virtual_loss = max(0.0, n.virtual_loss - 1.0)
                        n.visit_count = max(0, n.visit_count - 1)
                    continue 
                    
                node.is_evaluating = True
                node.virtual_loss += 1.0
                node.visit_count += 1
                
                paths_to_eval.append(path)
                states_to_eval.append(state_tensor) 
                sims_completed += 1
                
            if not paths_to_eval:
                break # Failsafe to prevent zero-batch locks
                
            batch_size = len(states_to_eval)
            if self.shared_data:
                w_id, s_states, s_policies, s_values, s_scores = self.shared_data
                s_states[w_id, :batch_size] = states_to_eval
                self.pipe.send(batch_size) 
                self.pipe.recv()     
                policies = s_policies[w_id, :batch_size].copy()
                values = s_values[w_id, :batch_size].copy()
                score_leads = s_scores[w_id, :batch_size].copy()
            else:
                preds = self.model.predict(np.array(states_to_eval), verbose=0)
                policies, values, score_leads = preds[0], preds[1].flatten(), preds[2].flatten()
            
            for i, path in enumerate(paths_to_eval):
                leaf = path[-1]
                policy, value, score_lead = policies[i], float(values[i]), float(score_leads[i])
                
                leaf.is_evaluating = False
                
                for action in legal_moves:
                    action_id = action[0]
                    leaf.children[action] = ExpertNode(prior=policy[action_id], parent=leaf, action_id=action_id)
                
                for node in reversed(path):
                    # Added max(0) safety floors
                    node.virtual_loss = max(0.0, node.virtual_loss - 1.0) 
                    node.value_sum += value  
                    node.score_sum += score_lead
                    value, score_lead = -value, -score_lead 

        # Build policy outputs with positive safeguards
        visits = [max(0, root.children[a].visit_count) if a in root.children else 0 for a in legal_moves]
        if is_training:
            visits = np.array(visits) ** (1.0 / 1.0)
            sum_visits = np.sum(visits)
            if sum_visits > 0:
                pi_probs = visits / sum_visits
            else:
                pi_probs = np.ones(len(legal_moves)) / len(legal_moves)
            chosen_idx = np.random.choice(len(legal_moves), p=pi_probs)
        else:
            chosen_idx = np.argmax(visits)
            pi_probs = np.zeros(len(legal_moves))
            pi_probs[chosen_idx] = 1.0

        full_pi = np.zeros(self.policy_dim)
        for idx, action in enumerate(legal_moves): full_pi[action[0]] = pi_probs[idx]

        return legal_moves[chosen_idx], full_pi

    def _build_state_tensor(self, board, color, inventories, first_moves):
        tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)
        board_np = np.array(board)
        for p in range(1, 5): tensor[:, :, p-1] = (board_np == p).astype(np.float32)
        tensor[:, :, 4] = color / 4.0
        tensor[:, :, 5] = 1.0 if first_moves[color] else 0.0
        return tensor

    def _get_legal_moves(self, board, color, inventories, first_moves):
        legal_moves = []
        is_first = first_moves[color]
        my_shapes = inventories.get(color, inventories.get(str(color), []))
        for shape_name in my_shapes:
            shape_idx = self.shape_to_id[shape_name]
            base_shape = SHAPES[shape_name]
            transforms = []
            seen_hashes = set()
            orient_id = 0
            for flip in [False, True]:
                curr_shape = base_shape if not flip else flip_shape(base_shape)
                for rot in range(4):
                    if rot > 0: curr_shape = rotate_shape(curr_shape)
                    min_r = min(r for r, c in curr_shape)
                    min_c = min(c for r, c in curr_shape)
                    norm_shape = tuple(sorted((r - min_r, c - min_c) for r, c in curr_shape))
                    if norm_shape not in seen_hashes:
                        seen_hashes.add(norm_shape)
                        transforms.append((orient_id, norm_shape))
                    orient_id += 1
            for orient_idx, shape_coords in transforms:
                max_r, max_c = max(r for r, c in shape_coords), max(c for r, c in shape_coords)
                for r in range(BOARD_SIZE - max_r):
                    for c in range(BOARD_SIZE - max_c):
                        shifted_coords = [(r + sr, c + sc) for sr, sc in shape_coords]
                        if is_valid_move(board, color, shifted_coords, is_first):
                            action_id = (shape_idx * 8 * 400) + (orient_idx * 400) + (r * 20) + c
                            legal_moves.append((action_id, shape_name, orient_idx, r, c, tuple(shifted_coords)))
        return legal_moves

    def _decode_action(self, action, legal_moves):
        if not action: return None, []
        return action[1], action[5]