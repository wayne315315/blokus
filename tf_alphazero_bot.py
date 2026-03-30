import numpy as np
import math

from helper import BOARD_SIZE, SHAPES, is_valid_move, rotate_shape, flip_shape

class AdvancedBlokusModel:
    def __init__(self, board_size=20, num_blocks=10, filters=128):
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.filters = filters
        self.model = self.build_unified_model()

    def build_unified_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models

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

        # 🛑 ARCHITECTURE REWRITE: The 67,200 Policy head is completely deleted! 
        # The network now only acts as a Q-Value evaluator for After-States.
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

        model = models.Model(inputs=inputs, outputs=[value_out, score_out])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'value': 'mean_squared_error', 'score_lead': 'huber'},
            loss_weights={'value': 1.0, 'score_lead': 0.05}
        )
        return model

class ExpertNode:
    def __init__(self, board, current_color, inventories, first_moves, parent=None, action_id=None, prior=0.0, q_init=0.0):
        self.board = board
        self.current_color = current_color
        self.inventories = inventories
        self.first_moves = first_moves
        
        self.parent = parent
        self.action_id = action_id 
        self.prior = prior
        self.children = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.q_init = q_init 

    def is_expanded(self):
        return len(self.children) > 0

    def q_value(self): 
        # If unvisited, return the exact Q-value predicted by the neural network's after-state!
        if self.visit_count == 0: return self.q_init
        return self.value_sum / self.visit_count

class ExpertBlokusBot:
    def __init__(self, name="ExpertZero", model=None, pipe=None, shared_data=None, is_training=False):
        self.name = name
        self.model = model
        self.pipe = pipe
        self.shared_data = shared_data
        self.is_training = is_training
        self.c_puct = 1.5

        self.shape_keys = list(SHAPES.keys())
        self.shape_to_id = {name: idx for idx, name in enumerate(self.shape_keys)}

    def get_play(self, board, color, inventories, first_moves, pass_count):
        legal_moves = self._get_legal_moves(board, color, inventories, first_moves)
        if not legal_moves: return None, []
        action = self.get_action(board, color, inventories, first_moves, legal_moves, self.is_training)
        return self._decode_action(action, legal_moves)

    def get_action(self, board, color, inventories, first_moves, legal_moves, is_training=True, fast_playout=False):
        # 🛑 AFTER-STATE MCTS: Because evaluating 1 node now evaluates ALL its after-states natively, 
        # we require vastly fewer simulations to achieve superhuman depth.
        num_simulations = 6 if fast_playout else 25
        
        root = ExpertNode(board, color, inventories, first_moves)
        self._expand_and_evaluate(root, legal_moves)

        if is_training:
            noise = np.random.dirichlet([0.05] * len(legal_moves))
            for i, child in enumerate(root.children.values()):
                child.prior = 0.75 * child.prior + 0.25 * noise[i]

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            # Selection Phase
            while node.is_expanded():
                best_score = -float('inf')
                best_child = None
                for child in node.children.values():
                    q = child.q_value() 
                    u = self.c_puct * child.prior * math.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
                    if q + u > best_score:
                        best_score, best_child = q + u, child
                node = best_child
                search_path.append(node)
                
            # Expansion Phase
            node_legal_moves = self._get_legal_moves(node.board, node.current_color, node.inventories, node.first_moves)
            if node_legal_moves:
                v_leaf = self._expand_and_evaluate(node, node_legal_moves)
            else:
                v_leaf = node.q_value() # Terminal/Pass

            # Backpropagation Phase
            for n in reversed(search_path):
                n.visit_count += 1
                n.value_sum += v_leaf

        visits = [root.children[a].visit_count for a in legal_moves]
        if is_training:
            sum_visits = sum(visits)
            pi_probs = [v / sum_visits for v in visits] if sum_visits > 0 else [1.0/len(legal_moves)] * len(legal_moves)
            chosen_idx = np.random.choice(len(legal_moves), p=pi_probs)
        else:
            chosen_idx = np.argmax(visits)

        return legal_moves[chosen_idx]

    def _expand_and_evaluate(self, node, legal_moves):
        """Generates all After-States for the legal moves, evaluates them in a batch, and creates children."""
        after_states = []
        for action in legal_moves:
            shape_name, coords = action[1], action[5]
            
            next_board = [row[:] for row in node.board]
            for r, c in coords: next_board[r][c] = node.current_color
                
            next_inv = {k: list(v) for k, v in node.inventories.items()}
            next_inv[node.current_color].remove(shape_name)
            
            next_first = dict(node.first_moves)
            next_first[node.current_color] = False
            next_color = (node.current_color % 4) + 1
            
            astate = self._build_state_tensor(next_board, next_color, next_inv, next_first)
            after_states.append(astate)
            
        batch_size = len(after_states)
        
        # 🛑 Send the After-States directly to the GPU to evaluate their Q-Values
        if self.shared_data:
            w_id, s_states, s_values, s_scores = self.shared_data
            s_states[w_id, :batch_size] = after_states
            self.pipe.send(batch_size) 
            self.pipe.recv()     
            values = s_values[w_id, :batch_size].copy()
        else:
            import tensorflow as tf
            preds = self.model.predict(np.array(after_states), verbose=0)
            values = preds[0].flatten()

        next_color = (node.current_color % 4) + 1
        is_enemy = (node.current_color % 2) != (next_color % 2)
        
        q_values = np.zeros(batch_size)
        for i in range(batch_size):
            # If the after-state is the enemy's turn, a high value for them is a negative Q-value for us!
            q_values[i] = -values[i] if is_enemy else values[i]
            
        # Convert Q-values to Softmax Policy Priors
        temperature = 0.25 if self.is_training else 0.1
        exp_q = np.exp((q_values - np.max(q_values)) / temperature)
        priors = exp_q / np.sum(exp_q)

        for i, action in enumerate(legal_moves):
            child_board = [row[:] for row in node.board]
            for r, c in action[5]: child_board[r][c] = node.current_color
            child_inv = {k: list(v) for k, v in node.inventories.items()}
            child_inv[node.current_color].remove(action[1])
            child_first = dict(node.first_moves)
            child_first[node.current_color] = False
            
            node.children[action] = ExpertNode(
                board=child_board, current_color=next_color, inventories=child_inv, first_moves=child_first,
                parent=node, action_id=action[0], prior=priors[i], q_init=q_values[i]
            )
            
        return np.max(q_values) 

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
        if not my_shapes: return []
        
        board_np = np.array(board, dtype=int)
        
        if is_first:
            corner_mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool)
            if color == 1: corner_mask[0, 0] = True
            elif color == 2: corner_mask[0, 19] = True
            elif color == 3: corner_mask[19, 19] = True
            elif color == 4: corner_mask[19, 0] = True
            forbidden_mask = (board_np != 0)
        else:
            color_mask = (board_np == color)
            up = np.zeros_like(color_mask); up[:-1, :] = color_mask[1:, :]
            down = np.zeros_like(color_mask); down[1:, :] = color_mask[:-1, :]
            left = np.zeros_like(color_mask); left[:, :-1] = color_mask[:, 1:]
            right = np.zeros_like(color_mask); right[:, 1:] = color_mask[:, :-1]
            forbidden_mask = up | down | left | right | (board_np != 0)
            
            ul = np.zeros_like(color_mask); ul[:-1, :-1] = color_mask[1:, 1:]
            ur = np.zeros_like(color_mask); ur[:-1, 1:] = color_mask[1:, :-1]
            dl = np.zeros_like(color_mask); dl[1:, :-1] = color_mask[:-1, 1:]
            dr = np.zeros_like(color_mask); dr[1:, 1:] = color_mask[:-1, :-1]
            corner_mask = (ul | ur | dl | dr) & (~forbidden_mask)
            
        valid_corners = np.argwhere(corner_mask)
        if len(valid_corners) == 0: return []

        seen_moves = set()
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
                    min_r, min_c = min(r for r, c in curr_shape), min(c for r, c in curr_shape)
                    norm_shape = tuple(sorted((r - min_r, c - min_c) for r, c in curr_shape))
                    if norm_shape not in seen_hashes:
                        seen_hashes.add(norm_shape)
                        transforms.append((orient_id, norm_shape))
                    orient_id += 1
                    
            for orient_idx, shape_coords in transforms:
                max_r = max(r for r, c in shape_coords)
                max_c = max(c for r, c in shape_coords)
                for cr, cc in valid_corners:
                    for br, bc in shape_coords:
                        r, c = cr - br, cc - bc
                        if r < 0 or c < 0 or r + max_r >= BOARD_SIZE or c + max_c >= BOARD_SIZE: continue
                            
                        move_hash = (shape_idx, orient_idx, r, c)
                        if move_hash in seen_moves: continue
                        seen_moves.add(move_hash)
                        
                        valid = False
                        for sr, sc in shape_coords:
                            tr, tc = r + sr, c + sc
                            if forbidden_mask[tr, tc]:
                                valid = False
                                break
                            if corner_mask[tr, tc]: valid = True
                        
                        if valid:
                            action_id = (shape_idx * 8 * 400) + (orient_idx * 400) + (r * 20) + c
                            shifted = tuple((r + sr, c + sc) for sr, sc in shape_coords)
                            legal_moves.append((action_id, shape_name, orient_idx, r, c, shifted))
                            
        return legal_moves

    def _decode_action(self, action, legal_moves):
        if not action: return None, []
        return action[1], action[5]
