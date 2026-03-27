import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import math

class AdvancedBlokusModel:
    def __init__(self, board_size=20, num_blocks=10, filters=128):
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.filters = filters
        self.model = self.build_unified_model()

    def build_unified_model(self):
        inputs = layers.Input(shape=(self.board_size, self.board_size, 6))

        x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # Advanced ResNet with Global Pooling Injection
        for _ in range(self.num_blocks):
            shortcut = x
            x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(self.filters, 3, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            
            # Global Broadcast: Instantly transmit inventory/board-state to all squares
            g = layers.GlobalAveragePooling2D()(x)
            g = layers.Dense(self.filters, use_bias=False)(g)
            g = layers.Reshape((1, 1, self.filters))(g)
            x = layers.Add()([x, g]) 
            
            x = layers.Add()([shortcut, x])
            x = layers.Activation('relu')(x)

        # 1. POLICY HEAD
        p = layers.Conv2D(2, 1, padding='same', use_bias=False)(x)
        p = layers.BatchNormalization()(p)
        p = layers.Activation('relu')(p)
        p = layers.Flatten()(p)
        policy_out = layers.Dense(self.board_size * self.board_size * 4, activation='softmax', name='policy')(p)

        # 2. VALUE HEAD (Win/Loss)
        v = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        v = layers.BatchNormalization()(v)
        v = layers.Activation('relu')(v)
        v = layers.Flatten()(v)
        v = layers.Dense(256, activation='relu')(v)
        value_out = layers.Dense(1, activation='tanh', name='value')(v)

        # 3. SCORE LEAD HEAD (Auxiliary: Expected Point Difference)
        s = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        s = layers.BatchNormalization()(s)
        s = layers.Activation('relu')(s)
        s = layers.Flatten()(s)
        s = layers.Dense(256, activation='relu')(s)
        score_out = layers.Dense(1, name='score_lead')(s)  

        # 4. OWNERSHIP HEAD (Auxiliary: Who owns each square at the end: 0=Empty, 1-4=Player)
        own = layers.Conv2D(5, 1, padding='same', name='ownership_conv')(x)
        ownership_out = layers.Reshape((self.board_size, self.board_size, 5))(own)
        ownership_out = layers.Softmax(axis=-1, name='ownership')(ownership_out)

        model = models.Model(inputs=inputs, outputs=[policy_out, value_out, score_out, ownership_out])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mean_squared_error',
                'score_lead': 'huber',  # Huber ignores extreme score outliers
                'ownership': 'sparse_categorical_crossentropy' 
            },
            loss_weights={
                'policy': 1.0,
                'value': 1.0,
                'score_lead': 0.05, # Auxiliary signal
                'ownership': 0.05   # Auxiliary signal
            }
        )
        return model

class ExpertNode:
    def __init__(self, prior, parent=None):
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.score_sum = 0.0

    def q_value(self): return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    def q_score(self): return self.score_sum / self.visit_count if self.visit_count > 0 else 0

    def expand(self, action_probs):
        for action, prob in action_probs:
            if action not in self.children:
                self.children[action] = ExpertNode(prior=prob, parent=self)

class ExpertBlokusBot:
    def __init__(self, name="AlphaZero", model=None, pipe=None, is_training=False):
        self.name = name
        self.model = model
        self.pipe = pipe
        self.is_training = is_training
        self.c_puct = 1.5
        self.score_utility_weight = 0.02 # Make the bot fight for points, not just win %

    def get_play(self, board, color, inventories, first_moves, pass_count):
        """UI COMPATIBILITY WRAPPER: Used by app.py"""
        state_tensor = self._build_state_tensor(board, color, inventories, first_moves)
        legal_moves = self._get_legal_moves(board, color, inventories, first_moves)
        
        if not legal_moves:
            return None, []
            
        action, _ = self.get_action(state_tensor, legal_moves, self.is_training)
        return self._decode_action(action, legal_moves)

    def get_action(self, state_tensor, legal_moves, is_training=True, fast_playout=False):
        # Playout Cap Randomization checks
        num_simulations = 50 if fast_playout else 400
        root = ExpertNode(1.0)
        
        if is_training:
            noise = np.random.dirichlet([0.03] * len(legal_moves))
            action_probs = [(action, 0.75 * (1.0/len(legal_moves)) + 0.25 * n) for action, n in zip(legal_moves, noise)]
        else:
            action_probs = [(action, 1.0/len(legal_moves)) for action in legal_moves]
            
        root.expand(action_probs)

        for _ in range(num_simulations):
            node = root
            search_path = [node]
            
            while len(node.children) > 0:
                best_score = -float('inf')
                best_action, best_child = None, None
                
                for action, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(node.visit_count) / (1 + child.visit_count)
                    # Mix Win Rate AND Expected Points
                    q_utility = child.q_value() + (self.score_utility_weight * child.q_score())
                    
                    if q_utility + u > best_score:
                        best_score, best_action, best_child = q_utility + u, action, child
                        
                node = best_child
                search_path.append(node)
                
            # INFERENCE: Uses Pipe for multiprocessing, or direct model predict
            if self.pipe:
                self.pipe.send(state_tensor)
                policy, value, score_lead, ownership = self.pipe.recv()
            else:
                policy, value, score_lead, ownership = self.model.predict(np.array([state_tensor]), verbose=0)
                policy, value, score_lead = policy[0], value[0][0], score_lead[0][0]
            
            # Backpropagate both Value and Score Difference
            for node in reversed(search_path):
                node.visit_count += 1
                node.value_sum += value
                node.score_sum += score_lead
                value = -value 
                score_lead = -score_lead 

        visits = [root.children[a].visit_count if a in root.children else 0 for a in legal_moves]
        if is_training:
            visits = np.array(visits) ** (1.0 / 1.0)
            pi = visits / (np.sum(visits) + 1e-8)
            chosen_idx = np.random.choice(len(legal_moves), p=pi)
        else:
            chosen_idx = np.argmax(visits)
            pi = np.zeros(len(legal_moves))
            pi[chosen_idx] = 1.0

        return legal_moves[chosen_idx], pi

    def _build_state_tensor(self, board, color, inventories, first_moves):
        # Implementation from your previous helper/logic
        return np.zeros((20, 20, 6))

    def _get_legal_moves(self, board, color, inventories, first_moves):
        # Implementation from your previous helper/logic
        return [0, 1, 2] # Dummy

    def _decode_action(self, action, legal_moves):
        # Implementation from your previous helper/logic
        return "shape_1", [(0,0)] # Dummy
