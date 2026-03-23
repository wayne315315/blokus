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
        
    # AlphaZero Style Heads
    if is_policy:
        # Policy Head
        x = layers.Conv2D(2, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1, activation='linear', dtype='float32')(x) # Single logit per action
    else:
        # Value Head
        x = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(1, activation='tanh', dtype='float32')(x) # Win/Loss evaluation [-1, 1]

    return models.Model(inputs=inputs, outputs=x)

def create_value_network():
    return create_spatial_resnet(is_policy=False)

def create_policy_network():
    return create_spatial_resnet(is_policy=True)

# ==============================================================================
# ALPHAZERO BLOKUS BOT
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
        # Channels: 0=Blue, 1=Yellow, 2=Red, 3=Green, 4=Proposed Move, 5=Turn Indicator
        seq = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)
        
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                cell = board[r][c]
                if cell > 0:
                    seq[r, c, cell - 1] = 1.0
                    
        seq[:, :, 5] = color_id / 4.0 # Normalize turn indicator
        return seq

    def get_play(self, board, color_id, available_shapes, is_first_move):
        legal_actions = self._get_legal_actions(board, color_id, available_shapes, is_first_move)
        
        if not legal_actions:
            return None, None
            
        if len(legal_actions) == 1:
            return legal_actions[0]
            
        base_board = self._encode_base_board(board, color_id)
        batch_inputs = np.empty((len(legal_actions), 20, 20, 6), dtype=np.float32)

        for i, action in enumerate(legal_actions):
            np.copyto(batch_inputs[i], base_board) 
            coords = action[1]
            for r, c in coords:
                batch_inputs[i, r, c, 4] = 1.0 # Plot the proposed move on Channel 4
        
        if self.is_training:
            if self.pipe:
                self.pipe.send((False, batch_inputs.astype(np.float16)))
                values = self.pipe.recv().flatten().tolist()
                
                self.pipe.send((True, batch_inputs.astype(np.float16)))
                policy_logits = self.pipe.recv().flatten()
            else:
                values = self.val_net(batch_inputs, training=False).numpy().flatten().tolist()
                policy_logits = self.policy_net(batch_inputs, training=False).numpy().flatten()
            
            # Softmax Policy
            exp_logits = np.exp(policy_logits - np.max(policy_logits))
            probabilities = (exp_logits / np.sum(exp_logits)).tolist()

            if self.exploration_rate > 0:
                explore_prob = self.exploration_rate / len(legal_actions)
                final_probs = [(p * (1.0 - self.exploration_rate)) + explore_prob for p in probabilities]
            else:
                final_probs = probabilities
                
            chosen_index = random.choices(range(len(legal_actions)), weights=final_probs, k=1)[0]
            
            self.episode_memory.append({
                'inputs': batch_inputs, 'chosen_index': chosen_index, 'value_pred': values[chosen_index]
            })
            for i in range(len(legal_actions)):
                self.policy_memory.append((batch_inputs[i], probabilities[i]))
                
            return legal_actions[chosen_index]
            
        else:
            if self.pipe:
                self.pipe.send((True, batch_inputs.astype(np.float16)))
                policy_logits = self.pipe.recv().flatten()
            else:
                policy_logits = self.policy_net(batch_inputs, training=False).numpy().flatten()
                
            best_action_idx = np.argmax(policy_logits)
            return legal_actions[best_action_idx]
