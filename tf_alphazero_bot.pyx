# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from helper import BOARD_SIZE, SHAPES, flip_shape, rotate_shape
import concurrent.futures

class AdvancedBlokusModel:
    def __init__(self, board_size=20, num_blocks=4, filters=16):
        self.board_size = board_size
        self.num_blocks = num_blocks
        self.filters = filters
        self.model = self.build_unified_model()

    def build_unified_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models

        inputs = layers.Input(shape=(self.board_size, self.board_size, 8))
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

        v = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        v = layers.BatchNormalization()(v)
        v = layers.Activation('relu')(v)
        v = layers.Flatten()(v)
        v = layers.Dense(256, activation='relu')(v)
        value_out = layers.Dense(1, activation='tanh', name='value', dtype='float32')(v)

        s = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
        s = layers.BatchNormalization()(s)
        s = layers.Activation('relu')(s)
        s = layers.Flatten()(s)
        s = layers.Dense(256, activation='relu')(s)
        score_out = layers.Dense(1, name='score_lead', dtype='float32')(s)  

        model = models.Model(inputs=inputs, outputs=[value_out, score_out])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'value': 'mean_squared_error', 'score_lead': 'huber'},
            loss_weights={'value': 1.0, 'score_lead': 0.05}
        )
        return model

cdef class ExpertNode:
    cdef public double prior
    cdef public double q_init
    cdef public dict children
    cdef public int visit_count
    cdef public double value_sum
    cdef public int virtual_loss 
    cdef public object lock       

    def __init__(self, double prior=0.0, double q_init=0.0):
        import threading
        self.prior = prior
        self.q_init = q_init
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.lock = threading.Lock()

    cdef bint is_expanded(self):
        return len(self.children) > 0

    cdef double q_value(self): 
        cdef int effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0: return self.q_init
        return (self.value_sum - self.virtual_loss) / effective_visits

cdef list get_valid_corners(int[:, :] board, int color, bint is_first):
    cdef int r, c
    cdef list corners = []
    cdef bint is_adj_color
    
    if is_first:
        if color == 1 and board[0, 0] == 0: corners.append((0, 0))
        elif color == 2 and board[0, 19] == 0: corners.append((0, 19))
        elif color == 3 and board[19, 19] == 0: corners.append((19, 19))
        elif color == 4 and board[19, 0] == 0: corners.append((19, 0))
        return corners
        
    for r in range(20):
        for c in range(20):
            if board[r, c] != 0:
                continue
            
            is_adj_color = False
            if r > 0 and board[r-1, c] == color: is_adj_color = True
            elif r < 19 and board[r+1, c] == color: is_adj_color = True
            elif c > 0 and board[r, c-1] == color: is_adj_color = True
            elif c < 19 and board[r, c+1] == color: is_adj_color = True
            
            if is_adj_color:
                continue 
                
            if r > 0 and c > 0 and board[r-1, c-1] == color: corners.append((r, c))
            elif r > 0 and c < 19 and board[r-1, c+1] == color: corners.append((r, c))
            elif r < 19 and c > 0 and board[r+1, c-1] == color: corners.append((r, c))
            elif r < 19 and c < 19 and board[r+1, c+1] == color: corners.append((r, c))
            
    return list(set(corners))

cdef bint check_valid_placement(int[:, :] board, int color, int r, int c, list shape_coords, bint is_first):
    cdef int sr, sc, tr, tc
    cdef bint touches_valid = False
    
    for sr, sc in shape_coords:
        tr = r + sr
        tc = c + sc
        if tr < 0 or tc < 0 or tr >= 20 or tc >= 20: return False
        if board[tr, tc] != 0: return False
        
        if tr > 0 and board[tr-1, tc] == color: return False
        if tr < 19 and board[tr+1, tc] == color: return False
        if tc > 0 and board[tr, tc-1] == color: return False
        if tc < 19 and board[tr, tc+1] == color: return False
        
        if is_first:
            if color == 1 and tr == 0 and tc == 0: touches_valid = True
            elif color == 2 and tr == 0 and tc == 19: touches_valid = True
            elif color == 3 and tr == 19 and tc == 19: touches_valid = True
            elif color == 4 and tr == 19 and tc == 0: touches_valid = True
        else:
            if tr > 0 and tc > 0 and board[tr-1, tc-1] == color: touches_valid = True
            elif tr > 0 and tc < 19 and board[tr-1, tc+1] == color: touches_valid = True
            elif tr < 19 and tc > 0 and board[tr+1, tc-1] == color: touches_valid = True
            elif tr < 19 and tc < 19 and board[tr+1, tc+1] == color: touches_valid = True
            
    return touches_valid

cdef class ExpertBlokusBot:
    cdef public str name
    cdef public object model
    cdef public object pipe
    cdef public object shared_data
    cdef public bint is_training
    cdef public double c_puct
    cdef public list shape_keys
    cdef public dict shape_to_id
    cdef public dict precomputed_shapes
    cdef public object pipe_lock
    cdef public object executor

    def __init__(self, name="ExpertZero", model=None, pipe=None, shared_data=None, is_training=False):
        import threading
        import concurrent.futures
        self.name = name
        self.model = model
        self.pipe = pipe
        self.shared_data = shared_data
        self.is_training = is_training
        self.c_puct = 1.5
        self.pipe_lock = threading.Lock()
        
        # 🚀 FIX: Persistent executor to prevent RAM thread stack fragmentation
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        self.shape_keys = list(SHAPES.keys())
        self.shape_to_id = {name: idx for idx, name in enumerate(self.shape_keys)}
        
        self.precomputed_shapes = {}
        for shape_name, base_shape in SHAPES.items():
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
                        transforms.append((orient_id, list(norm_shape)))
                    orient_id += 1
            self.precomputed_shapes[shape_name] = transforms

    cpdef list _get_legal_moves(self, list board, int color, dict inventories, dict first_moves):
        cdef np.ndarray board_np = np.array(board, dtype=np.int32)
        cdef int[:, :] board_view = board_np
        cdef list my_shapes
        
        if color in inventories: my_shapes = inventories[color]
        elif str(color) in inventories: my_shapes = inventories[str(color)]
        else: my_shapes = []
        
        return self.c_get_legal_moves(board_view, color, my_shapes, first_moves[color])

    cpdef np.ndarray _build_state_tensor(self, list board, int color, dict inventories, dict first_moves):
        cdef np.ndarray board_np = np.array(board, dtype=np.int32)
        cdef int[:, :] board_view = board_np
        return self.c_build_state_tensor(board_view, color, first_moves)

    cpdef get_play(self, list board, int color, dict inventories, dict first_moves, int pass_count):
        cdef np.ndarray board_np = np.array(board, dtype=np.int32)
        cdef int[:, :] board_view = board_np
        cdef list my_shapes
        
        if color in inventories: my_shapes = inventories[color]
        elif str(color) in inventories: my_shapes = inventories[str(color)]
        else: my_shapes = []
        
        cdef list legal_moves = self.c_get_legal_moves(board_view, color, my_shapes, first_moves[color])
        if not legal_moves: return None, []
        action = self.get_action(board, color, inventories, first_moves, legal_moves, self.is_training)
        return self._decode_action(action, legal_moves)

    def _run_single_simulation(self, ExpertNode root, np.ndarray board_np, int color, dict inventories, dict first_moves):
        cdef ExpertNode node = root
        cdef list search_path = [node]
        cdef list colors_in_path = [color]
        
        cdef np.ndarray curr_board_np = np.array(board_np, copy=True, dtype=np.int32)
        cdef int[:, :] curr_board_view = curr_board_np
        
        cdef dict curr_inv = {k: list(v) for k, v in inventories.items()}
        cdef dict curr_first = dict(first_moves)
        cdef int curr_color = color
        cdef double best_score = -999999.0
        cdef double q = 0.0, u = 0.0
        cdef tuple best_action = None
        cdef ExpertNode best_child = None
        cdef ExpertNode child = None
        cdef ExpertNode n = None
        cdef str shape_name
        cdef tuple shifted_coords
        cdef list my_shapes
        cdef list node_legal_moves
        cdef double v_leaf = 0.0
        cdef int r, c, step_color 
        
        while True:
            with node.lock:
                if not node.is_expanded():
                    break
                best_score = -999999.0
                best_action = None
                best_child = None
                for action, child_obj in node.children.items():
                    child = <ExpertNode>child_obj
                    q = child.q_value()
                    u = self.c_puct * child.prior * sqrt(max(1.0, float(node.visit_count + node.virtual_loss))) / (1.0 + child.visit_count + child.virtual_loss)
                    if q + u > best_score:
                        best_score = q + u
                        best_action = action
                        best_child = child
                
                best_child.virtual_loss += 3
                node = best_child
                search_path.append(node)
            
            shape_name = best_action[1]
            shifted_coords = best_action[5]
            
            for r_c in shifted_coords:
                r = r_c[0]
                c = r_c[1]
                curr_board_view[r, c] = curr_color
                
            curr_inv[curr_color].remove(shape_name)
            curr_first[curr_color] = False
            curr_color = (curr_color % 4) + 1
            colors_in_path.append(curr_color)

        if curr_color in curr_inv:
            my_shapes = curr_inv[curr_color]
        elif str(curr_color) in curr_inv:
            my_shapes = curr_inv[str(curr_color)]
        else:
            my_shapes = []

        node_legal_moves = self.c_get_legal_moves(curr_board_view, curr_color, my_shapes, curr_first[curr_color])
        
        if node_legal_moves:
            v_leaf = self._expand_and_evaluate(node, curr_board_view, curr_color, curr_inv, curr_first, node_legal_moves)
        else:
            v_leaf = node.q_value()
            
        for n_obj, step_color in zip(search_path, colors_in_path):
            n = <ExpertNode>n_obj
            with n.lock:
                if n is not root:
                    n.virtual_loss -= 3
                n.visit_count += 1
                if (step_color % 2) == (curr_color % 2):
                    n.value_sum += v_leaf
                else:
                    n.value_sum -= v_leaf

    cpdef get_action(self, list board, int color, dict inventories, dict first_moves, list legal_moves, bint is_training=True, bint fast_playout=False):
        cdef int num_simulations = 6 if fast_playout else 32
        cdef ExpertNode root = ExpertNode()
        cdef np.ndarray board_np = np.array(board, dtype=np.int32)
        cdef int[:, :] board_view = board_np
        
        self._expand_and_evaluate(root, board_view, color, inventories, first_moves, legal_moves)

        if is_training:
            noise = np.random.dirichlet([0.05] * len(legal_moves))
            for idx, action in enumerate(legal_moves):
                (<ExpertNode>root.children[action]).prior = 0.75 * (<ExpertNode>root.children[action]).prior + 0.25 * noise[idx]

        import concurrent.futures
        futures = [self.executor.submit(self._run_single_simulation, root, board_np, color, inventories, first_moves) for _ in range(num_simulations)]
        concurrent.futures.wait(futures)

        cdef list visits = [(<ExpertNode>root.children[a]).visit_count for a in legal_moves]
        cdef int chosen_idx
        
        if is_training:
            sum_visits = sum(visits)
            if sum_visits > 0:
                pi_probs = [v / sum_visits for v in visits] 
            else:
                pi_probs = [1.0/len(legal_moves)] * len(legal_moves)
            chosen_idx = np.random.choice(len(legal_moves), p=pi_probs)
        else:
            chosen_idx = np.argmax(visits)

        return legal_moves[chosen_idx]

    cdef double _expand_and_evaluate(self, ExpertNode node, int[:, :] board_view, int color, dict inventories, dict first_moves, list legal_moves):
        cdef list after_states = []
        cdef int next_color = (color % 4) + 1
        cdef np.ndarray next_board_np
        cdef int[:, :] next_board_view
        cdef int r, c
        cdef tuple action, shifted_coords
        cdef str shape_name
        cdef dict next_inv, next_first
        
        for action in legal_moves:
            shape_name = action[1]
            shifted_coords = action[5]
            
            next_board_np = np.array(board_view, copy=True, dtype=np.int32)
            next_board_view = next_board_np
            
            for r_c in shifted_coords:
                r = r_c[0]
                c = r_c[1]
                next_board_view[r, c] = color
                
            next_inv = {k: list(v) for k, v in inventories.items()}
            next_inv[color].remove(shape_name)
            next_first = dict(first_moves)
            next_first[color] = False
            
            after_states.append(self.c_build_state_tensor(next_board_view, next_color, next_first))
            
        cdef int batch_size = len(after_states)
        cdef np.ndarray values
        cdef int i
        
        if self.shared_data is not None:
            w_id, s_states, s_values, s_scores = self.shared_data
            with self.pipe_lock:
                s_states[w_id, :batch_size] = after_states
                self.pipe.send(batch_size)
                self.pipe.recv()
                values = s_values[w_id, :batch_size].copy()
        else:
            with self.pipe_lock:
                import tensorflow as tf
                preds = self.model.predict(np.array(after_states), verbose=0)
                values = preds[0].flatten()

        cdef bint is_enemy = (color % 2) != (next_color % 2)
        cdef np.ndarray q_values = np.zeros(batch_size, dtype=np.float64)
        cdef double max_q = -999999.0
        
        for i in range(batch_size):
            q_values[i] = -values[i] if is_enemy else values[i]
            if q_values[i] > max_q:
                max_q = q_values[i]
                
        cdef double temperature = 0.25 if self.is_training else 0.1
        cdef np.ndarray exp_q = np.exp((q_values - max_q) / temperature)
        cdef np.ndarray priors = exp_q / np.sum(exp_q)

        with node.lock:
            for i, action in enumerate(legal_moves):
                if action not in node.children:
                    node.children[action] = ExpertNode(prior=priors[i], q_init=q_values[i])
            
        return max_q

    cdef np.ndarray c_build_state_tensor(self, int[:, :] board_view, int color, dict first_moves):
        cdef np.ndarray tensor = np.zeros((20, 20, 8), dtype=np.float32)
        cdef float[:, :, :] tensor_view = tensor
        cdef int r, c, p
        cdef float color_val = color / 4.0
        cdef float first_val = 1.0 if first_moves[color] else 0.0
        
        for r in range(20):
            for c in range(20):
                p = board_view[r, c]
                if p > 0:
                    tensor_view[r, c, p-1] = 1.0
                tensor_view[r, c, 4] = color_val
                tensor_view[r, c, 5] = first_val
                
        return tensor

    cdef list c_get_legal_moves(self, int[:, :] board_view, int color, list my_shapes, bint is_first):
        cdef list legal_moves = []
        cdef int shape_idx, orient_idx, r, c, cr, cc, br, bc, max_r, max_c, sr, sc
        cdef str shape_name
        cdef list shape_coords, shifted
        cdef tuple move_hash
        cdef set seen_moves = set()
        cdef list valid_corners = get_valid_corners(board_view, color, is_first)
        
        if not valid_corners:
            return []
            
        for shape_name in my_shapes:
            shape_idx = self.shape_to_id[shape_name]
            transforms = self.precomputed_shapes[shape_name]
            
            for orient_idx, shape_coords in transforms:
                max_r = 0
                max_c = 0
                for sr, sc in shape_coords:
                    if sr > max_r: max_r = sr
                    if sc > max_c: max_c = sc
                    
                for cr, cc in valid_corners:
                    for br, bc in shape_coords:
                        r = cr - br
                        c = cc - bc
                        
                        if r < 0 or c < 0 or r + max_r >= 20 or c + max_c >= 20: continue
                            
                        move_hash = (shape_idx, orient_idx, r, c)
                        if move_hash in seen_moves: continue
                        seen_moves.add(move_hash)
                        
                        if check_valid_placement(board_view, color, r, c, shape_coords, is_first):
                            action_id = (shape_idx * 8 * 400) + (orient_idx * 400) + (r * 20) + c
                            shifted = [(r + sr, c + sc) for sr, sc in shape_coords]
                            legal_moves.append((action_id, shape_name, orient_idx, r, c, tuple(shifted)))
                                
        return legal_moves

    cpdef _decode_action(self, action, list legal_moves):
        if not action: return None, []
        return action[1], list(action[5])
