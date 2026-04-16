import os
import sys
import time
import glob
import multiprocessing as mp
import multiprocessing.connection
import ctypes
import numpy as np
import threading
import concurrent.futures
from collections import deque
import pickle 

# 🚀 OS MEMORY FIX: Force thread stacks to 256KB to prevent Thread Stack Fragmentation
threading.stack_size(256 * 1024)

from helper import BOARD_SIZE, SHAPES

# 🚀 TF GC FIX: Prevents TensorFlow from dynamically tracing thousands of graphs
MAX_ORDER = 16
_NUM_CPUS = mp.cpu_count()
NUM_WORKERS = min(31, _NUM_CPUS - 1 if _NUM_CPUS > 1 else 1)

MAX_CAPACITY = 2048

NUM_THREADS = 2
TOTAL_THREADS = NUM_WORKERS * NUM_THREADS

def generate_expert_game(bot):
    states, players = [], []
    board = [[0 for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    inventories = {i: list(SHAPES.keys()) for i in range(1, 5)} 
    first_moves = {1: True, 2: True, 3: True, 4: True}
    current_player = 1
    pass_count = 0
    
    while pass_count < 4:
        legal_moves = bot._get_legal_moves(board, current_player, inventories, first_moves)
        if not legal_moves:
            pass_count += 1
            current_player = (current_player % 4) + 1
            continue
            
        pass_count = 0
        is_fast_playout = np.random.rand() < 0.80 
        
        state_tensor = bot._build_state_tensor(board, current_player, inventories, first_moves)
        action = bot.get_action(board, current_player, inventories, first_moves, legal_moves, is_training=True, fast_playout=is_fast_playout)
        
        if not is_fast_playout:
            states.append(state_tensor)
            players.append(current_player)
            
        shape_name, coords = bot._decode_action(action, legal_moves)
        for r, c in coords: board[r][c] = current_player
        if shape_name in inventories[current_player]: inventories[current_player].remove(shape_name)
        first_moves[current_player] = False
        current_player = (current_player % 4) + 1
        
    def get_score(pid): return sum(int(s.split('_')[0]) for s in inventories.get(pid, []))
    team1_score, team2_score = get_score(1) + get_score(3), get_score(2) + get_score(4)
    
    val_targets, score_targets = [], []
    for p in players:
        if p in [1, 3]:
            val_targets.append(1 if team1_score < team2_score else -1)
            score_targets.append(team2_score - team1_score) 
        else:
            val_targets.append(1 if team2_score < team1_score else -1)
            score_targets.append(team1_score - team2_score)

    total_turns_played = 84 - sum(len(inv) for inv in inventories.values())
    return states, val_targets, score_targets, total_turns_played

def single_game_thread(games_per_thread, conn, thread_idx, shared_data_bases, total_threads, shared_counter):
    from tf_alphazero_bot import ExpertBlokusBot
    
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(thread_idx, shared_states, shared_values, shared_scores), is_training=True)
    
    S, V, SC = [], [], []
    thread_total_turns = 0 
    
    for _ in range(games_per_thread):
        s, v, sc, turns = generate_expert_game(bot)
        S.extend(s); V.extend(v); SC.extend(sc)
        thread_total_turns += turns
        with shared_counter.get_lock(): shared_counter.value += 1
            
    conn.send("DONE")
    if hasattr(bot, 'executor') and bot.executor is not None:
        bot.executor.shutdown(wait=False)
    return S, V, SC, thread_total_turns

def distributed_train_worker(games_per_thread, conns, result_queue, thread_indices, shared_counter, shared_data_bases, total_threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    threading.stack_size(256 * 1024)
    
    S, V, SC = [], [], []
    worker_total_turns = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(conns)) as executor:
        futures = []
        for conn, tid in zip(conns, thread_indices):
            futures.append(executor.submit(single_game_thread, games_per_thread, conn, tid, shared_data_bases, total_threads, shared_counter))
        
        for f in concurrent.futures.as_completed(futures):
            s, v, sc, turns = f.result()
            S.extend(s); V.extend(v); SC.extend(sc)
            worker_total_turns += turns
            
    result_queue.put((S, V, SC, worker_total_turns))

def training_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, total_threads):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 8))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(total_threads)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()

    CHUNK_SIZE =  2 ** MAX_ORDER
    MIN_BATCH_SIZE = 64

    print("🔥 Allocating Static Server Buffer in System RAM...", flush=True)
    MAX_POSSIBLE_BATCH = total_threads * MAX_CAPACITY
    SERVER_BUFFER = np.empty((MAX_POSSIBLE_BATCH, 20, 20, 8), dtype=np.float32)
    SERVER_BUFFER.fill(0.0) 

    while active_conns:
        readable = multiprocessing.connection.wait(active_conns, timeout=0.001)
        for p in readable:
            try:
                msg = p.recv()
                if msg == "DONE":
                    active_conns.remove(p)
                else:
                    ready_indices.append(conn_to_id[p])
                    batch_sizes.append(msg) 
                    ready_pipes.append(p)
                    total_states_queued += msg
            except EOFError:
                if p in active_conns: active_conns.remove(p)

        if total_states_queued > 0:
            
            t0 = time.time()
            actual_size = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                SERVER_BUFFER[actual_size : actual_size + size] = shared_states[w_id, :size]
                actual_size += size
                
            batch_tensor = SERVER_BUFFER[:actual_size]
            t_copy = (time.time() - t0) * 1000 
            
            t1 = time.time()
            v_preds, sc_preds = [], []
            
            tensor_cursor = 0
            rem = actual_size
            
            while rem > 0:
                curr_size = min(rem, CHUNK_SIZE)
                
                pad_target = 1 << (curr_size - 1).bit_length() if curr_size > 0 else 0
                if pad_target < MIN_BATCH_SIZE: 
                    pad_target = MIN_BATCH_SIZE
                
                chunk = batch_tensor[tensor_cursor : tensor_cursor + curr_size]
                
                if pad_target > curr_size:
                    pad_array = np.zeros((pad_target - curr_size, 20, 20, 8), dtype=np.float32)
                    chunk_padded = np.concatenate([chunk, pad_array], axis=0)
                else:
                    chunk_padded = chunk
                    
                preds = fast_infer(chunk_padded)
                v_preds.append(preds[0].numpy().flatten()[:curr_size])
                sc_preds.append(preds[1].numpy().flatten()[:curr_size])
                
                tensor_cursor += curr_size
                rem -= curr_size
                
            t_infer = (time.time() - t1) * 1000 
            t_per_sample = (t_infer / actual_size) if actual_size > 0 else 0.0
            
            t2 = time.time()
            v_numpy = np.concatenate(v_preds)
            sc_numpy = np.concatenate(sc_preds)
            
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size
            t_write = (time.time() - t2) * 1000

            for pipe in ready_pipes: pipe.send(True)

            curr_time = time.time()
            if curr_time - last_print_time > 0.5:
                games_completed = shared_counter.value
                print(f"⚡ GPU BATCH {actual_size:<5} | "
                      f"Copy: {t_copy:>5.1f}ms | "
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.4f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Games: {games_completed}/{total_games}   ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All GPU processes finished gathering data.", flush=True)

def run_training_pipeline(num_iteration=1000):
    import tensorflow as tf
    from tf_alphazero_bot import AdvancedBlokusModel

    # Apple Silicon struggles with mixed precision, so we selectively disable it.
    is_mac = sys.platform == "darwin"
    if not is_mac:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    model_path = "blokus_expert_latest.keras"
    if os.path.exists(model_path):
        print(f"🚀 Found existing model! Resuming training from {model_path}...", flush=True)
        # 🚀 FIX 1: load with compile=True to preserve the Adam Optimizer's internal momentum state!
        model = tf.keras.models.load_model(model_path, compile=True)
    else:
        print("🚀 No existing model found. Building a new AdvancedBlokusModel from scratch...", flush=True)
        adv_model = AdvancedBlokusModel()
        model = adv_model.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), loss={'value': 'mean_squared_error', 'score_lead': 'huber'}, loss_weights={'value': 1.0, 'score_lead': 0.05})

    @tf.function(reduce_retracing=True, jit_compile=not is_mac)
    def fast_infer(batch_tensor): 
        return model(batch_tensor, training=False)

    print(f"🔥 Pre-compiling Power-of-2 XLA Buckets (64 to {2 ** MAX_ORDER}) into System RAM...", flush=True)
    for o in range(6, MAX_ORDER + 1):
        _ = fast_infer(tf.zeros((2 ** o, 20, 20, 8), dtype=tf.float32))
    print("✅ All dynamic graphs successfully compiled and cached!", flush=True)
    
    TOTAL_GAMES_PER_ITERATION = 1024
    games_per_thread = max(1, TOTAL_GAMES_PER_ITERATION // TOTAL_THREADS)
    actual_total_games = TOTAL_THREADS * games_per_thread

    # 🚀 REPLAY BUFFER INITIALIZATION
    REPLAY_BUFFER_SIZE = 200000
    start_iteration = 1
    
    # 🚀 FIX 2: CHECK IF HISTORICAL BUFFER EXISTS ON DISK
    buffer_path = "replay_buffer.pkl"
    if os.path.exists(buffer_path):
        print(f"💾 Found existing Replay Buffer! Loading historical states from {buffer_path}...", flush=True)
        with open(buffer_path, "rb") as f:
            data = pickle.load(f)
            # Safely unpack in case we are loading from an older buffer version
            if len(data) == 4:
                replay_states, replay_values, replay_scores, start_iteration = data
            else:
                replay_states, replay_values, replay_scores = data
                start_iteration = 1
        print(f"✅ Successfully loaded {len(replay_states)} historical states. Resuming at Iteration {start_iteration}.", flush=True)
    else:
        print(f"💾 No existing Replay Buffer found. Initializing empty queues.", flush=True)
        replay_states = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_values = deque(maxlen=REPLAY_BUFFER_SIZE)
        replay_scores = deque(maxlen=REPLAY_BUFFER_SIZE)

    print("🔥 Allocating Static Inter-Process Communication Arrays (Once)...", flush=True)
    ctx = mp.get_context('spawn')
    shared_counter = ctx.Value('i', 0)
    
    shared_states_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY * 20 * 20 * 8)
    shared_values_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY)
    shared_scores_base = mp.Array(ctypes.c_float, TOTAL_THREADS * MAX_CAPACITY)
    shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

    pipes = [ctx.Pipe() for _ in range(TOTAL_THREADS)]
    parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]

    for iteration in range(start_iteration, start_iteration + num_iteration):
        
        # 🚀 FIX 3: LEARNING RATE EXPONENTIAL SCHEDULING
        # Drops from 2e-4 by ~5% every iteration, with a hard floor at 1e-5
        current_lr = max(2e-4 * (0.95 ** (iteration - 1)), 1e-5)
        model.optimizer.learning_rate.assign(current_lr)

        print(f"\n" + "="*70, flush=True)
        print(f"🚀 STARTING Q-LEARNING ITERATION {iteration} | LR: {current_lr:.2e} | Generating {actual_total_games} Games", flush=True)
        print("="*70, flush=True)
        
        start_time = time.time()
        shared_counter.value = 0
        
        result_queue = ctx.Queue()
        processes = []
        
        for i in range(NUM_WORKERS):
            conns_for_worker = child_conns[i * NUM_THREADS : (i + 1) * NUM_THREADS]
            indices_for_worker = list(range(i * NUM_THREADS, (i + 1) * NUM_THREADS))
            
            p = ctx.Process(target=distributed_train_worker, args=(games_per_thread, conns_for_worker, result_queue, indices_for_worker, shared_counter, shared_data_bases, TOTAL_THREADS))
            p.start()
            processes.append(p)

        training_inference_server(parent_conns, fast_infer, shared_counter, actual_total_games, shared_data_bases, TOTAL_THREADS)

        S, V, SC = [], [], []
        iteration_total_turns = 0
        
        for _ in range(NUM_WORKERS):
            s, v, sc, turns = result_queue.get()
            S.extend(s); V.extend(v); SC.extend(sc)
            iteration_total_turns += turns

        for p in processes: p.join()
        
        generation_time = time.time() - start_time
        avg_game_length = iteration_total_turns / actual_total_games
        
        print(f"\n✅ Data Gen Complete in {generation_time:.1f}s | Extracted {len(S)} New States | EXACT Avg Turns/Game: {avg_game_length:.1f}", flush=True)
        
        # POPULATE THE REPLAY BUFFER
        replay_states.extend(S)
        replay_values.extend(V)
        replay_scores.extend(SC)
        
        current_buffer_size = len(replay_states)
        print(f"🧠 Replay Buffer Size: {current_buffer_size} / {REPLAY_BUFFER_SIZE}. Constructing Pipeline...", flush=True)
        # 🚀 FIX: Force the massive 2.5 GB EagerTensor to stay in System RAM (CPU)
        # This prevents TensorFlow from attempting to copy the entire history to the GPU at once.
        with tf.device('/CPU:0'):
            dataset = tf.data.Dataset.from_tensor_slices((
                np.array(replay_states, dtype=np.float32), 
                {
                    'value': np.array(replay_values, dtype=np.float32), 
                    'score_lead': np.array(replay_scores, dtype=np.float32)
                }
            ))
        
        dataset = dataset.shuffle(buffer_size=min(50000, current_buffer_size), reshuffle_each_iteration=True).batch(64).prefetch(tf.data.AUTOTUNE)

        print(f"🧠 Training Q-Value Neural Network...", flush=True)
        model.fit(dataset, epochs=4, verbose=1)
        
        print(f"💾 Saving generation {iteration} model and replay buffer...", flush=True)
        current_model_name = f"blokus_expert_v{iteration}.keras"
        model.save(current_model_name)
        model.save("blokus_expert_latest.keras") 

        # 🚀 FIX 4: SAVE THE HISTORICAL BUFFER & NEXT ITERATION COUNT TO DISK
        with open(buffer_path, "wb") as f:
            pickle.dump((replay_states, replay_values, replay_scores, iteration + 1), f, protocol=pickle.HIGHEST_PROTOCOL)

        for old_file in glob.glob("blokus_expert_v*.keras"):
            if old_file != current_model_name:
                try: os.remove(old_file); print(f"🗑️ Auto-deleted old model: {old_file}", flush=True)
                except OSError: pass

if __name__ == "__main__":
    mp.freeze_support()
    run_training_pipeline()