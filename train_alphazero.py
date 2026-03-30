import os
import time
import glob
import multiprocessing as mp
import multiprocessing.connection
import ctypes
import numpy as np

from helper import BOARD_SIZE, SHAPES

# 🛑 IMPORT ISOLATION: No global TensorFlow import here to save ~15GB of worker RAM!
MAX_ORDER = 10
_NUM_CPUS = mp.cpu_count()
# 🛑 Pure Python Parallelism: 1 Process = 1 Game (No GIL Blocking!)
NUM_WORKERS = min(31, _NUM_CPUS - 1 if _NUM_CPUS > 1 else 1)
MAX_CAPACITY = 2048

def generate_expert_game(bot):
    states, players = [], []
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
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

    return states, val_targets, score_targets

def distributed_train_worker(games_per_worker, conn, result_queue, worker_idx, shared_counter, shared_data_bases, num_workers):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    from tf_alphazero_bot import ExpertBlokusBot
    
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(worker_idx, shared_states, shared_values, shared_scores), is_training=True)
    
    S, V, SC = [], [], []
    for _ in range(games_per_worker):
        s, v, sc = generate_expert_game(bot)
        S.extend(s); V.extend(v); SC.extend(sc)
        with shared_counter.get_lock(): shared_counter.value += 1
            
    conn.send("DONE") 
    result_queue.put((S, V, SC))

def training_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, num_workers):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(num_workers)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()

    CHUNK_SIZE =  2 ** MAX_ORDER

    print("🔥 Allocating Static Server Buffer in System RAM...", flush=True)
    MAX_POSSIBLE_BATCH = num_workers * MAX_CAPACITY
    SERVER_BUFFER = np.empty((MAX_POSSIBLE_BATCH, 20, 20, 6), dtype=np.float32)
    print("🔥 Pre-Warming physical RAM to bypass OS allocation lag...", flush=True)
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
            
            # 🚀 POWER-OF-2 DECOMPOSITION ALGORITHM
            sizes = []
            rem = actual_size
            while rem > 0:
                if rem >= CHUNK_SIZE:
                    p = CHUNK_SIZE
                else:
                    p = 1 << (rem.bit_length() - 1)
                sizes.append(p)
                rem -= p
                
            tensor_cursor = 0
            for p in sizes:
                chunk = batch_tensor[tensor_cursor : tensor_cursor + p]
                preds = fast_infer(chunk)
                v_preds.append(preds[0].numpy().flatten())
                sc_preds.append(preds[1].numpy().flatten())
                tensor_cursor += p
                
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
                      f"Infer: {t_infer:>5.1f}ms ({t_per_sample:>4.2f}ms/st) | "
                      f"Write: {t_write:>5.1f}ms | "
                      f"Games: {games_completed}/{total_games}   ", end='\r', flush=True)
                last_print_time = curr_time

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All GPU processes finished gathering data.", flush=True)

def run_training_pipeline(num_iteration=50):
    # 🛑 IMPORT ISOLATION: Load TF only in main process to save 15 GB RAM
    import tensorflow as tf
    from tf_alphazero_bot import AdvancedBlokusModel

    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    model_path = "blokus_expert_latest.keras"
    if os.path.exists(model_path):
        print(f"🚀 Found existing model! Resuming training from {model_path}...", flush=True)
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss={'value': 'mean_squared_error', 'score_lead': 'huber'}, loss_weights={'value': 1.0, 'score_lead': 0.05}, jit_compile=False)
    else:
        print("🚀 No existing model found. Building a new AdvancedBlokusModel from scratch...", flush=True)
        adv_model = AdvancedBlokusModel()
        model = adv_model.model
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss={'value': 'mean_squared_error', 'score_lead': 'huber'}, loss_weights={'value': 1.0, 'score_lead': 0.05}, jit_compile=False)

    @tf.function(reduce_retracing=True)
    def fast_infer(batch_tensor): 
        return model(batch_tensor, training=False)

    print(f"🔥 Pre-compiling Power-of-2 XLA Buckets (1 to {MAX_CAPACITY}) into System RAM...")
    for o in range(MAX_ORDER + 1):
        _ = fast_infer(tf.zeros((2 ** o, 20, 20, 6), dtype=tf.float32))
    print("✅ All dynamic graphs successfully compiled and cached!", flush=True)
    
    TOTAL_GAMES_PER_ITERATION = 1000 
    games_per_worker = max(1, TOTAL_GAMES_PER_ITERATION // NUM_WORKERS)
    actual_total_games = NUM_WORKERS * games_per_worker

    for iteration in range(1, num_iteration + 1):
        print(f"\n" + "="*70, flush=True)
        print(f"🚀 STARTING Q-LEARNING ITERATION {iteration} | Generating {actual_total_games} Games", flush=True)
        print("="*70, flush=True)
        
        start_time = time.time()
        ctx = mp.get_context('spawn')
        
        shared_counter = ctx.Value('i', 0)
        shared_states_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY * 20 * 20 * 6)
        shared_values_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY)
        shared_scores_base = mp.Array(ctypes.c_float, NUM_WORKERS * MAX_CAPACITY)
        shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

        pipes = [ctx.Pipe() for _ in range(NUM_WORKERS)]
        parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
        
        result_queue = ctx.Queue()
        processes = []
        
        for i in range(NUM_WORKERS):
            p = ctx.Process(target=distributed_train_worker, args=(games_per_worker, child_conns[i], result_queue, i, shared_counter, shared_data_bases, NUM_WORKERS))
            p.start()
            processes.append(p)

        training_inference_server(parent_conns, fast_infer, shared_counter, actual_total_games, shared_data_bases, NUM_WORKERS)

        S, V, SC = [], [], []
        for _ in range(NUM_WORKERS):
            s, v, sc = result_queue.get()
            S.extend(s); V.extend(v); SC.extend(sc)

        for p in processes: p.join()
        
        generation_time = time.time() - start_time
        print(f"\n✅ Data Generation Complete in {generation_time:.1f}s. Extracted {len(S)} Deep States.", flush=True)
        print(f"🧠 Constructing tf.data.Dataset Pipeline...", flush=True)

        dataset = tf.data.Dataset.from_tensor_slices((np.array(S, dtype=np.float32), {'value': np.array(V, dtype=np.float32), 'score_lead': np.array(SC, dtype=np.float32)}))
        dataset = dataset.shuffle(buffer_size=min(15000, len(S)), reshuffle_each_iteration=True).batch(64).prefetch(tf.data.AUTOTUNE)

        print(f"🧠 Training Q-Value Neural Network...", flush=True)
        model.fit(dataset, epochs=8, verbose=1)
        
        print(f"💾 Saving generation {iteration}...", flush=True)
        current_model_name = f"blokus_expert_v{iteration}.keras"
        model.save(current_model_name)
        model.save("blokus_expert_latest.keras") 

        for old_file in glob.glob("blokus_expert_v*.keras"):
            if old_file != current_model_name:
                try: os.remove(old_file); print(f"🗑️ Auto-deleted old model: {old_file}", flush=True)
                except OSError: pass

if __name__ == "__main__":
    mp.freeze_support()
    run_training_pipeline()