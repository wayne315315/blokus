import os
import time
import glob
import multiprocessing as mp
import multiprocessing.connection
import ctypes
import numpy as np
import tensorflow as tf
import concurrent.futures

from helper import BOARD_SIZE, SHAPES

# 🛑 Massive 16384 capacity securely fits parallel MCTS unexpanded leaves perfectly
MAX_CAPACITY = 16384
THREADS_PER_WORKER = 4

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

# 🛑 A dedicated isolated thread function. Drops the GIL while waiting on I/O pipes.
def single_thread_task(games_per_thread, conn, thread_id, shared_data_bases, total_threads, shared_counter):
    from tf_alphazero_bot import ExpertBlokusBot
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))
    
    bot = ExpertBlokusBot(pipe=conn, shared_data=(thread_id, shared_states, shared_values, shared_scores), is_training=True)
    
    S, V, SC = [], [], []
    for _ in range(games_per_thread):
        s, v, sc = generate_expert_game(bot)
        S.extend(s); V.extend(v); SC.extend(sc)
        with shared_counter.get_lock(): shared_counter.value += 1
            
    conn.send("DONE") 
    return S, V, SC

def distributed_train_worker(games_per_thread, worker_conns, result_queue, worker_thread_ids, shared_counter, shared_data_bases, total_threads):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    # Bypasses the GIL perfectly by threading I/O bound pipe calls
    S, V, SC = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS_PER_WORKER) as executor:
        futures = [
            executor.submit(single_thread_task, games_per_thread, conn, tid, shared_data_bases, total_threads, shared_counter)
            for conn, tid in zip(worker_conns, worker_thread_ids)
        ]
        for f in concurrent.futures.as_completed(futures):
            s, v, sc = f.result()
            S.extend(s); V.extend(v); SC.extend(sc)
            
    result_queue.put((S, V, SC))

def training_inference_server(conns, fast_infer, shared_counter, total_games, shared_data_bases, total_threads):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((total_threads, MAX_CAPACITY, 20, 20, 6))
    shared_values = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((total_threads, MAX_CAPACITY))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((total_threads, MAX_CAPACITY))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(total_threads)}
    
    ready_indices, batch_sizes, ready_pipes = [], [], []
    total_states_queued = 0
    last_print_time = time.time()

    CHUNK_SIZE = 2048 

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
            print("start concat....", flush=True)
            t1 = time.time()
            flat_states = [shared_states[w_id, :size] for w_id, size in zip(ready_indices, batch_sizes)]
            batch_tensor = np.concatenate(flat_states, axis=0)
            actual_size = batch_tensor.shape[0]
            t2 = time.time()
            print("end concat... size %d took %.2f second" % (actual_size, t2-t1), flush=True)
            
            v_preds, sc_preds = [], []
            for i in range(0, actual_size, CHUNK_SIZE):
                chunk = batch_tensor[i : i + CHUNK_SIZE]
                curr_size = chunk.shape[0]
                
                pad_target = 1 << (curr_size - 1).bit_length()
                if curr_size < pad_target:
                    pad = np.zeros((pad_target - curr_size, 20, 20, 6), dtype=np.float32)
                    chunk_padded = np.concatenate([chunk, pad], axis=0)
                else:
                    chunk_padded = chunk
                    
                preds = fast_infer(chunk_padded)
                v_preds.append(preds[0].numpy().flatten()[:curr_size])
                sc_preds.append(preds[1].numpy().flatten()[:curr_size])
                
            v_numpy = np.concatenate(v_preds)
            sc_numpy = np.concatenate(sc_preds)
            
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size
            
            curr_time = time.time()
            if curr_time - last_print_time > 0.5:
                games_completed = shared_counter.value
                print(f"⚡ GPU BATCH | After-States evaluated: {actual_size:<5} | Games Generated: {games_completed}/{total_games} ", end='\r', flush=True)
                last_print_time = curr_time

            for pipe in ready_pipes: pipe.send(True)

            ready_indices, batch_sizes, ready_pipes = [], [], []
            total_states_queued = 0
            
    print("\n✅ All GPU processes finished gathering data.", flush=True)

def run_training_pipeline(num_iteration=50):
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
    
    print("🔥 Pre-compiling Power-of-2 XLA Buckets (1 to 2048) into System RAM...", flush=True)
    for p in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        _ = fast_infer(tf.zeros((p, 20, 20, 6), dtype=tf.float32))
    print("✅ All dynamic graphs successfully compiled and cached!", flush=True)

    num_workers = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1 
    total_threads = num_workers * THREADS_PER_WORKER
    
    TOTAL_GAMES_PER_ITERATION = total_threads
    games_per_thread = max(1, TOTAL_GAMES_PER_ITERATION // total_threads)
    actual_total_games = total_threads * games_per_thread

    for iteration in range(1, num_iteration + 1):
        print(f"\n" + "="*70, flush=True)
        print(f"🚀 STARTING Q-LEARNING ITERATION {iteration} | Generating {actual_total_games} Games", flush=True)
        print("="*70, flush=True)
        
        start_time = time.time()
        ctx = mp.get_context('spawn')
        
        shared_counter = ctx.Value('i', 0)
        shared_states_base = mp.Array(ctypes.c_float, total_threads * MAX_CAPACITY * 20 * 20 * 6)
        shared_values_base = mp.Array(ctypes.c_float, total_threads * MAX_CAPACITY)
        shared_scores_base = mp.Array(ctypes.c_float, total_threads * MAX_CAPACITY)
        shared_data_bases = (shared_states_base, shared_values_base, shared_scores_base)

        pipes = [ctx.Pipe() for _ in range(total_threads)]
        parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
        
        result_queue = ctx.Queue()
        processes = []
        
        for i in range(num_workers):
            # Pass a subset of pipes to each worker process corresponding to its threads
            worker_conns = child_conns[i*THREADS_PER_WORKER : (i+1)*THREADS_PER_WORKER]
            worker_thread_ids = list(range(i*THREADS_PER_WORKER, (i+1)*THREADS_PER_WORKER))
            
            p = ctx.Process(target=distributed_train_worker, args=(games_per_thread, worker_conns, result_queue, worker_thread_ids, shared_counter, shared_data_bases, total_threads))
            p.start()
            processes.append(p)

        training_inference_server(parent_conns, fast_infer, shared_counter, actual_total_games, shared_data_bases, total_threads)

        S, V, SC = [], [], []
        for _ in range(num_workers):
            s, v, sc = result_queue.get()
            S.extend(s); V.extend(v); SC.extend(sc)

        for p in processes: p.join()
        
        generation_time = time.time() - start_time
        print(f"✅ Data Generation Complete in {generation_time:.1f}s. Extracted {len(S)} Deep States.", flush=True)
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