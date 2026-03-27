import os
import time
import multiprocessing as mp
import multiprocessing.connection
import ctypes
import numpy as np
import tensorflow as tf

from tf_alphazero_bot import AdvancedBlokusModel, ExpertBlokusBot
from helper import BOARD_SIZE, SHAPES

VIRTUAL_THREADS = 256 # KataGo thread multiplier

def generate_expert_game(bot):
    states, policies, players = [], [], []
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))
    inventories = {i: list(SHAPES.keys()) for i in range(1, 5)} 
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
        is_fast_playout = np.random.rand() < 0.80 
        action, pi = bot.get_action(state_tensor, legal_moves, is_training=True, fast_playout=is_fast_playout)
        
        if not is_fast_playout:
            states.append(state_tensor)
            policies.append(pi) 
            players.append(current_player)
            
        shape_name, coords = bot._decode_action(action, legal_moves)
        for r, c in coords: board[r][c] = current_player
        if shape_name in inventories[current_player]: inventories[current_player].remove(shape_name)
        first_moves[current_player] = False
        current_player = (current_player % 4) + 1
        
    def get_score(pid): return sum(int(s.split('_')[0]) for s in inventories.get(pid, []))
    team1_score, team2_score = get_score(1) + get_score(3), get_score(2) + get_score(4)
    ownership_matrix = np.copy(board) 
    
    val_targets, score_targets, own_targets = [], [], []
    for p in players:
        if p in [1, 3]:
            val_targets.append(1 if team1_score < team2_score else -1)
            score_targets.append(team2_score - team1_score) 
        else:
            val_targets.append(1 if team2_score < team1_score else -1)
            score_targets.append(team1_score - team2_score)
        own_targets.append(ownership_matrix)

    return states, policies, val_targets, score_targets, own_targets

def distributed_train_worker(num_games, conn, result_queue, worker_idx, shared_counter, shared_data_bases, num_workers):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
    
    # 🚀 Map multi-dimensional shared memory to accept VIRTUAL_THREADS array batches
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, VIRTUAL_THREADS, 20, 20, 6))
    shared_policies = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, VIRTUAL_THREADS, 67200))
    shared_values = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, VIRTUAL_THREADS))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[3].get_obj()).reshape((num_workers, VIRTUAL_THREADS))
    
    shared_data = (worker_idx, shared_states, shared_policies, shared_values, shared_scores)
    bot = ExpertBlokusBot(pipe=conn, shared_data=shared_data, is_training=True, virtual_threads=VIRTUAL_THREADS)
    
    S, P, V, SC, O = [], [], [], [], []
    for _ in range(num_games):
        s, p, v, sc, o = generate_expert_game(bot)
        S.extend(s); P.extend(p); V.extend(v); SC.extend(sc); O.extend(o)
        with shared_counter.get_lock(): shared_counter.value += 1
            
    conn.send("DONE") 
    result_queue.put((S, P, V, SC, O))

def training_inference_server(conns, model, shared_counter, total_games, shared_data_bases, num_workers):
    shared_states = np.ctypeslib.as_array(shared_data_bases[0].get_obj()).reshape((num_workers, VIRTUAL_THREADS, 20, 20, 6))
    shared_policies = np.ctypeslib.as_array(shared_data_bases[1].get_obj()).reshape((num_workers, VIRTUAL_THREADS, 67200))
    shared_values = np.ctypeslib.as_array(shared_data_bases[2].get_obj()).reshape((num_workers, VIRTUAL_THREADS))
    shared_scores = np.ctypeslib.as_array(shared_data_bases[3].get_obj()).reshape((num_workers, VIRTUAL_THREADS))

    active_conns = list(conns)
    conn_to_id = {conns[i]: i for i in range(num_workers)}
    
    @tf.function(reduce_retracing=True)
    def fast_infer(x): return model(x, training=False)
    _ = fast_infer(np.zeros((1, 20, 20, 6), dtype=np.float32))
    
    ready_indices = []
    batch_sizes = []
    ready_pipes = []
    total_states_queued = 0
    last_fire_time = time.time()

    MIN_BATCH_SIZE = 1024 
    MAX_WAIT_TIME = 0.01

    while active_conns:
        readable = multiprocessing.connection.wait(active_conns, timeout=0.002)
        
        for p in readable:
            try:
                msg = p.recv()
                if msg == "DONE":
                    active_conns.remove(p)
                else:
                    ready_indices.append(conn_to_id[p])
                    batch_sizes.append(msg) # Extract jagged batch sizes generated by Virtual Loss
                    ready_pipes.append(p)
                    total_states_queued += msg
            except EOFError:
                if p in active_conns: active_conns.remove(p)

        time_since_fire = time.time() - last_fire_time

        # 🚀 GPU triggers instantly when a massive matrix is ready
        if total_states_queued >= MIN_BATCH_SIZE or (total_states_queued > 0 and time_since_fire > MAX_WAIT_TIME):
            
            # Extract and flatten the jagged state tensors
            flat_states = []
            for w_id, size in zip(ready_indices, batch_sizes):
                flat_states.append(shared_states[w_id, :size])
            
            batch_tensor = np.concatenate(flat_states, axis=0)
            print("batch tensor shape:", batch_tensor.shape)  # Debugging line
            p_out, v_out, sc_out, _ = fast_infer(batch_tensor)
            
            p_numpy = p_out.numpy()
            v_numpy = v_out.numpy().flatten()
            sc_numpy = sc_out.numpy().flatten()
            
            # Repack the flattened predictions back into jagged RAM slots
            cursor = 0
            for w_id, size in zip(ready_indices, batch_sizes):
                shared_policies[w_id, :size] = p_numpy[cursor:cursor+size]
                shared_values[w_id, :size] = v_numpy[cursor:cursor+size]
                shared_scores[w_id, :size] = sc_numpy[cursor:cursor+size]
                cursor += size
            
            print(f"⚡ GPU MCTS BATCH | Virtual Size: {total_states_queued:<4} | Games: {shared_counter.value}/{total_games} ", end='\r', flush=True)

            for pipe in ready_pipes: pipe.send(True)

            ready_indices = []
            batch_sizes = []
            ready_pipes = []
            total_states_queued = 0
            last_fire_time = time.time()
            
    print("\n✅ All GPU processes finished gathering data.")

def run_training_pipeline():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus: tf.config.experimental.set_memory_growth(gpus[0], True)

    adv_model = AdvancedBlokusModel()
    
    TOTAL_GAMES_PER_ITERATION = 128 
    num_workers = 31 # Perfectly maps your 32-thread CPU
    games_per_worker = max(1, TOTAL_GAMES_PER_ITERATION // num_workers)
    actual_total_games = num_workers * games_per_worker

    for iteration in range(50):
        print(f"\n" + "="*70)
        print(f"🚀 STARTING ITERATION {iteration} | Generating {actual_total_games} Games")
        print("="*70)
        
        start_time = time.time()
        ctx = mp.get_context('spawn')
        
        # 🚀 Expanded RAM to accommodate 16 Virtual Threads per worker
        shared_states_base = mp.Array(ctypes.c_float, num_workers * VIRTUAL_THREADS * 20 * 20 * 6)
        shared_policies_base = mp.Array(ctypes.c_float, num_workers * VIRTUAL_THREADS * 67200)
        shared_values_base = mp.Array(ctypes.c_float, num_workers * VIRTUAL_THREADS)
        shared_scores_base = mp.Array(ctypes.c_float, num_workers * VIRTUAL_THREADS)
        shared_data_bases = (shared_states_base, shared_policies_base, shared_values_base, shared_scores_base)

        pipes = [ctx.Pipe() for _ in range(num_workers)]
        parent_conns, child_conns = [p[0] for p in pipes], [p[1] for p in pipes]
        
        result_queue = ctx.Queue()
        games_completed = ctx.Value('i', 0) 
        processes = []
        
        for i in range(num_workers):
            p = ctx.Process(target=distributed_train_worker, args=(games_per_worker, child_conns[i], result_queue, i, games_completed, shared_data_bases, num_workers))
            p.start()
            processes.append(p)

        training_inference_server(parent_conns, adv_model.model, games_completed, actual_total_games, shared_data_bases, num_workers)

        S, P, V, SC, O = [], [], [], [], []
        for _ in range(num_workers):
            s, p, v, sc, o = result_queue.get()
            S.extend(s); P.extend(p); V.extend(v); SC.extend(sc); O.extend(o)

        for p in processes: p.join()
        
        generation_time = time.time() - start_time
        print(f"✅ Data Generation Complete in {generation_time:.1f}s. Extracted {len(S)} Deep States.")
        print(f"🧠 Training Neural Network...")

        adv_model.model.fit(
            np.array(S), 
            {'policy': np.array(P), 'value': np.array(V), 'score_lead': np.array(SC), 'ownership': np.array(O)},
            batch_size=256,
            epochs=2
        )
        
        adv_model.model.save("blokus_expert_v0.keras") 
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    mp.freeze_support()
    run_training_pipeline()
