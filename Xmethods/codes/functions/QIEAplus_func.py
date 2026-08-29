import math
import random

import numpy as np

from functions.QIEA_func import (
    H_gate_max,
    H_gate_min,
    Individual,
    observe_random_order,
    observe_random_order_hops,
    observe_random_order_timeXtime,
    search_ex_graph,
)


def Quantum_individual_optimizationA(
    agent_iteration,
    model,
    link_data,
    can_events,
    H,
    teta,
    factual,
    existed,
    max_events,
    initial_output,
    flag,
    hops,
    theta_s_min=0.001,
    theta_s_max=0.01,
    theta_l_min=0.1,
    theta_l_max=0.5,
    normalization_factor=1.0,
    exchange_interval=10,
    max_exchange_extension=50,
    large_stagnation_patience=80,
    stop_stagnation_patience=150,
    tau_low=0.1,
    tau_high=0.9,
):
    """QIEA-TGX+ search built on the public QIEA-TGX implementation.

    Only the QIEA-TGX+ mechanisms are implemented here. The surrounding
    candidate generation, model loading, target-event handling, observation
    helpers, H-gate, and fitness evaluation are shared with QIEA-TGX.

    Rotation-angle arguments are coefficients of pi. Their defaults and the
    other QIEA-TGX+ defaults match the experimental setting in the paper.
    """
    model_time = []

    if not (0 < theta_s_min <= theta_s_max):
        raise ValueError('Small-scale rotation bounds must satisfy 0 < min <= max.')
    if not (0 < theta_l_min <= theta_l_max):
        raise ValueError('Large-scale rotation bounds must satisfy 0 < min <= max.')
    if normalization_factor <= 0:
        raise ValueError('normalization_factor must be positive.')
    if exchange_interval < 1:
        raise ValueError('exchange_interval must be at least 1.')
    if max_exchange_extension < 0:
        raise ValueError('max_exchange_extension must be non-negative.')
    if large_stagnation_patience < 1:
        raise ValueError('large_stagnation_patience must be at least 1.')
    if stop_stagnation_patience < large_stagnation_patience:
        raise ValueError(
            'stop_stagnation_patience must be greater than or equal to '
            'large_stagnation_patience.'
        )
    if not (0 <= tau_low < tau_high <= 1):
        raise ValueError('Post-optimization bounds must satisfy 0 <= low < high <= 1.')

    # Keep the original QIEA-TGX hop preprocessing unchanged.
    if 'ob' in flag:
        one_hop = hops[:, 0]
        two_hop = hops[:, 1:]

        _, unique_indices = np.unique(one_hop, return_index=True)
        unique_indices = np.sort(unique_indices)
        one_hop = one_hop[unique_indices]
        two_hop = two_hop[unique_indices]

        hop1_index = np.zeros(len(one_hop), dtype=int)
        for i in range(len(one_hop)):
            indices = np.where(can_events == one_hop[i])[0]
            hop1_index[i] = int(indices[0])

        hop2_index = np.zeros(len(can_events) - len(one_hop), dtype=int)
        two_hop_unique = np.setdiff1d(can_events, one_hop)
        assert len(two_hop_unique) == len(can_events) - len(one_hop)
        for i in range(len(two_hop_unique)):
            indices = np.where(can_events == two_hop_unique[i])[0]
            hop2_index[i] = int(indices[0])

        hop2_array = []
        for i in hop2_index:
            hop_links = [i]
            candidate = can_events[i]
            for j in range(len(two_hop)):
                if candidate in two_hop[j]:
                    hop_links.append(hop1_index[j])
            hop2_array.append(hop_links)

    genome_size = len(can_events)
    raw_1hop_ids = np.unique(hops[:, 0])
    up_down = int(factual == existed)
    agents = int(agent_iteration[0])
    iteration = int(agent_iteration[1])

    # QIEA-TGX+ uses paired Large/Small agents.
    if agents < 2 or agents % 2 != 0:
        raise ValueError('QIEA-TGX+ requires an even number of agents of at least 2.')

    # Shared QIEA-TGX initialization, with separate Large/Small global bests.
    Q = np.full((agents, genome_size), np.pi / 4)
    p = np.zeros((agents, genome_size), dtype=int)
    if up_down:
        B = np.zeros((agents, genome_size), dtype=int)
        best_agent_fitness = np.full(agents, -math.inf)
        best_global_fitness_L = -math.inf
        best_global_fitness_S = -math.inf
    else:
        B = np.ones((agents, genome_size), dtype=int)
        best_agent_fitness = np.full(agents, math.inf)
        best_global_fitness_L = math.inf
        best_global_fitness_S = math.inf

    fitness = np.zeros(agents)
    best_global_genome_L = np.zeros(genome_size, dtype=int)
    best_global_genome_S = np.zeros(genome_size, dtype=int)
    result_list = []
    loop_list = []

    # Initial observation/evaluation follows QIEA-TGX.
    for i in range(agents):
        mode = 'Large' if i % 2 == 0 else 'Small'

        if flag == 'ob_hops':
            p[i] = observe_random_order_hops(
                Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
            )
        else:
            p[i] = observe_random_order(Q[i], factual, max_events)

        if factual:
            assert sum(p[i]) <= max_events
        else:
            assert sum(p[i]) >= genome_size - max_events

        individual = Individual(p[i], model, link_data, can_events)
        model_time.append(individual.runtime)
        fitness[i] = individual.get_fitness()

        if up_down:
            if fitness[i] > best_agent_fitness[i]:
                best_agent_fitness[i] = fitness[i]
                B[i] = p[i].copy()
            if mode == 'Large' and fitness[i] > best_global_fitness_L:
                best_global_fitness_L = fitness[i]
                best_global_genome_L = p[i].copy()
            elif mode == 'Small' and fitness[i] > best_global_fitness_S:
                best_global_fitness_S = fitness[i]
                best_global_genome_S = p[i].copy()
        else:
            if fitness[i] < best_agent_fitness[i]:
                best_agent_fitness[i] = fitness[i]
                B[i] = p[i].copy()
            if mode == 'Large' and fitness[i] < best_global_fitness_L:
                best_global_fitness_L = fitness[i]
                best_global_genome_L = p[i].copy()
            elif mode == 'Small' and fitness[i] < best_global_fitness_S:
                best_global_fitness_S = fitness[i]
                best_global_genome_S = p[i].copy()

    if up_down:
        best_of_both = max(best_global_fitness_L, best_global_fitness_S)
        initial_best_genome = (
            best_global_genome_L
            if best_global_fitness_L >= best_global_fitness_S
            else best_global_genome_S
        )
    else:
        best_of_both = min(best_global_fitness_L, best_global_fitness_S)
        initial_best_genome = (
            best_global_genome_L
            if best_global_fitness_L <= best_global_fitness_S
            else best_global_genome_S
        )

    if factual:
        result_list.append([best_of_both, sum(initial_best_genome), initial_output])
    else:
        result_list.append(
            [best_of_both, len(initial_best_genome) - sum(initial_best_genome), initial_output]
        )

    # Two-stage early stopping and information-exchange state.
    stagnation_count = 0
    prev_best_fitness = best_of_both
    large_active = True

    # Increase total_iteration for the main experiments.
    total_iteration = iteration
    iter_count = 0

    iters_since_sync = 0
    next_sync_check = exchange_interval
    prev_sync_fitness = best_of_both

    while iter_count < total_iteration:
        for i in range(agents):
            mode = 'Large' if i % 2 == 0 else 'Small'
            if not large_active and mode == 'Large':
                continue

            # Preserve the observation modes of public QIEA-TGX.
            if flag == 'ob_hops':
                p[i] = observe_random_order_hops(
                    Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                )
            elif flag == 'ob_half':
                if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:
                    p[i] = observe_random_order(Q[i], factual, max_events)
                else:
                    p[i] = observe_random_order_hops(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
            elif flag == 'ob_hop_timeXtime':
                p[i] = observe_random_order_timeXtime(
                    Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                )
            elif flag == 'ob_hop_timeXtime_half':
                if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:
                    p[i] = observe_random_order(Q[i], factual, max_events)
                else:
                    p[i] = observe_random_order_timeXtime(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
            elif flag == 'ob_hopXtime':
                if np.random.choice([0, 1], p=[0.5, 0.5]) == 0:
                    p[i] = observe_random_order_hops(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
                else:
                    p[i] = observe_random_order_timeXtime(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
            elif flag == 'ob_hop_all':
                roll = random.randint(1, 3)
                if roll == 1:
                    p[i] = observe_random_order(Q[i], factual, max_events)
                elif roll == 2:
                    p[i] = observe_random_order_hops(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
                else:
                    p[i] = observe_random_order_timeXtime(
                        Q[i], factual, max_events, hop1_index, hop2_index, hop2_array
                    )
            else:
                p[i] = observe_random_order(Q[i], factual, max_events)

            individual = Individual(p[i], model, link_data, can_events)
            model_time.append(individual.runtime)
            fitness[i] = individual.get_fitness()

            # ARGO: theta = theta_min + |p_best-p_i|/K*(theta_max-theta_min).
            if mode == 'Small':
                theta_min = theta_s_min * np.pi
                theta_max = theta_s_max * np.pi
            else:
                theta_min = theta_l_min * np.pi
                theta_max = theta_l_max * np.pi

            fitness_dif = abs(fitness[i] - best_agent_fitness[i])
            normalized_dif = min(fitness_dif / normalization_factor, 1.0)
            step = theta_min + normalized_dif * (theta_max - theta_min)

            if up_down:
                should_update = fitness[i] < best_agent_fitness[i]
            else:
                should_update = fitness[i] > best_agent_fitness[i]

            if should_update:
                Q[i] += (B[i] - p[i]) * step

            Q[i] = H_gate_min(Q[i], H)
            Q[i] = H_gate_max(Q[i], H)

            x_graph = search_ex_graph(can_events, factual, p[i])
            candiates_str = ','.join(map(str, x_graph))
            num_use_genom = sum(p[i])
            if factual == 0:
                num_use_genom = genome_size - num_use_genom
            result_list.append([fitness[i], num_use_genom, initial_output])
            loop_list.append([fitness[i], num_use_genom, candiates_str])

        # Update personal and scale-specific global bests.
        for i in range(agents):
            mode = 'Large' if i % 2 == 0 else 'Small'
            if not large_active and mode == 'Large':
                continue

            if up_down:
                if fitness[i] > best_agent_fitness[i]:
                    best_agent_fitness[i] = fitness[i]
                    B[i] = p[i].copy()
                if mode == 'Large' and fitness[i] > best_global_fitness_L:
                    best_global_fitness_L = fitness[i]
                    best_global_genome_L = p[i].copy()
                elif mode == 'Small' and fitness[i] > best_global_fitness_S:
                    best_global_fitness_S = fitness[i]
                    best_global_genome_S = p[i].copy()
            else:
                if fitness[i] < best_agent_fitness[i]:
                    best_agent_fitness[i] = fitness[i]
                    B[i] = p[i].copy()
                if mode == 'Large' and fitness[i] < best_global_fitness_L:
                    best_global_fitness_L = fitness[i]
                    best_global_genome_L = p[i].copy()
                elif mode == 'Small' and fitness[i] < best_global_fitness_S:
                    best_global_fitness_S = fitness[i]
                    best_global_genome_S = p[i].copy()

        # Information exchange. A base exchange is attempted every I_ex.
        # If there is no improvement, it may be deferred by up to I_max
        # additional iterations before exchange is forced.
        iters_since_sync += 1
        if iters_since_sync == next_sync_check and large_active:
            if up_down:
                current_sync_best = max(best_global_fitness_L, best_global_fitness_S)
                is_sync_improved = current_sync_best > prev_sync_fitness
            else:
                current_sync_best = min(best_global_fitness_L, best_global_fitness_S)
                is_sync_improved = current_sync_best < prev_sync_fitness

            extension = next_sync_check - exchange_interval
            force_exchange = extension >= max_exchange_extension

            if is_sync_improved or force_exchange:
                if up_down:
                    large_wins = best_global_fitness_L > best_global_fitness_S
                else:
                    large_wins = best_global_fitness_L < best_global_fitness_S

                if large_wins:
                    winner_fitness = best_global_fitness_L
                    winner_genome = best_global_genome_L.copy()
                    best_global_fitness_S = best_global_fitness_L
                    best_global_genome_S = best_global_genome_L.copy()
                    source_indices = [i for i in range(agents) if i % 2 == 0]
                    target_indices = [i for i in range(agents) if i % 2 != 0]
                else:
                    winner_fitness = best_global_fitness_S
                    winner_genome = best_global_genome_S.copy()
                    best_global_fitness_L = best_global_fitness_S
                    best_global_genome_L = best_global_genome_S.copy()
                    source_indices = [i for i in range(agents) if i % 2 != 0]
                    target_indices = [i for i in range(agents) if i % 2 == 0]

                if up_down:
                    source_idx = source_indices[int(np.argmax(best_agent_fitness[source_indices]))]
                else:
                    source_idx = source_indices[int(np.argmin(best_agent_fitness[source_indices]))]

                source_Q = Q[source_idx].copy()
                for i in target_indices:
                    Q[i] = source_Q.copy()
                    best_agent_fitness[i] = winner_fitness
                    B[i] = winner_genome.copy()

                prev_sync_fitness = current_sync_best
                iters_since_sync = 0
                next_sync_check = exchange_interval
            else:
                next_sync_check += exchange_interval

        if large_active:
            if up_down:
                best_of_both = max(best_global_fitness_L, best_global_fitness_S)
            else:
                best_of_both = min(best_global_fitness_L, best_global_fitness_S)
        else:
            best_of_both = best_global_fitness_S

        if up_down:
            is_improved = best_of_both > prev_best_fitness
        else:
            is_improved = best_of_both < prev_best_fitness

        if is_improved:
            prev_best_fitness = best_of_both
            stagnation_count = 0
        else:
            stagnation_count += 1

        if stagnation_count >= large_stagnation_patience and large_active:
            large_active = False

        if stagnation_count >= stop_stagnation_patience:
            break

        iter_count += 1

    # Select the winning scale before post-optimization.
    if up_down:
        winning_mode = 'Large' if best_global_fitness_L > best_global_fitness_S else 'Small'
    else:
        winning_mode = 'Large' if best_global_fitness_L < best_global_fitness_S else 'Small'

    if winning_mode == 'Large':
        best_global_fitness = best_global_fitness_L
        best_global_genome = best_global_genome_L.copy()
        mode_indices = [i for i in range(agents) if i % 2 == 0]
    else:
        best_global_fitness = best_global_fitness_S
        best_global_genome = best_global_genome_S.copy()
        mode_indices = [i for i in range(agents) if i % 2 != 0]

    if up_down:
        best_agent_idx = mode_indices[np.argmax(fitness[mode_indices])]
    else:
        best_agent_idx = mode_indices[np.argmin(fitness[mode_indices])]
    best_Q = Q[best_agent_idx].copy()

    # Post-optimization: only uncertain Q-bits are considered. Deletions are
    # evaluated before additions, with 1-hop candidates prioritized.
    current_refined_genome = best_global_genome.copy()
    current_refined_fitness = best_global_fitness

    if factual:
        current_usage = int(np.sum(current_refined_genome))
    else:
        current_usage = int(len(current_refined_genome) - np.sum(current_refined_genome))

    final_probs = np.sin(best_Q) ** 2
    uncertain_indices = np.where(
        (final_probs > tau_low) & (final_probs < tau_high)
    )[0]

    deletion_indices = []
    addition_indices = []
    for idx in uncertain_indices:
        target_bit = 1 - current_refined_genome[idx]
        if factual:
            usage_change = 1 if target_bit == 1 else -1
        else:
            usage_change = 1 if target_bit == 0 else -1

        if usage_change == -1:
            deletion_indices.append(idx)
        else:
            addition_indices.append(idx)

    is_hop1 = np.isin(can_events, raw_1hop_ids)
    deletion_indices.sort(key=lambda idx: is_hop1[idx], reverse=True)
    addition_indices.sort(key=lambda idx: is_hop1[idx], reverse=True)

    for idx in deletion_indices:
        target_bit = 1 - current_refined_genome[idx]
        temp_genome = current_refined_genome.copy()
        temp_genome[idx] = target_bit
        individual = Individual(temp_genome, model, link_data, can_events)
        model_time.append(individual.runtime)

        gain = individual.get_fitness() - current_refined_fitness
        if up_down == 0:
            gain = -gain

        if gain > 0:
            current_refined_genome[idx] = target_bit
            if up_down:
                current_refined_fitness += gain
            else:
                current_refined_fitness -= gain
            current_usage -= 1

    for idx in addition_indices:
        if current_usage >= max_events:
            continue

        target_bit = 1 - current_refined_genome[idx]
        temp_genome = current_refined_genome.copy()
        temp_genome[idx] = target_bit
        individual = Individual(temp_genome, model, link_data, can_events)
        model_time.append(individual.runtime)

        gain = individual.get_fitness() - current_refined_fitness
        if up_down == 0:
            gain = -gain

        if gain > 0:
            current_refined_genome[idx] = target_bit
            if up_down:
                current_refined_fitness += gain
            else:
                current_refined_fitness -= gain
            current_usage += 1

    best_global_genome = current_refined_genome.copy()
    best_global_fitness = current_refined_fitness

    if factual:
        final_usage_count = int(np.sum(best_global_genome))
    else:
        final_usage_count = int(len(best_global_genome) - np.sum(best_global_genome))

    if final_usage_count > max_events:
        raise ValueError(
            f'The number of selected events ({final_usage_count}) exceeds '
            f'max_events ({max_events}).'
        )

    flag_res = 0
    if factual == 0:
        if existed == 0:
            if best_global_fitness > 0.5:
                flag_res = 1
        else:
            if best_global_fitness < 0.5:
                flag_res = 1

    if factual:
        num_use_genom = sum(best_global_genome)
    else:
        num_use_genom = len(best_global_genome) - sum(best_global_genome)

    result = [flag_res, best_global_fitness, num_use_genom]
    return result, result_list, loop_list, model_time
