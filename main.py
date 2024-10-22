import numpy as np
import matplotlib.pyplot as plt
import all_fun
from bka import BKA
from argparse import ArgumentParser
import os
import json
import multiprocessing


def run_bka_once(args):
    
    search_agents, iteration, lb, ub, dim, fobj = args
    BKA_score, Best_Pos_BKA, BKA_Convergence_curve = BKA(search_agents, iteration, [lb], [ub], dim, fobj)
    return BKA_score, Best_Pos_BKA, BKA_Convergence_curve


def train(dataset, fuction, iteration, repeat_times, search_agents=30):

    fobj, lb, ub, dim = getattr(all_fun, dataset)(fuction)
    bka_list = []
    cur_list = []
    pose_list = []
    params = [(search_agents, iteration, lb, ub, dim, fobj) for _ in range(repeat_times)]
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_bka_once, params)

    for BKA_score, Best_Pos_BKA, BKA_Convergence_curve in results:
        bka_list.append(BKA_score)
        pose_list.append(Best_Pos_BKA)
        cur_list.append(BKA_Convergence_curve)

    min_index = min(enumerate(bka_list), key=lambda x: x[1])[0]
    print("objective function value: ", bka_list[min_index])

    return bka_list[min_index], pose_list[min_index], cur_list[min_index]


def save_result(Best_score, Best_Pos, Convergence_curve, dataset, fuction, iteration):
    
    OUTPUT_PATH = f"output/{dataset}/{fuction}/"
    CNT = 100
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    data = {'Best_score': float(Best_score), 'Best_Pos': Best_Pos.tolist()}
    with open(f'{OUTPUT_PATH}/result.json', 'w') as f:
        json.dump(data, f)

    k = np.round(np.linspace(1, iteration-1, CNT)).astype(int)  # Randomly selected points
    iter_ = np.arange(1, iteration + 1)
    plt.subplot(1, 1, 1)
    plt.plot(iter_[k], np.array(Convergence_curve)[k], 'r->', linewidth=1)
    plt.grid(True)
    plt.title(fuction + ' convergence curve')
    plt.xlabel('iterations')
    plt.ylabel('fitness value')
    plt.legend(['BKA'])
    plt.savefig(OUTPUT_PATH + 'convergence_curve.png')
    print(f"Result stored in {OUTPUT_PATH}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Main script parameters")
    parser.add_argument("--dataset", default="CEC2005", type=str)
    parser.add_argument("--fuction", default="F1", type=str)
    parser.add_argument("--iteration", default=2000, type=int)
    parser.add_argument("--repeat_times", default=1, type=int)
    parser.add_argument("--search_agents", default=30, type=int)
    args = parser.parse_args()
    print(f"Start training: {args.dataset} {args.fuction} for {args.repeat_times} times each {args.iteration} iterations")

    Best_score, Best_Pos, Convergence_curve = train(args.dataset, args.fuction, args.iteration, args.repeat_times, args.search_agents)
    save_result(Best_score, Best_Pos, Convergence_curve, args.dataset, args.fuction, args.iteration)
