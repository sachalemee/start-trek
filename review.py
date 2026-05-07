#!.venv/bin/python
## EPITECH PROJECT, 2026
## StartTrek
## File description:
## review
##

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def review(path: str):
    try:
        os.mkdir(path + "/images")
    except FileExistsError:
        pass
    df = pd.read_csv(path + "/csv/episodes.csv")
    fit = np.polyfit(df['episode'], df['total_reward'], 10)
    pf = np.poly1d(fit)
    plt.plot(df['episode'], df['total_reward'], marker='o', linestyle=' ', color='blue')
    plt.plot(df['episode'], df['total_reward'], 'b', df['episode'], pf(df['episode']), 'r')
    plt.xlabel("Episode Number")
    plt.ylabel("Reward")
    plt.title("Total Reward for each episode")
    ax = plt.gca()
    ax.set_ylim([-600, 600])
    plt.grid()
    plt.savefig(path + "/images/episodes.png")


review(sys.argv[1])