from textwrap import wrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
AGGREGATIONS = ["mean", "max", "min"]
COLORS = {"green":"#83E377".lower(),"blue":"#2C699A".lower() , "purple":"#54478C".lower() , "grey":"#737370".lower(),
          "turquoise": "#0DB39E".lower()}
def distribution_plot(lang: str, distribution_df: pd.DataFrame, col_x: str, col_y: str, label_texts: list, title: str):
    """
    Dumps vertical bar plot for distributions.
    @Todo add horizontal legend
    """
    labels = distribution_df.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 18)) for l in labels]
    fig, ax = plt.subplots()
    p1 = ax.bar(distribution_df.reset_index()[col_x], distribution_df.reset_index()[col_y],
                color=COLORS["turquoise"])
    ax.set_xlabel(label_texts[0], fontsize=8)
    ax.set_ylabel(label_texts[1], fontsize=8)
    ax.set_xticks(np.arange(len(distribution_df.reset_index()[col_x].values)))
    ax.set_xticklabels(labels, fontsize=6, rotation=90)
    plt.title(title)
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(f'{lang}/plots/{col_x}_distribution.png', bbox_inches="tight")
    plt.show()


def lower_court_effect_plot(df_1, df_2, col_y: str, col_x: str, label_texts: list, legend_texts: list,
                            title: str, filepath:str):
    """
    Dumps horizontal bar plot fo opposing value sets.
    @Todo add horizontal legend
    """
    labels = df_1.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 18)) for l in labels]
    fig, ax = plt.subplots()
    ax.barh(df_1.reset_index()[col_x], df_1.reset_index()[col_y],
                 label=legend_texts[0],
                 color=COLORS["green"])
    ax.barh(df_2.reset_index()[col_x], df_2.reset_index()[col_y],
                 label=legend_texts[1],
                 color=COLORS["purple"])
    ax.set_xlabel(label_texts[0], fontsize=8)
    ax.set_ylabel(label_texts[1], fontsize=8)
    ax.set_yticks(np.arange(len(df_1.reset_index()["lower_court"].values)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    plt.title(title, fontsize=12)
    ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()


def mean_plot(mean_df: pd.DataFrame, filepath: str, title:str, mean_lines:dict):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    colors = [ COLORS["purple"],COLORS["green"], COLORS["blue"]]
    mean_df.plot(kind='bar',color=colors)
    plt.rcParams.update({'font.size': 8})
    plt.title(title, fontsize=12)
    labels =list(mean_df.columns)
    i = 0
    plt.legend(labels, ncol=len(labels),loc="upper right")
    for key in mean_lines.keys():
        plt.axhline(y=mean_lines[key],c=colors[i], linestyle='dashed', label="horizontal")
        labels.append(key)
        i += 1
        plt.legend(labels, loc="upper left")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()

def IAA_Agreement_plots():
    """
    @Todo
    """
    pass