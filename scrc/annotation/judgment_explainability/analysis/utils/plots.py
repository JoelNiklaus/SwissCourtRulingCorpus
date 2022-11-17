from textwrap import wrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def distribution_plot(lang: str, distribution_df: pd.DataFrame, col_x: str, col_y: str, label_texts: list, title: str):
    """
    Dumps vertical bar plot for distributions.
    """
    labels = distribution_df.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 18)) for l in labels]
    fig, ax = plt.subplots()
    p1 = ax.bar(distribution_df.reset_index()[col_x], distribution_df.reset_index()[col_y],
                color="#D2D2CF".lower())
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
    """
    labels = df_1.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 18)) for l in labels]
    fig, ax = plt.subplots()
    p1 = ax.barh(df_1.reset_index()[col_x], df_1.reset_index()[col_y],
                 label=legend_texts[0],
                 color="#C9E4DE".lower())
    p2 = ax.barh(df_2.reset_index()[col_x], df_2.reset_index()[col_y],
                 label=legend_texts[1],
                 color="#DBCDF0".lower())
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
