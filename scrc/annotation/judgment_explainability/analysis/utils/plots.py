from textwrap import wrap

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
LANGUAGES = ["de", "fr", "it"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
AGGREGATIONS = ["mean", "max", "min"]
COLORS = {"red": "#883955".lower(),"yellow":"#EFEA5A".lower(), "green": "#83E377".lower(), "blue": "#2C699A".lower(), "purple": "#54478C".lower(), "grey": "#737370".lower(),
          "turquoise": "#0DB39E".lower()}
WRAPS = {"de": 18, "fr": 30, "it": 20}


def set_texts(ax, label_texts, labels, title, len_ticks, orientation):
    plt.title(title, fontsize=12)
    ax.set_xlabel(label_texts[0], fontsize=8)
    ax.set_ylabel(label_texts[1], fontsize=8)
    if orientation == "h":
        ax.set_yticks(np.arange(len_ticks))
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks(np.arange(len_ticks))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)

    ax.legend(fontsize=6)


def distribution_plot_1(lang: str, distribution_df: pd.DataFrame, col_x: str, col_y: str, label_texts: list, title: str,
                        filepath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    labels = distribution_df.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in labels]
    fig, ax = plt.subplots()
    ax.bar(distribution_df.reset_index()[col_x], distribution_df.reset_index()[col_y],
           color=COLORS["turquoise"])

    set_texts(ax, label_texts, labels, title, len(labels), "v")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")

def lower_court_effect_plot(df_1, df_2, col_y: str, col_x: str, label_texts: list, legend_texts: list,
                            title: str, filepath: str):
    """
    Dumps horizontal bar plot for opposing value sets.
    """
    labels = df_1.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 35)) for l in labels]
    fig, ax = plt.subplots()
    ax.barh(df_1.reset_index()[col_x], df_1.reset_index()[col_y],
            label=legend_texts[0],
            color=COLORS["green"])
    ax.barh(df_2.reset_index()[col_x], df_2.reset_index()[col_y],
            label=legend_texts[1],
            color=COLORS["purple"])
    set_texts(ax, label_texts, labels, title, len(labels), "h")
    ax.invert_yaxis()
    plt.tight_layout()
    fig.subplots_adjust(left=0.3)
    plt.savefig(filepath, bbox_inches="tight")


def mean_plot_1(mean_df: pd.DataFrame, labels: list, legend: list, title: str, mean_lines: dict, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    fig, ax = plt.subplots()
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"]]
    mean_df.plot(kind='bar', color=colors)
    plt.rcParams.update({'font.size': 8})
    plt.title(title, fontsize=12)
    i = 0
    plt.legend(legend, ncol=len(legend), loc="upper right")
    for key in mean_lines.keys():
        plt.axhline(y=mean_lines[key], c=colors[i], linestyle='dashed', label="horizontal")
        legend.append(key)
        i += 1
        plt.legend(legend, loc="upper left")
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def mean_plot_2(lang:str, N, mean_df: pd.DataFrame, cols: list, ax_labels: list, x_labels, legend_texts: tuple, title: str, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    plt.clf()
    plt.title(title, fontsize=12)
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots()
    bar1 = ax.bar(ind, mean_df.loc[[cols[0]]].values.flatten().tolist(), width,
                  color=COLORS["red"])
    bar2 = ax.bar(ind + width, mean_df.loc[[cols[1]]].values.flatten().tolist(), width,
                   color=COLORS["purple"])
    bar3 = ax.bar(ind + width * 2, mean_df.loc[[cols[2]]].values.flatten().tolist(), width,
                   color=COLORS["blue"])
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])

    if N > 10:
        x_labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in x_labels]
        ax.set_xticks(ind + width)
        ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
    else:
        plt.xticks(ind + width, x_labels)
    plt.legend((bar1, bar2, bar3), legend_texts)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def ttest_plot(lang, pvalue_df: pd.DataFrame, col_x, col_y, alpha, label_texts, title, filepath):
    """
    Dumps a stem plot from p-values.
    """
    labels = pvalue_df[col_y].values
    labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in labels]
    fig, ax = plt.subplots()
    stem_plt = ax.stem(pvalue_df[col_y], pvalue_df[col_x], linefmt=COLORS["turquoise"], use_line_collection=True)
    (markers, stemlines, baseline) = stem_plt
    plt.setp(markers, markeredgecolor=COLORS["turquoise"])
    plt.setp(baseline, color=COLORS["turquoise"])
    plt.axhline(y=alpha, c="black", linestyle='dashed', label="horizontal")
    set_texts(ax, label_texts, labels, title, len(labels), "v")
    plt.legend(["α=0.05"])
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def IAA_Agreement_plots():
    """
    @Todo
    """
    pass
