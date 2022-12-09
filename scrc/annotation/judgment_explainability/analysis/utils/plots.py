import re
from pathlib import Path
from textwrap import wrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Patch

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
LANGUAGES = ["de", "fr", "it"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
AGGREGATIONS = ["mean", "max", "min"]
COLORS = {"red": "#883955".lower(), "yellow": "#EFEA5A".lower(), "green": "#83E377".lower(), "blue": "#2C699A".lower(),
          "purple": "#54478C".lower(), "grey": "#737370".lower(),
          "turquoise": "#0DB39E".lower()}

WRAPS = {"de": 18, "fr": 30, "it": 20}


def set_texts(ax, label_texts, labels, title, len_ticks, orientation, rotation):
    """
    @todo
    """
    plt.title(title, fontsize=12)
    ax.set_xlabel(label_texts[0], fontsize=10)
    ax.set_ylabel(label_texts[1], fontsize=10)
    if orientation == "h":
        ax.set_yticks(np.arange(len_ticks))
        ax.set_yticklabels(labels, fontsize=9, rotation=rotation)
    else:
        ax.set_xticks(np.arange(len_ticks))
        ax.set_xticklabels(labels, fontsize=9, rotation=rotation)


def add_mean_lines(mean_lines: dict, legend: list, colors: list):
    """
    Adds mean lines to a plot.
    """
    i = 0
    for key in mean_lines.keys():
        plt.axhline(y=mean_lines[key], c=colors[i], linestyle='dashed', label="horizontal")
        legend.append(key)
        i += 1
        plt.legend(legend, loc='best')


def get_labels_from_list(df_list: list, col):
    labels = []
    for df in df_list:
        labels = labels + list(df[col].values)
    return list(set(labels))


def flipped_distribution_plot(distribution_df_list: list, width: float, shift: float, col_x: str, col_y_1: str,
                              col_y_2: str,
                              label_texts: list, legend_texts: list, title: str,
                              filepath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    colors = ["#C36F8C".lower(), "#652A3F".lower(),
              "#776AB4".lower(), "#41376D".lower(),
              "#408BC9".lower(), "#1F4B6F".lower()]
    # @todo Change colors to distinguish languages?
    fig, ax = plt.subplots(dpi=1200)
    labels = get_labels_from_list(distribution_df_list, col_x)
    i = 0
    for distribution_df in distribution_df_list:
        distribution_df = preprocessing.normalize_df_length(distribution_df, col_x, labels)
        ind = np.arange(len(labels))
        if col_x == "lower_court":
            set_texts(ax, label_texts, labels, title, len(labels), "v", 20)
        else:
            set_texts(ax, label_texts, labels, title, len(labels), "v", 0)
        if not (distribution_df[col_y_1] == 0).all():
            ax.bar(ind + shift, distribution_df[col_y_1], width, color=colors[i], hatch='//', edgecolor="black")
        if not (distribution_df[col_y_2] == 0).all():
            ax.bar(ind + shift, distribution_df[col_y_2], width, color=colors[i], edgecolor="black")
        shift = shift + width
        i += 1

    legend = ax.legend(legend_texts, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=8)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_extra_artists=(legend,), bbox_inches="tight")


def effect_plot(df_1: pd.DataFrame, df_2: pd.DataFrame, col_y: str, col_x: str, label_texts: list, legend_texts: list,
                xlim: list, title: str, filepath: str):
    """
    Dumps horizontal bar plot for opposing value sets.
    """
    colors = {"green": ["#93E788".lower(),"#47D534".lower()], "purple": ["#8477BB".lower(), "#41376D".lower()]}
    second_legend = {"N": "Neutral", "O": "Opposes Judgement", "S": "Supports Judgement"}
    fig, ax = plt.subplots(dpi=1200, figsize=(9,6))
    labels = df_1[col_x].values
    if col_x == 'explainability_label':
        labels = []
        for char in second_legend.keys():
            labels = labels + [f'{char}{nr}' for nr in range(1, 5)]
        df_1[col_x] = labels
        df_2[col_x] = labels

    ax.barh(df_1[df_1[f"significance_{col_y}"] == False][col_x].values, df_1[df_1[f"significance_{col_y}"] == False][col_y],
                color=colors["green"][0])
    ax.barh(df_1[df_1[f"significance_{col_y}"] == True][col_x].values, df_1[df_1[f"significance_{col_y}"] == True][col_y],
                color=colors["green"][1])
    ax.barh(df_2[df_2[f"significance_{col_y}"] == False][col_x].values, df_2[df_2[f"significance_{col_y}"] == False][col_y],
                color=colors["purple"][0])
    ax.barh(df_2[df_2[f"significance_{col_y}"] == True][col_x].values, df_2[df_2[f"significance_{col_y}"] == True][col_y],
                color=colors["purple"][1])
    set_texts(ax, label_texts, labels, title, len(labels), "h", 0)

    legend1 = plt.legend(labels=legend_texts, fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)
    ax.invert_yaxis()


    plt.xlim(xlim)
    plt.tight_layout()
    plt.grid(axis="x")
    fig.subplots_adjust(left=0.3)
    plt.savefig(filepath, bbox_extra_artists=(legend1,), bbox_inches="tight")

    if col_x == 'explainability_label':
        labels = [f"{label}: {second_legend[re.sub('[0-9]', '', label)]}" for label in labels]
        legend2 = plt.legend(
            handles=[Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0) for label in labels],
            labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.savefig(filepath, bbox_extra_artists=(legend1, legend2), bbox_inches="tight")


def mean_plot_1(mean_df: pd.DataFrame, labels: list, legend_texts: list, ylim: list, title: str, mean_lines: dict,
                filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    plt.subplots(dpi=1200)
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"]]
    mean_df.plot(kind='bar', color=colors)
    plt.rcParams.update({'font.size': 9})
    plt.title(title, fontsize=12)
    plt.legend(legend_texts, ncol=len(legend_texts), loc='best')
    add_mean_lines(mean_lines, legend_texts, colors)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if len(ylim) != 0:
        plt.ylim(ylim)
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def mean_plot_2(N: int, mean_df: pd.DataFrame, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                ylim: list, title: str, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    """
    colors = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    plt.clf()
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots(dpi=1200)

    plt.title(title, fontsize=12)

    bar1 = ax.bar(ind, mean_df.loc[[rows[0]]].values.flatten().tolist(), width,
                  color=colors[0])
    bar2 = ax.bar(ind + width, mean_df.loc[[rows[1]]].values.flatten().tolist(), width,
                  color=colors[1])
    bar3 = ax.bar(ind + width * 2, mean_df.loc[[rows[2]]].values.flatten().tolist(), width,
                  color=colors[2])
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    if len(ylim) != 0:
        plt.ylim(ylim)
    if N > 6:
        x_labels = ['\n'.join(wrap(l, 15)) for l in x_labels]
        ax.set_xticks(ind + width)
        ax.set_xticklabels(x_labels, fontsize=9, rotation=20)
    else:
        plt.xticks(ind + width, x_labels)
    plt.legend((bar1, bar2, bar3), legend_texts)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    return


def scatter_plot(df_1, df_2, mode: bool, title, filepath):
    fig = plt.figure(dpi=1200)
    ax = fig.add_subplot(111)
    colors = [COLORS["red"], COLORS['green'], COLORS['blue']]
    plt.title(title, fontsize=12)
    df_1_0, df_2_0 = df_1[df_1["prediction"] == 0], df_2[df_2["prediction"] == 0]
    df_1_1, df_2_1 = df_1[df_1["prediction"] == 1], df_2[df_2["prediction"] == 1]
    if mode:
        ax.scatter(x=df_1_0["confidence_scaled"], y=df_1_0["norm_explainability_score"],
                   c=df_1_0["numeric_label_model"], alpha=0.5, marker="o",
                   cmap=matplotlib.colors.ListedColormap([colors[1]]))
        ax.scatter(x=df_2_0["confidence_scaled"], y=df_2_0["norm_explainability_score"],
                   c=df_2_0["numeric_label_model"], alpha=0.5, marker="o",
                   cmap=matplotlib.colors.ListedColormap([colors[2]]))
        ax.scatter(x=df_1_1["confidence_scaled"], y=df_1_1["norm_explainability_score"],
                   c=df_1_1["numeric_label_model"], alpha=0.5, marker="+",
                   cmap=matplotlib.colors.ListedColormap([colors[1]]))
        ax.scatter(x=df_2_1["confidence_scaled"], y=df_2_1["norm_explainability_score"],
                   c=df_2_1["numeric_label_model"], alpha=0.5, marker="+",
                   cmap=matplotlib.colors.ListedColormap([colors[2]]))
        patches = []
        labels = [LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]
        i = 0
        for color in [colors[1], colors[2]]:
            patches.append(Patch(color=color, label=labels[i]))
            i += 1
        legend1 = ax.legend(handles=patches,
                            loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title="Explainability Labels")
        ax.add_artist(legend1)
    else:
        ax.scatter(x=df_1_0["confidence_scaled"], y=df_1_0["norm_explainability_score"],
                   c=df_1_0["numeric_label_model"], alpha=0.5, marker="o",
                   cmap=matplotlib.colors.ListedColormap([colors[0], colors[2]]))
        ax.scatter(x=df_2_0["confidence_scaled"], y=df_2_0["norm_explainability_score"],
                   c=df_2_0["numeric_label_model"], alpha=0.5, marker="o",
                   cmap=matplotlib.colors.ListedColormap([colors[0], colors[2]]))
        ax.scatter(x=df_1_1["confidence_scaled"], y=df_1_1["norm_explainability_score"],
                   c=df_1_1["numeric_label_model"], alpha=0.5, marker="+",
                   cmap=matplotlib.colors.ListedColormap([colors[0], colors[2]]))
        ax.scatter(x=df_2_1["confidence_scaled"], y=df_2_1["norm_explainability_score"],
                   c=df_2_1["numeric_label_model"], alpha=0.5, marker="+",
                   cmap=matplotlib.colors.ListedColormap([colors[0], colors[2]]))
        patches = []
        labels = [LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]
        i = 0
        for color in colors:
            patches.append(Patch(color=color, label=labels[i]))
            i += 1
        legend1 = ax.legend(handles=patches,
                            loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, title="Explainability Labels")
        ax.add_artist(legend1)
    ax.set_xlabel("Scaled Confidence")
    ax.set_ylabel("Normalized Explainability Score")
    ax.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_extra_artists=(legend1,), bbox_inches="tight")
    plt.figure().clear()


def create_lc_la_distribution_plot(lang: str, lc_la_df: pd.DataFrame):
    """
    Creates lower_court legal_area distribution plot.
    """
    mean_plot_2(len(lc_la_df.columns),
                lc_la_df, rows=LEGAL_AREAS,
                ax_labels=["Lower Court", "Ration"],
                x_labels=lc_la_df.columns,
                legend_texts=tuple([la.replace("_", " ") for la in LEGAL_AREAS]),
                ylim=[0, 0.5],
                title="Distribution of Lower Courts over Legal Areas",
                filepath=f'plots/lc_distribution_la_{lang}.png')


def multilingual_annotation_plot(df_list: list):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = df_list[0].merge(df_list[1], on="index", how="inner", suffixes=(f'_{LANGUAGES[0]}', f'_{LANGUAGES[1]}'))
    df = df.merge(df_list[2], on="index", how="inner").rename(columns={'mean_token': f'mean_token_{LANGUAGES[2]}'})
    df.drop([f"label_{LANGUAGES[0]}", f"label_{LANGUAGES[1]}", "index"], axis=1, inplace=False)
    mean_plot_2(len(LABELS_OCCLUSION), df.set_index("label").T,
                rows=[f"mean_token_{lang}" for lang in LANGUAGES],
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=LABELS_OCCLUSION,
                legend_texts=tuple([f"Mean Number of Tokens {i.upper()}" for i in LANGUAGES]),
                ylim=[],
                title="Token Distribution of Annotation Labels in Gold Standard Dataset.",
                filepath=f"plots/ann_mean_tokens_exp_labels_gold.png")


def create_lc_group_by_flipped_plot(occlusion_df: pd.DataFrame, cols: list, label_texts: list,
                                    legend_texts: list, title: str, filepath: str):
    flipped_df_list = []
    for df in [occlusion_df[occlusion_df["prediction"] == p] for p in [0, 1]]:
        flipped_df_list.append(preprocessing.group_by_flipped(df, cols[0]))
    flipped_distribution_plot(flipped_df_list, width=0.375, shift=(-0.375 / 2), col_x=cols[0], col_y_1=cols[1],
                              col_y_2=cols[2],
                              label_texts=label_texts, legend_texts=legend_texts,
                              title=title, filepath=filepath.format(1))


def create_occ_group_by_flipped_plot(occlusion_df_dict: dict, cols: list, label_texts: list,
                                     legend_texts: list, title: str, filepath: str):
    for nr in occlusion_df_dict.keys():
        flipped_df_list = []
        for df_1 in occlusion_df_dict[nr]:
            for df_2 in [df_1[df_1["prediction"] == p] for p in [0, 1]]:
                flipped_df_list.append(preprocessing.group_by_flipped(df_2, cols[0]))
        flipped_distribution_plot(flipped_df_list, width=0.125, shift=(-0.125 * 2.39), col_x=cols[0], col_y_1=cols[1],
                                  col_y_2=cols[2],
                                  label_texts=label_texts, legend_texts=legend_texts,
                                  title=title.format(nr), filepath=filepath.format(nr))


def create_effect_plot(occlusion_df_dict: dict, cols: list,
                       label_texts: list, legend_texts: list, xlim: list, title: str, filepath: str):
    for key in occlusion_df_dict:
        if key in LANGUAGES:
            mean_pos_df_list, mean_neg_df_list = [], []
            for occlusion_df in occlusion_df_dict[key]:
                mean_pos_df = preprocessing.get_one_sided_effect_df(occlusion_df[occlusion_df["confidence_direction"] > 0],
                                                                    occlusion_df_dict[f"{key}_mu"], cols[0], "pos")
                mean_neg_df = preprocessing.get_one_sided_effect_df(occlusion_df[occlusion_df["confidence_direction"] < 0],
                                                                    occlusion_df_dict[f"{key}_mu"], cols[0], "neg")

                mean_pos_df_list.append(mean_pos_df)
                mean_neg_df_list.append(mean_neg_df)
            effect_plot(pd.concat(mean_pos_df_list), pd.concat(mean_neg_df_list), col_y=cols[1],
                col_x=cols[0],
                label_texts=label_texts,
                legend_texts=legend_texts,
                xlim=xlim,
                title=title,
                filepath=filepath.format(key))


def create_scatter_plot(occlusion_df_dict):
    for l in LANGUAGES:
        correct_df_list =[]
        correct_df_list = []
        for key in occlusion_df_dict[l]:
            if key.startswith("c",0):
                correct
                scatter_plot(o_judgement_c, o_judgement_c, mode=True,
                                      title="Models Classification of Explainability Label (Correctly Classified)",
                                      filepath=f'plots/occ_correct_classification_{l}_{nr}.png')
                   """plots.scatter_plot(o_judgement_f, s_judgement_f, mode=False,
                                      title="Models Classification of Explainability Label (Incorrectly Classified)",
                                      filepath=f'plots/occ_false_classification_{l}_{nr}.png')"""






def create_multilingual_occlusion_plot(df_list: list, nr_exp: int):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = pd.concat(df_list).set_index(pd.Index(LANGUAGES))
    preprocessing.write_csv(Path(f"tables/occ_mean_chunk_length_{nr_exp}.csv"),df)
    mean_plot_2(len(LABELS_OCCLUSION[1:]), df,
                rows=LANGUAGES,
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=LABELS_OCCLUSION[1:],
                legend_texts=tuple([f"Mean Chunk Length {i.upper()}" for i in LANGUAGES]),
                ylim=[0, 120],
                title=f"Chunk Length per Explainability Label in {nr_exp} Sentence Occlusion Experiment",
                filepath=f"plots/occ_mean_chunk_length_{nr_exp}.png")


def IAA_Agreement_plots():
    """
    @Todo
    """
    pass
