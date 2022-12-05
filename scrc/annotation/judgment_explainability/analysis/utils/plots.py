from textwrap import wrap

import matplotlib
import numpy as np
import pandas as pd
import re
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


"""
colors = {"green": ["#37BA26".lower(), "#47d534".lower(), "#65DC56".lower(), "#83E377".lower()],
             "purple": ["#54478C".lower(), "#6B5CAD".lower(), "#8477BB".lower(), "#9C92C8".lower()]}

"""


def set_texts(ax, label_texts, labels, title, len_ticks, orientation):
    """
    @todo
    """
    plt.title(title, fontsize=12)
    ax.set_xlabel(label_texts[0], fontsize=8)
    ax.set_ylabel(label_texts[1], fontsize=8)
    if orientation == "h":
        ax.set_yticks(np.arange(len_ticks))
        ax.set_yticklabels(labels, fontsize=6)
    else:
        ax.set_xticks(np.arange(len_ticks))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)


def add_mean_lines(mean_lines: dict, legend: list, colors: list):
    i = 0
    for key in mean_lines.keys():
        plt.axhline(y=mean_lines[key], c=colors[i], linestyle='dashed', label="horizontal")
        legend.append(key)
        i += 1
        plt.legend(legend, loc='best')


def add_var_numbers(bar, text, ):
    # Add counts above the two bar graphs
    for rect in bar:
        plt.text(-0.0001, rect.get_y() + rect.get_height() / 2.0, f'{text}', ha='center', va='center', c="white")


def distribution_plot_1(lang: str, distribution_df: pd.DataFrame, col_x: str, col_y_1: str,
                        label_texts: list, title: str,
                        filepath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    labels = distribution_df[col_x].values
    labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in labels]
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    ax.bar(distribution_df[col_x], distribution_df[col_y_1],
           color=COLORS["turquoise"])
    set_texts(ax, label_texts, labels, title, len(labels), "v")
    ax.legend(fontsize=6)
    if col_x != "lower_court":
        ax.set_xticklabels(labels, fontsize=6, rotation=0)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def distribution_plot_2(lang: str, distribution_df_list: list, col_x: str, col_y_1: str, col_y_2: str,
                        label_texts: list, legend_texts: list, title: str,
                        filepath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    colors = [[COLORS["turquoise"], COLORS["purple"]], [COLORS["green"], COLORS["red"]]]
    i = 0
    width = 0.25
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    shift = 0
    legend = []
    for distribution_df in distribution_df_list:
        labels = distribution_df[col_x].values
        labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in labels]
        ind = np.arange(len(labels))
        if i == 0:
            set_texts(ax, label_texts, labels, title, len(labels), "v")
            ax.legend(fontsize=6)
            if col_x != "lower_court":
                ax.set_xticklabels(labels, fontsize=6, rotation=0)
        bar1 = ax.bar(ind + shift, distribution_df[col_y_1], width, color=colors[i][0])
        bar2 = ax.bar(ind + shift, distribution_df[col_y_2], width, color=colors[i][1])
        i += 1
        shift = width
        legend = legend + [bar1, bar2]
    plt.legend(legend, legend_texts)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def effect_plot(df_1: pd.DataFrame, df_2: pd.DataFrame, col_y: str, col_x: str, label_texts: list, legend_texts: list,
                title: str, filepath: str):
    """
    Dumps horizontal bar plot for opposing value sets.
    @ Todo change axis, add grid
    """
    second_legend = {"N": "Neutral", "O": "Opposes Judgement", "S": "Supports Judgement"}
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    labels = df_1.reset_index()[col_x].values
    labels = ['\n'.join(wrap(l, 35)) for l in labels]
    if col_x == 'explainability_label':
        labels = []
        for char in second_legend.keys():
            labels = labels + [f'{char}{nr}' for nr in range(1, 5)]
        df_1[col_x] = labels
        df_2[col_x] = labels

    bar1 = ax.barh(df_1.reset_index()[col_x].values, df_1.reset_index()[col_y],
                           color=COLORS["green"])
    bar2 = ax.barh(df_2.reset_index()[col_x].values, df_2.reset_index()[col_y],
                           color=COLORS["purple"])
    set_texts(ax, label_texts, labels, title, len(labels), "h")

    legend1 = plt.legend(labels=legend_texts, fontsize=6)
    ax.add_artist(legend1)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.grid()
    fig.subplots_adjust(left=0.3)
    plt.savefig(filepath, bbox_extra_artists=(), bbox_inches="tight")

    if col_x == 'explainability_label':
        labels = [f"{label}: {second_legend[re.sub('[0-9]', '',label)]}" for label in labels]
        legend2 = plt.legend(
            handles=[Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0) for label in labels],
            labels=labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
        plt.gca().add_artist(legend2)
        plt.savefig(filepath, bbox_extra_artists=(legend2,), bbox_inches="tight")


def mean_plot_1(mean_df: pd.DataFrame, labels: list, legend_texts: list, title: str, mean_lines: dict, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"]]
    mean_df.plot(kind='bar', color=colors)
    plt.rcParams.update({'font.size': 8})
    plt.title(title, fontsize=12)
    plt.legend(legend_texts, ncol=len(legend_texts), loc='best')
    add_mean_lines(mean_lines, legend_texts, colors)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.xticks(rotation=0)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def mean_plot_2(lang: str, N: int, mean_df: pd.DataFrame, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                ylim: list, title: str, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    colors = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    plt.clf()
    ind = np.arange(N)
    width = 0.25
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    ax.clear()
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

    if N > 10:
        x_labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in x_labels]
        ax.set_xticks(ind + width)
        ax.set_xticklabels(x_labels, fontsize=6, rotation=90)
    else:
        plt.xticks(ind + width, x_labels)
    plt.legend((bar1, bar2, bar3), legend_texts)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def ttest_plot(lang, pvalue_df: pd.DataFrame, col_x, col_y, alpha, label_texts, title, filepath):
    """
    Dumps a stem plot from p-values.
    """
    labels = pvalue_df[col_y].values
    labels = ['\n'.join(wrap(l, WRAPS[lang])) for l in labels]
    plt.figure(dpi=1200)
    fig, ax = plt.subplots()
    stem_plt = ax.stem(pvalue_df[col_y], pvalue_df[col_x], linefmt=COLORS["turquoise"], use_line_collection=True)
    (markers, stemlines, baseline) = stem_plt
    plt.setp(markers, markeredgecolor=COLORS["turquoise"])
    plt.setp(baseline, color=COLORS["turquoise"])
    plt.axhline(y=alpha, c="black", linestyle='dashed', label="horizontal")
    set_texts(ax, label_texts, labels, title, len(labels), "v")
    ax.legend(fontsize=6)
    plt.legend(["α=0.05"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def scatter_plot(df_1, df_2, mode: bool, title, filepath):
    fig = plt.figure(dpi=1200)
    ax = fig.add_subplot(111)
    colors = [COLORS["red"], COLORS['green'], COLORS['blue']]
    plt.title(title, fontsize=12)
    if mode:
        ax.scatter(x=df_1["confidence_scaled"], y=df_1["norm_explainability_score"],
                              c=df_1["numeric_label_model"], s=10, alpha=0.5,
                              cmap=matplotlib.colors.ListedColormap([colors[1]]))
        ax.scatter(x=df_2["confidence_scaled"], y=df_2["norm_explainability_score"],
                              c=df_2["numeric_label_model"], s=10, alpha=0.5,
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
        ax.scatter(x=df_1["confidence_scaled"],y=df_1["norm_explainability_score"],
                c=df_1["numeric_label_model"], s=10, alpha=0.5,
                cmap=matplotlib.colors.ListedColormap([colors[0], colors[1]]))
        ax.scatter( x=df_2["confidence_scaled"], y=df_2["norm_explainability_score"],
                c=df_2["numeric_label_model"], s=10, alpha=0.5,
                cmap=matplotlib.colors.ListedColormap([colors[0], colors[2]]))
        patches = []
        labels = [LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]
        i = 0
        for color in colors:
            patches.append(Patch(color=color, label=labels[i]))
            i += 1
        legend1 = ax.legend(handles=patches,
                            loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, title="Explainability Labels")
        ax.add_artist(legend1)
    ax.set_xlabel("confidence_scaled")
    ax.set_ylabel("norm_explainability_score")
    # Add prediction threshold
    df = pd.concat([df_1, df_2])
    line = df[df["prediction"] == 1]["confidence_scaled"].mean()
    axvline = ax.axvline(x=line, c="black", linestyle='dashed', label="Approximate Threshold for Prediction = 1")
    legend2 = ax.legend(handles=[axvline], fontsize=8)
    ax.add_artist(legend2)
    ax.grid()
    plt.tight_layout()
    plt.savefig(filepath, bbox_extra_artists=(legend1, legend2), bbox_inches="tight")
    plt.figure().clear()


def create_lc_la_distribution_plot(lang: str, lc_la_df: pd.DataFrame):
    """
    Creates lower_court legal_area distribution plot.
    """
    mean_plot_2(lang, len(lc_la_df.columns),
                lc_la_df, rows=LEGAL_AREAS,
                ax_labels=["Lower Court", ""],
                x_labels=lc_la_df.columns,
                legend_texts=tuple([la.replace("_", " ") for la in LEGAL_AREAS]),
                ylim=[],
                title="Distribution of Lower Courts in Legal Areas",
                filepath=f'plots/lc_distribution_la_{lang}.png')


def create_ttest_plot(lang: str, mu_df: pd.DataFrame, sample_df: pd.DataFrame, filepath: str, col, title):
    """
    @todo
    """
    ttest_plot(lang, preprocessing.ttest(preprocessing.group_to_list(sample_df, "lower_court", col), mu_df, col),
               "pvalue",
               "lower_court", 0.05,
               label_texts=["P-value", "Lower Courts"],
               title=title,
               filepath=filepath)


def multilingual_annotation_plot(df_list: list):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = df_list[0].merge(df_list[1], on="index", how="inner", suffixes=(f'_{LANGUAGES[0]}', f'_{LANGUAGES[1]}'))
    df = df.merge(df_list[2], on="index", how="inner").rename(columns={'mean_token': f'mean_token_{LANGUAGES[2]}'})
    df.drop([f"label_{LANGUAGES[0]}", f"label_{LANGUAGES[1]}", "index"], axis=1, inplace=False)
    mean_plot_2("", len(LABELS_OCCLUSION), df.set_index("label").T,
                rows=[f"mean_token_{lang}" for lang in LANGUAGES],
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=LABELS_OCCLUSION,
                legend_texts=tuple([f"Mean Number of Tokens {i.upper()}" for i in LANGUAGES]),
                ylim=[],
                title="Token Distribution of Annotation Labels in Gold Standard Dataset.",
                filepath=f"plots/ann_mean_tokens_exp_labels_gold.png")


def create_group_by_flipped_plot(lang: str, exp_nr, occlusion_df: pd.DataFrame, cols: list, label_texts: list,
                                 legend_texts: list, title: str, filepath: str):
    flipped_df_list = []
    i = 0
    if not "":
        filepath = filepath.format(lang, exp_nr)
    else:
        filepath = filepath.format(lang)
    for df in [occlusion_df[occlusion_df["prediction"] == p] for p in [0, 1]]:
        flipped_df_list.append(preprocessing.group_by_flipped(df, cols[0]))
        i += 1
    distribution_plot_2(lang, flipped_df_list, col_x=cols[0], col_y_1=cols[1], col_y_2=cols[2],
                        label_texts=label_texts, legend_texts=legend_texts, title=title, filepath=filepath)


def create_effect_plot(occlusion_df_dict: dict, distribution_df_dict: dict, cols: list,
                       label_texts: list, legend_texts: list, title: str, filepath: str):
    for key in occlusion_df_dict:
        mean_pos_df_list, mean_neg_df_list = [], []
        for occlusion_df in occlusion_df_dict[key]:
            mean_pos_df = preprocessing.get_one_sided_agg(occlusion_df[occlusion_df["confidence_direction"] > 0],
                                                          distribution_df_dict[key], cols[0])
            mean_neg_df = preprocessing.get_one_sided_agg(occlusion_df[occlusion_df["confidence_direction"] < 0],
                                                          distribution_df_dict[key], cols[0])

            mean_pos_df_list.append(mean_pos_df)
            mean_neg_df_list.append(mean_neg_df)
        effect_plot(pd.concat(mean_pos_df_list), pd.concat(mean_neg_df_list), col_y=cols[1],
                    col_x=cols[0],
                    label_texts=label_texts,
                    legend_texts=legend_texts,
                    title=title,
                    filepath=filepath.format(key))


def create_multilingual_occlusion_plot(df_list: list, nr_exp: int):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = pd.concat(df_list).set_index(pd.Index(LANGUAGES))
    mean_plot_2("", len(LABELS_OCCLUSION[1:]), df,
                rows=LANGUAGES,
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=LABELS_OCCLUSION[1:],
                legend_texts=tuple([f"Mean Chunk Length {i.upper()}" for i in LANGUAGES]),
                ylim=[0, 120],
                title=f"Chunk Length Distribution of Explainability Labels in Occlusion Experiment {nr_exp}",
                filepath=f"plots/occ_mean_chunk_length_{nr_exp}.png")


def IAA_Agreement_plots():
    """
    @Todo
    """
    pass
