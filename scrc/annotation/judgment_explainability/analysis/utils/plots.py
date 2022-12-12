from pathlib import Path
from textwrap import wrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
LANGUAGES = ["de", "fr", "it"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LABELS_OCCLUSION = ["Lower court", "Supports judgment", "Opposes judgment", "Neutral"]
AGGREGATIONS = ["mean", "max", "min"]
COLORS = {"red": "#883955".lower(), "yellow": "#EFEA5A".lower(), "green": "#83E377".lower(), "blue": "#2C699A".lower(),
          "purple": "#54478C".lower(), "grey": "#737370".lower(),
          "turquoise": "#0DB39E".lower(), "dark blue": "#163650".lower(), "dark green": "#25691c".lower()}

WRAPS = {"de": 18, "fr": 30, "it": 20}


def set_texts(ax, label_texts, labels, title, len_ticks, orientation, rotation, fontsize_ticks: int):
    """
    @todo
    """
    plt.title(title, fontsize=12)
    ax.set_xlabel(label_texts[0], fontsize=10)
    ax.set_ylabel(label_texts[1], fontsize=10)
    if orientation == "h":
        ax.set_yticks(np.arange(len_ticks))
        ax.set_yticklabels(labels, fontsize=fontsize_ticks, rotation=rotation)
    else:
        ax.set_xticks(np.arange(len_ticks))
        ax.set_xticklabels(labels, fontsize=fontsize_ticks, rotation=rotation)


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


def export_legend(legend, filename):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=1200, bbox_inches=bbox)


def scatter_legend(colors: list, labels, legendpath):
    patches = [Patch(color="white", label="+ : Prediction = 1"), Patch(color="white", label="o : Prediction = 0")]
    c = 0
    for color in colors:
        patches.append(Patch(color=color, label=labels[c]))
        c += 1
    legend = plt.legend(handles=patches, fontsize=12, title="Explainability Labels")
    export_legend(legend, legendpath)


def get_df_slices(df_list: list):
    df_list_return = []
    for df_a in df_list:
        df_a_0, df_a_1 = df_a[df_a["prediction"] == 0], df_a[df_a["prediction"] == 1]
        for df_b in [df_a_0, df_a_1]:
            df_b_s = df_b[df_b["significance_confidence_direction"] == True]
            df_b_s = df_b_s[df_b_s["significance_norm_explainability_score"] == True]
            df_b_ns = df_b.drop(list(df_b_s.index))
            df_list_return.append(df_b_s)
            df_list_return.append(df_b_ns)
    return df_list_return


def flipped_distribution_plot_1(lang: str, distribution_df_list: list, width: float, col_x: str, col_y_1: str,
                                col_y_2: str,
                                label_texts: list, legend_texts: list, title: str,
                                filepath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    colors = {"de": ["#C36F8C".lower(), "#652A3F".lower()], "fr": ["#776AB4".lower(), "#41376D".lower()],
              "it": ["#408BC9".lower(), "#1F4B6F".lower()]}
    fig, ax = plt.subplots(dpi=1200)
    labels = get_labels_from_list(distribution_df_list, col_x)
    i = 0
    shift = (-0.375 / 2)
    for distribution_df in distribution_df_list:
        distribution_df = preprocessing.normalize_df_length(distribution_df, col_x, labels)
        ind = np.arange(len(labels))
        if col_x == "lower_court":
            set_texts(ax, label_texts, labels, title, len(labels), "v", 20, 8)
        else:
            set_texts(ax, label_texts, labels, title, len(labels), "v", 0, 9)
        if not (distribution_df[col_y_1] == 0).all():
            ax.bar(ind + shift, distribution_df[col_y_1], width, color=colors[lang][i], hatch='//', edgecolor="black")
        if not (distribution_df[col_y_2] == 0).all():
            ax.bar(ind + shift, distribution_df[col_y_2], width, color=colors[lang][i], edgecolor="black")
        shift = shift + width
        i += 1

    legend = ax.legend(legend_texts, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=8)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_extra_artists=(legend,), bbox_inches="tight")


def flipped_distribution_plot_2(distribution_df_list: list, width: float, col_x: str, col_y_1: str,
                                col_y_2: str, label_texts: list, legend_texts: list, title: str,
                                filepath: str, legendpath: str):
    """
    Dumps vertical bar plot for distributions.
    """
    colors = ["#C36F8C".lower(), "#652A3F".lower(),
              "#776AB4".lower(), "#41376D".lower(),
              "#408BC9".lower(), "#1F4B6F".lower()]
    fig = plt.figure(dpi=1200)
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
    axs = gs.subplots(sharex='col', sharey='row')
    i, j, fig_count = 0, 0, 0
    for distribution_dfs in distribution_df_list:
        labels = get_labels_from_list(distribution_dfs, col_x)
        c = 0
        shift = width
        ind = np.arange(len(labels))
        axs[i, j].set_xticks(ind + shift*3.25, [label.replace(' ', '\n') for label in LABELS_OCCLUSION[1:]], fontsize=6)
        for distribution_df in distribution_dfs:
            distribution_df = preprocessing.normalize_df_length(distribution_df, col_x, labels)
            if not (distribution_df[col_y_1] == 0).all():
                axs[i, j].bar(ind + shift, distribution_df[col_y_1], width, color=colors[c], hatch='//',
                              edgecolor="black")
            if not (distribution_df[col_y_2] == 0).all():
                axs[i, j].bar(ind + shift, distribution_df[col_y_2], width, color=colors[c], edgecolor="black")
            axs[i, j].grid(axis="y")
            axs[i, j].annotate(fig_count, (2.1, 0))
            shift = shift + width
            c += 1
        j += 1
        fig_count += 1
        if j == 2:
            i = 1
            j = 0

    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    for ax in axs.flat:
        ax.label_outer()
    fig.text(0.5, 0.9, title, fontsize=12, ha='center')
    fig.text(0.5, 0, label_texts[0], ha='center')
    fig.text(0, 0.5, label_texts[1], va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.figure().clear()
    c = 0
    patches = []
    for color in colors:
        patches.append(Patch(color=color, label=legend_texts[c],edgecolor="black"))
        patches.append(Patch(color=color, label=legend_texts[c], hatch='//', edgecolor="black"))
        c += 1
    legend = plt.legend(handles=patches, bbox_to_anchor=(1, 0.5), loc='center left', fontsize=10)
    export_legend(legend, legendpath)


def effect_plot(df_1: pd.DataFrame, df_2: pd.DataFrame, col_y: str, col_x: str, label_texts: list, legend_texts: list,
                xlim: list, title: str, filepath: str):
    """
    Dumps horizontal bar plot for opposing value sets.
    """
    colors = {"green": ["#93E788".lower(), "#47D534".lower()], "purple": ["#8477BB".lower(), "#41376D".lower()]}
    fig, ax = plt.subplots(dpi=1200, figsize=(9, 6))
    labels = df_1[col_x].values

    ax.barh(df_1[df_1[f"significance_{col_y}"] == False][col_x].values,
            df_1[df_1[f"significance_{col_y}"] == False][col_y],
            color=colors["green"][0])
    ax.barh(df_1[df_1[f"significance_{col_y}"] == True][col_x].values,
            df_1[df_1[f"significance_{col_y}"] == True][col_y],
            color=colors["green"][1])
    ax.barh(df_2[df_2[f"significance_{col_y}"] == False][col_x].values,
            df_2[df_2[f"significance_{col_y}"] == False][col_y],
            color=colors["purple"][0])
    ax.barh(df_2[df_2[f"significance_{col_y}"] == True][col_x].values,
            df_2[df_2[f"significance_{col_y}"] == True][col_y],
            color=colors["purple"][1])
    set_texts(ax, label_texts, labels, title, len(labels), "h", 0, 9)

    legend1 = plt.legend(labels=legend_texts, fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.5))
    ax.add_artist(legend1)
    ax.invert_yaxis()

    plt.xlim(xlim)
    plt.tight_layout()
    plt.grid(axis="x")
    fig.subplots_adjust(left=0.3)
    plt.savefig(filepath, bbox_extra_artists=(legend1,), bbox_inches="tight")


def mean_plot_1(mean_df: pd.DataFrame, labels: list, legend_texts: list, ylim: list, title: str,
                error_bars: pd.DataFrame, mean_lines: dict,
                filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    With optional mean axlines.
    """
    plt.subplots(dpi=1200)
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"]]

    if not error_bars.empty:
        mean_df.plot(kind='bar', yerr=error_bars.T.values, error_kw=dict(lw=0.5, capsize=5, capthick=0.5), color=colors, ecolor='black')
    else:
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


def bar_ax(ax, ind, errorbars, mean_df, rows, i, width, colors):
    if not errorbars.empty:
        bar = ax.bar(ind, mean_df.loc[[rows[i]]].values.flatten().tolist(), width,
           color=colors[i], yerr=errorbars.loc[[rows[i]]].values, error_kw=dict(lw=0.5, capsize=5, capthick=0.5),
           ecolor='black')
    else:
        bar = ax.bar(ind, mean_df.loc[[rows[i]]].values.flatten().tolist(), width,
               color=colors[i])
    return bar


def mean_plot_2(N: int, mean_df: pd.DataFrame, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                ylim: list, errorbars: pd.DataFrame, title: str, filepath: str):
    """
    Dumps a vertical bar plot for mean values.
    """
    colors = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    plt.clf()
    ind = np.arange(N)
    width = 0.25
    fig, ax = plt.subplots(dpi=1200)

    plt.title(title, fontsize=12)
    bar1 = bar_ax(ax, ind, errorbars, mean_df, rows, 0, width, colors)
    bar2 = bar_ax(ax, ind + width, errorbars, mean_df, rows, 1, width, colors)
    bar3 = bar_ax(ax, ind + width * 2, errorbars, mean_df, rows, 2, width, colors)
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


def mean_plot_3(N: int, mean_df_list: list, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                ylim: list, errorbar_list: list, title: str, filepath: str, legendpath: str):
    """
    Dumps a vertical bar plot for mean values.
    """
    colors = [COLORS["red"], COLORS["purple"], COLORS["blue"]]
    plt.clf()
    ind = np.arange(N)
    width = 0.25
    fig = plt.figure(dpi=1200)
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
    axs = gs.subplots(sharex='col', sharey='row')
    i, j = 0, 0
    for fig_count in range(0, 4):
        bar1 = axs[i, j].bar(ind, mean_df_list[fig_count].loc[[rows[0]]].values.flatten().tolist(), width,
                             color=colors[0], yerr=errorbar_list[fig_count].loc[[rows[0]]].values, error_kw=dict(lw=0.5,
                             capsize=5, capthick=0.5),ecolor='black')
        bar2 = axs[i, j].bar(ind + width, mean_df_list[fig_count].loc[[rows[1]]].values.flatten().tolist(), width,
                             color=colors[1], yerr=errorbar_list[fig_count].loc[[rows[1]]].values, error_kw=dict(lw=0.5,
                             capsize=5, capthick=0.5),
                             ecolor='black')
        bar3 = axs[i, j].bar(ind + width * 2, mean_df_list[fig_count].loc[[rows[2]]].values.flatten().tolist(), width,
                             color=colors[2], yerr=errorbar_list[fig_count].loc[[rows[2]]].values, error_kw=dict(lw=0.5,
                             capsize=5, capthick=0.5),
                             ecolor='black')
        axs[i, j].grid(axis="y")
        axs[i, j].set_ylim(ylim)
        axs[i, j].annotate(fig_count, (2.5, 0.5))
        axs[i, j].set_xticks(ind + width, x_labels, fontsize=6)
        j += 1
        if j == 2:
            i = 1
            j = 0

    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    for ax in axs.flat:
        ax.label_outer()
    fig.text(0.5, 0.9, title, fontsize=12, ha='center')
    fig.text(0.5, 0, ax_labels[0], ha='center')
    fig.text(0, 0.5, ax_labels[1], va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    legend = plt.legend((bar1, bar2, bar3), legend_texts, bbox_to_anchor=(1, 0.5))
    plt.figure().clear()
    export_legend(legend, legendpath)


def scatter_axis(axs, i, j, df, marker, cmap, alpha):
    axs[i, j].scatter(x=df["confidence_scaled"], y=df["norm_explainability_score"],
                      c=df["numeric_label_human"], alpha=alpha, marker=marker,
                      cmap=cmap, s=20)


def scatter_plot(df_list, title, filepath: str, colors: list):
    fig = plt.figure(dpi=1200, figsize=(6, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
    axs = gs.subplots(sharex='col', sharey='row')

    i, j, fig_count = 0, 0, 0
    if len(df_list) == 2:
        for df_1, df_2 in zip(df_list[0], df_list[1]):
            df_1_0_s, df_1_0_ns, df_1_1_s, df_1_1_ns, df_2_0_s, df_2_0_ns, df_2_1_s, df_2_1_ns = get_df_slices(
                [df_1, df_2])
            scatter_axis(axs, i, j, df_1_0_s, "o", matplotlib.colors.ListedColormap(colors[2]), 0.2)
            scatter_axis(axs, i, j, df_1_0_ns, "o", matplotlib.colors.ListedColormap(colors[0]), 0.2)
            scatter_axis(axs, i, j, df_1_1_s, "+", matplotlib.colors.ListedColormap(colors[2]), 0.2)
            scatter_axis(axs, i, j, df_1_1_ns, "+", matplotlib.colors.ListedColormap(colors[0]), 0.2)
            scatter_axis(axs, i, j, df_2_0_s, "o", matplotlib.colors.ListedColormap(colors[3]), 0.2)
            scatter_axis(axs, i, j, df_2_0_ns, "o", matplotlib.colors.ListedColormap(colors[1]), 0.2)
            scatter_axis(axs, i, j, df_2_1_s, "+", matplotlib.colors.ListedColormap(colors[3]), 0.2)
            scatter_axis(axs, i, j, df_2_1_ns, "+", matplotlib.colors.ListedColormap(colors[1]), 0.2)
            axs[i, j].grid()
            axs[i, j].set_ylim([-0.3, 0.3])
            axs[i, j].annotate(fig_count, (0.5, -0.29999))
            axs[i, j].axhline(0, color='grey')
            j += 1
            fig_count += 1
            if j == 2:
                i = 1
                j = 0

    if len(df_list) == 4:
        for df in df_list:
            df_1_0_s, df_1_0_ns, df_1_1_s, df_1_1_ns = get_df_slices([df])
            scatter_axis(axs, i, j, df_1_0_s, "o", matplotlib.colors.ListedColormap([colors[1]]), 0.3)
            scatter_axis(axs, i, j, df_1_0_ns, "o", matplotlib.colors.ListedColormap([colors[0]]), 0.3)
            scatter_axis(axs, i, j, df_1_1_s, "+", matplotlib.colors.ListedColormap([colors[1]]), 0.3)
            scatter_axis(axs, i, j, df_1_1_ns, "+", matplotlib.colors.ListedColormap([colors[0]]), 0.3)
            axs[i, j].grid()
            axs[i, j].set_ylim([-0.3, 0.3])
            axs[i, j].annotate(fig_count, (0.5, -0.29999))
            axs[i, j].axhline(0, color='grey')
            j += 1
            fig_count += 1
            if j == 2:
                i = 1
                j = 0

    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    for ax in axs.flat:
        ax.label_outer()
    fig.text(0.5, 0.9, title, fontsize=12, ha='center')
    fig.text(0.5, 0, "Scaled Confidence", ha='center')
    fig.text(0, 0.5, "Normalized Explainability Score", va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
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
                errorbars=pd.DataFrame(),
                title="Distribution of Lower Courts over Legal Areas",
                filepath=f'plots/lc_distribution_la_{lang}.png')


def multilingual_annotation_plot(df_list: list):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    df = df_list[0].merge(df_list[1], on="index", how="inner", suffixes=(f'_{LANGUAGES[0]}', f'_{LANGUAGES[1]}'))
    df = df.merge(df_list[2], on="index", how="inner").rename(columns={'mean_token': f'mean_token_{LANGUAGES[2]}',
                                                                       "error": f"error_{LANGUAGES[2]}"})
    df.drop([f"label_{LANGUAGES[0]}", f"label_{LANGUAGES[1]}", "index"], axis=1, inplace=True)
    errorbars = df.set_index("label")[[f"error_{l}" for l in LANGUAGES]].T
    errorbars.index = [f"mean_token_{l}" for l in LANGUAGES]
    mean_plot_2(len(LABELS_OCCLUSION), df.set_index("label")[[f"mean_token_{l}" for l in LANGUAGES]].T,
                rows=[f"mean_token_{lang}" for lang in LANGUAGES],
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=LABELS_OCCLUSION,
                legend_texts=tuple([f"Mean Number of Tokens {i.upper()}" for i in LANGUAGES]),
                ylim=[],
                errorbars=errorbars,
                title="Token Distribution of Annotation Labels in Gold Standard Dataset.",
                filepath=f"plots/ann_mean_tokens_exp_labels_gold.png")


def create_lc_group_by_flipped_plot(lang: str, occlusion_df: pd.DataFrame, cols: list, label_texts: list,
                                    legend_texts: list, title: str, filepath: str):
    flipped_df_list = []
    for df in [occlusion_df[occlusion_df["prediction"] == p] for p in [0, 1]]:
        flipped_df_list.append(preprocessing.group_by_flipped(df, cols[0]))
    flipped_distribution_plot_1(lang, flipped_df_list, width=0.375, col_x=cols[0], col_y_1=cols[1],
                                col_y_2=cols[2],
                                label_texts=label_texts, legend_texts=legend_texts,
                                title=title, filepath=filepath.format(1))


def create_occ_group_by_flipped_plot(occlusion_df_dict: dict):
    legend_texts = [f"{lst[0]}Flipped Prediction {lst[1]} {{}}" for lst in [["", 0], ["", 1], ["Not ", 0], ["Not ", 1]]]
    cols = ["explainability_label", "has_not_flipped", "has_flipped"]

    flipped_df_list = []
    for nr in occlusion_df_dict.keys():
        prediction_list = []
        for df_1 in occlusion_df_dict[nr]:
            for df_2 in [df_1[df_1["prediction"] == p] for p in [0, 1]]:
                prediction_list.append(preprocessing.group_by_flipped(df_2, cols[0]))
        flipped_df_list.append(prediction_list)
    flipped_distribution_plot_2(flipped_df_list, width=0.125, col_x=cols[0], col_y_1=cols[1],
                                col_y_2=cols[2],
                                label_texts=["Explainability label", "Number of Experiments"],
                                legend_texts=[string.format(l.upper()) for string in legend_texts for l in
                                              LANGUAGES],
                                title="Distribution of Flipped Sentences Occlusion Experiments",
                                filepath='plots/occ_flipped_distribution.png',
                                legendpath='plots/occ_flipped_distribution_legend.png')


def create_effect_plot(occlusion_df_dict: dict, cols: list,
                       label_texts: list, legend_texts: list, xlim: list, title: str, filepath: str):
    for key in occlusion_df_dict:
        if key in LANGUAGES:
            mean_pos_df_list, mean_neg_df_list = [], []
            for occlusion_df in occlusion_df_dict[key]:
                mean_pos_df = preprocessing.get_one_sided_effect_df(
                    occlusion_df[occlusion_df["confidence_direction"] > 0],
                    occlusion_df_dict[f"{key}_mu"], cols[0], "pos")
                mean_neg_df = preprocessing.get_one_sided_effect_df(
                    occlusion_df[occlusion_df["confidence_direction"] < 0],
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


def create_scatter_plot(occlusion_df_list: list, title: str, filepath: str, colors_p: list, colors_l: list,
                        labels: list, legendpath: str):
    scatter_plot(occlusion_df_list, title=title, filepath=filepath, colors=colors_p)
    scatter_legend(colors_l, labels, legendpath)


def preprocessing_scatter_plot(occlusion_df_dict):
    for l in LANGUAGES:
        correct_df_list = []
        incorrect_df_list = []
        for key in occlusion_df_dict[l]:
            if key.startswith("c", 0):
                correct_df_list.append(occlusion_df_dict[l][key])  # Appends all correct o_j and s_j
            if key.startswith("f", 0):
                incorrect_df_list.append(occlusion_df_dict[l][key])
            if key.startswith("s", 0):
                create_scatter_plot(occlusion_df_dict[l][key],
                                    f"Trend Supports Judgement (effect on confidence) {key.upper()}",
                                    f"plots/occ_{key}_effect_{l}.png", colors_p=[COLORS['dark blue'], "#ff8800"],
                                    colors_l=[COLORS['dark blue'], "#ff8800"],
                                    labels=[f"{LABELS_OCCLUSION[1]} significant", LABELS_OCCLUSION[1]],
                                    legendpath=f"plots/occ_{key}_effect_legend_{l}.png")
            if key.startswith("o", 0):
                create_scatter_plot(occlusion_df_dict[l][key],
                                    f"Trend Opposes Judgement (effect on confidence) {l.upper()}",
                                    f"plots/occ_{key}_effect_{l}.png", colors_p=[COLORS['red'], COLORS["dark green"]],
                                    colors_l=[COLORS['red'], COLORS["dark green"]],
                                    labels=[f"{LABELS_OCCLUSION[2]} significant", LABELS_OCCLUSION[2]],
                                    legendpath=f"plots/occ_{key}_effect_legend_{l}.png")

        # Correct classification plots

        labels = [f"{label} significant" for label in [LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]]
        create_scatter_plot(correct_df_list,
                            f"Models Classification of Explainability Label (Correctly Classified) {l.upper()}",
                            f"plots/occ_correct_classification_{l}.png",
                            colors_p=[[COLORS['dark blue']], [COLORS["red"]], ["#ff8800"], [COLORS["dark green"]]],
                            colors_l=[COLORS['dark blue'], COLORS['red'], "#ff8800", COLORS["dark green"]],
                            labels=[LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]] + labels,
                            legendpath=f"plots/occ_correct_classification_legend_{l}.png")
        # False classification plots
        labels = [f"{label} significant" for label in
                  [LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]]

        create_scatter_plot(incorrect_df_list,
                            f"Models Classification of Explainability Label (Incorrectly Classified){l.upper()}",
                            f"plots/occ_false_classification_{l}.png",
                            colors_p=[[COLORS['dark blue'], COLORS['yellow']], [COLORS['yellow'], COLORS["red"]],
                                      ["#ff8800", "#8477bb"], ["#8477bb", COLORS["dark green"]]],
                            colors_l=[COLORS["yellow"], COLORS['dark blue'], COLORS['red'], "#8477bb", "#ff8800",
                                      COLORS["dark green"]],
                            labels=[LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]] + labels,
                            legendpath=f"plots/occ_false_classification_legend_{l}.png")


def create_multilingual_occlusion_plot(df_dict: dict):
    """
    Prepares language Dataframes for plots.
    Creates multilingual plots.
    """
    mean_df_list = []
    error_df_list = []
    for key in df_dict:
        df = pd.concat(df_dict[key]).set_index(pd.Index(LANGUAGES))
        preprocessing.write_csv(f"tables/occ_mean_chunk_length_{key}.csv", df)
        mean_df_list.append(df[LABELS_OCCLUSION[1:]])
        error_df_list.append(df[[f"{label}_error" for label in LABELS_OCCLUSION[1:]]])
    mean_plot_3(len(LABELS_OCCLUSION[1:]), mean_df_list,
                rows=LANGUAGES,
                ax_labels=["Explainability Labels", "Number of Tokens"],
                x_labels=[label.replace(' ', '\n') for label in LABELS_OCCLUSION[1:]],
                legend_texts=tuple([f"Mean Chunk Length {i.upper()}" for i in LANGUAGES]),
                ylim=[0, 120],
                errorbar_list=error_df_list,
                title=f"Chunk Length per Explainability Label in Occlusion Experiment",
                filepath=f"plots/occ_mean_chunk_length.png", legendpath=f"plots/occ_mean_chunk_length_legend.png")


def IAA_Agreement_plots():
    """
    @Todo
    """
    pass
