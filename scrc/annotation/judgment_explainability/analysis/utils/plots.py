import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from scipy.stats import norm

import scrc.annotation.judgment_explainability.analysis.utils.preprocessing as preprocessing

"""
Contains functions for plot creations. Uses preprocessing.py. 
Is used by quantitative_analysis and explanation_creator.
"""

LEGAL_AREAS = ["penal_law", "social_law", "civil_law"]
LANGUAGES = ["de", "fr", "it"]
LABELS = ["Lower court", "Supports judgment", "Opposes judgment"]
LABELS_OCCLUSION = LABELS + ["Neutral"]
COLORS = {"red": "#883955".lower(), "yellow": "#EFEA5A".lower(), "green": "#83E377".lower(), "blue": "#2C699A".lower(),
          "purple": "#54478C".lower(), "grey": "#737370".lower(),
          "turquoise": "#0DB39E".lower(), "dark blue": "#163650".lower(), "dark green": "#0d2818".lower(),
          "light purple": "#8477bb"}


def set_texts(ax, ax_labels, ytick_labels, title, len_ticks, orientation, rotation, fontsize_ticks: int):
    """
    Sets title, xlabel, ylabel and ticks according to orientation for ax.
    """
    plt.title(title, fontsize=12)
    ax.set_xlabel(ax_labels[0], fontsize=10)
    ax.set_ylabel(ax_labels[1], fontsize=10)
    if orientation == "h":
        ax.set_yticks(np.arange(len_ticks))
        ax.set_yticklabels(ytick_labels, fontsize=fontsize_ticks, rotation=rotation)
    else:
        ax.set_xticks(np.arange(len_ticks))
        ax.set_xticklabels(ytick_labels, fontsize=fontsize_ticks, rotation=rotation)


def set_ax_labels(ax_labels: list, ylim: list):
    """
    Sets ax labels and optional y-lim for plot.
    """
    plt.xlabel(ax_labels[0])
    plt.ylabel(ax_labels[1])
    if len(ylim) != 0:
        plt.ylim(ylim)


def add_mean_lines(mean_lines: dict, legend: list, colors: list):
    """
    Adds mean lines to plot.
    """
    i = 0
    for key in mean_lines.keys():
        plt.axhline(y=mean_lines[key], c=colors[i], linestyle='dashed', label="horizontal")
        legend.append(key)
        i += 1
        plt.legend(legend, loc='best')


def get_labels_from_list(df_list: list, col) -> list:
    """
    Returns label list from multiple Dataframes.
    """
    labels = []
    for df in df_list:
        labels = labels + list(df[col].values)
    return list(set(labels))


def export_legend(legend, filename):
    """
    Saves additional legend as PNG.
    """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=1200, bbox_inches=bbox)


def scatter_legend(colors: list, labels: list, filename: str):
    """
    Saves additional legend for scatter plot as PNG.
    """
    plt.figure().clear()
    patches = create_colors_patches(colors, labels) + [Patch(color="white", label="+ : Prediction = 1"),
                                                       Patch(color="white", label="o  : Prediction = 0")]
    legend = plt.legend(handles=patches, fontsize=12, title="Explainability Labels", ncols=3, facecolor='white',
                        framealpha=1)
    export_legend(legend, filename)


def create_colors_patches(colors: list, labels: list) -> list:
    """
    Returns list of colored patches with labels from list.
    """
    patches = []
    c = 0
    for color in colors:
        patches.append(Patch(color=color, label=labels[c]))
        c += 1
    return patches


def create_white_patches(labels) -> list:
    """
    Returns list of white patches with labels from list.
    """
    patches = []
    for i in range(0, len(labels)):
        patches.append(Patch(color="white", label=f"{i + 1}: {labels[i]}"))
    return patches


def flatten_axis(axs):
    """
    Flattens axis.
    """
    for ax in axs.flat:
        ax.set(xlabel='', ylabel='')
    for ax in axs.flat:
        ax.label_outer()


def append_missing_rows(df_long: pd.DataFrame, df_short: pd.DataFrame, col_x: str) -> pd.DataFrame:
    """
    Appends missing lower_court rows to Dataframe.
    Returns Dataframe containing missing lower courts and other columns filled with 0.
    """
    for value in list(df_long[col_x].values):
        if value not in df_short.values:
            col = {column: 0 for column in df_short.columns}
            col[col_x] = value
            df_short = df_short.append(col, ignore_index=True)
    return df_short.sort_values("lower_court")


def get_df_slices(df_list: list) -> list:
    """
    Splits Dataframe according to predictions and significance.
    Returns list of split Dataframes.
    """
    df_list_return = []
    for df_a in df_list:
        for df_b in [df_a[df_a["prediction"] == 0], df_a[df_a["prediction"] == 1]]:
            df_b_s = df_b[df_b["significance_confidence_direction"] == True]
            df_b_s = df_b_s[df_b_s["significance_norm_explainability_score"] == True]
            df_list_return.append(df_b_s)
            df_list_return.append(df_b.drop(list(df_b_s.index)))
    return df_list_return


def lc_flipped_bar_plot(lang: str, distribution_df_list: list, col_x: str, col_y_1: str,
                        col_y_2: str, label_texts: list, legend_texts: list, title: str,
                        filepath: str):
    """
    Creates flipped prediction lower court distribtuion plot.
    Saves figure as PNG and exports legend.
    """
    width = 0.375
    colors = {"de": ["#C36F8C".lower(), "#652A3F".lower()], "fr": ["#776AB4".lower(), "#41376D".lower()],
              "it": ["#408BC9".lower(), "#1F4B6F".lower()]}

    labels = sorted(get_labels_from_list(distribution_df_list, col_x))

    fig, ax = plt.subplots(dpi=1200)
    i = 0
    shift = (-0.375 / 2)
    for distribution_df in distribution_df_list:
        distribution_df = preprocessing.normalize_df_length(distribution_df, col_x, labels)
        ind = np.arange(len(labels))
        set_texts(ax, label_texts, [i + 1 for i in range(0, len(labels))], title, len(labels), "v", 0, 8)
        ax.bar(ind + shift, distribution_df[col_y_1], width, color=colors[lang][i], hatch='//', edgecolor="black")
        ax.bar(ind + shift, distribution_df[col_y_2], width, color=colors[lang][i], edgecolor="black")
        shift = shift + width
        i += 1

    plt.grid(axis="y")
    plt.tight_layout()
    legend_1 = ax.legend(legend_texts, bbox_to_anchor=(1, 0.5), loc='upper left', fontsize=8)
    plt.savefig(filepath, bbox_extra_artists=(legend_1,), bbox_inches="tight")
    plt.figure().clear()
    legend_2 = plt.legend(handles=create_white_patches(labels), bbox_to_anchor=(1, 0.5), fontsize=8, loc="upper left",
                          ncol=2)
    export_legend(legend_2, f"plots/lc_legend_{lang}.png")


def ax_brah_0(ax, ind, df_log: pd.DataFrame, df: pd.DataFrame, col_y: str, col_x: str, sign: bool, color):
    """
    Creates horizontal bars for significant columns
    """
    ax.barh((ind * 1.5) - 0.25,
            append_missing_rows(df_log, df[df[f"significance_{col_y}"] == sign], col_x)[col_y],
            color=color, height=0.5)


def ax_brah_1(ax, ind, df_log: pd.DataFrame,df: pd.DataFrame, col_y: str, col_x: str, sign: bool, color):
    """
    Creates horizontal bars for not significant columns.
    """
    ax.barh((ind * 1.5) + 0.25,
            append_missing_rows(df_log, df[df[f"significance_{col_y}"] == sign], col_x)[col_y],
            color=color, hatch="//", edgecolor="black", height=0.5)


def horizontal_bar_effect_plot(df_0_pos: pd.DataFrame, df_0_neg: pd.DataFrame, df_1_pos: pd.DataFrame,
                               df_1_neg: pd.DataFrame,
                               col_y: str, col_x: str, label_texts: list,
                               legend_texts: list,
                               xlim: list, title: str, filepath: str):
    """
    Creates horizontal lower court effect plots.
    Saves Figure as PNG and exports legend.
    """
    colors = {"green": [COLORS["green"], "#25691c"], "purple": [COLORS["light purple"], "#41376D".lower()]}
    fig, ax = plt.subplots(dpi=1200, figsize=(9, 6))
    labels = df_0_pos[col_x].values
    ind = np.arange(len(df_0_pos[col_x].values))

    for tup in [(df_0_pos, False, colors["green"][0]), (df_0_pos, True, colors["green"][1]),
                (df_0_neg, False, colors["purple"][0]), (df_0_neg, True, colors["purple"][1])]:
        ax_brah_0(ax, ind,df_0_pos ,tup[0], col_y, col_x, tup[1], tup[2])

    for tup in [(df_1_pos, False, colors["green"][0]), (df_1_pos, True, colors["green"][1]),
                (df_1_neg, False, colors["purple"][0]), (df_1_neg, True, colors["purple"][1])]:
        ax_brah_1(ax, ind,df_0_pos, tup[0], col_y, col_x, tup[1], tup[2])

    set_texts(ax, label_texts, labels, title, len(labels), "h", 0, 9)

    ax.set_yticks(ind * 1.5)
    ax.set_yticklabels(labels, fontsize=9, rotation=0)
    ax.invert_yaxis()
    plt.xlim(xlim)
    plt.tight_layout()
    plt.grid()
    fig.subplots_adjust(left=0.3)
    plt.savefig(filepath, bbox_inches="tight")
    legend1 = plt.legend(labels=legend_texts, fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.5), ncols=2)
    ax.add_artist(legend1)
    export_legend(legend1, "plots/lc_effect_legend.png")


def bar_ax(ax, ind, errorbars, mean_df, rows, i, width, colors):
    """
    Creates vertical bars for ax and returns created bar.
    """
    if not errorbars.empty:
        bar = ax.bar(ind, mean_df.loc[[rows[i]]].values.flatten().tolist(), width,
                     color=colors[i], yerr=errorbars.loc[[rows[i]]].values,
                     error_kw=dict(lw=0.5, capsize=5, capthick=0.5),
                     ecolor='black')
    else:
        bar = ax.bar(ind, mean_df.loc[[rows[i]]].values.flatten().tolist(), width,
                     color=colors[i])
    return bar


def distribution_plot(N: int, mean_df: pd.DataFrame, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                      ylim: list, errorbars: pd.DataFrame, title: str, filepath: str):
    """
    Creates vertical distribution bar plot.
    Saves Figure as PNG.
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
    set_ax_labels(ax_labels, ylim)
    plt.xticks(ind + width, x_labels)
    plt.legend((bar1, bar2, bar3), legend_texts)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def ann_distribution_plot(mean_df: pd.DataFrame, ax_labels: list, legend_texts: list, ylim: list, title: str,
                          error_bars: pd.DataFrame, mean_lines: dict,
                          filepath: str):
    """
    Creates vertical annotation token distribution bar plot with optional mean ax-lines.
    Saves Figure as PNG.
    """
    plt.subplots(dpi=1200)
    colors = [COLORS["purple"], COLORS["green"], COLORS["blue"]]

    if not error_bars.empty:
        mean_df.plot(kind='bar', yerr=error_bars.T.values, error_kw=dict(lw=0.5, capsize=5, capthick=0.5), color=colors,
                     ecolor='black')
    else:
        mean_df.plot(kind='bar', color=colors)
    plt.rcParams.update({'font.size': 9})
    plt.title(title, fontsize=12)
    plt.legend(legend_texts, ncol=len(legend_texts), loc='best')
    add_mean_lines(mean_lines, legend_texts, colors)
    set_ax_labels(ax_labels, ylim)
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def occ_distribution_plot(N: int, mean_df_list: list, rows: list, ax_labels: list, x_labels, legend_texts: tuple,
                          ylim: list, errorbar_list: list, title: str, filepath: str, legendpath: str):
    """
    Creates vertical occlusion token distribution bar plot.
    Saves Figure as PNG and exports legend.
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
                                                                                                                 capsize=5,
                                                                                                                 capthick=0.5),
                             ecolor='black')
        bar2 = axs[i, j].bar(ind + width, mean_df_list[fig_count].loc[[rows[1]]].values.flatten().tolist(), width,
                             color=colors[1], yerr=errorbar_list[fig_count].loc[[rows[1]]].values, error_kw=dict(lw=0.5,
                                                                                                                 capsize=5,
                                                                                                                 capthick=0.5),
                             ecolor='black')
        bar3 = axs[i, j].bar(ind + width * 2, mean_df_list[fig_count].loc[[rows[2]]].values.flatten().tolist(), width,
                             color=colors[2], yerr=errorbar_list[fig_count].loc[[rows[2]]].values, error_kw=dict(lw=0.5,
                                                                                                                 capsize=5,
                                                                                                                 capthick=0.5),
                             ecolor='black')
        axs[i, j].grid(axis="y")
        axs[i, j].set_ylim(ylim)
        axs[i, j].annotate(fig_count + 1, (2.5, 0.5))
        axs[i, j].set_xticks(ind + width, x_labels, fontsize=6)
        j += 1
        if j == 2:
            i = 1
            j = 0

    flatten_axis(axs)
    fig.text(0.5, 0.9, title, fontsize=12, ha='center')
    fig.text(0.5, 0, ax_labels[0], ha='center')
    fig.text(0, 0.5, ax_labels[1], va='center', rotation='vertical')

    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    legend = plt.legend((bar1, bar2, bar3), legend_texts, bbox_to_anchor=(1, 0.5), ncols=3)
    plt.figure().clear()
    export_legend(legend, legendpath)


def scatter_axis_1(axs, i, j, fig_count, df, marker, cmap, alpha):
    """
    Creates scatter axis for classification plots.
    """
    axs[i, j].scatter(x=df["confidence_scaled"], y=df["norm_explainability_score"],
                      c=df["numeric_label_model"], alpha=alpha, marker=marker,
                      cmap=cmap, s=20)
    axs[i, j].grid(True)
    axs[i, j].set_ylim([-0.3, 0.3])
    axs[i, j].set_xlim([-0.55, 0.55])
    axs[i, j].annotate(fig_count, (0.5, -0.29999))
    axs[i, j].axhline(0, color='grey')
    axs[i, j].axvline(color='grey')


def scatter_axis_2(axs, i, j, fig_count, df, marker, cmap, alpha, mode):
    """
    Creates scatter axis for trend plots.
    """
    axs[i, j].scatter(x=df["confidence_scaled"], y=df["norm_explainability_score"],
                      c=df["numeric_label_model"], alpha=alpha, marker=marker,
                      cmap=cmap, s=20)
    axs[i, j].grid(True)
    if mode == "o":
        axs[i, j].set_ylim([0, 0.3])
        axs[i, j].set_xlim([0, 0.55])
        axs[i, j].annotate(fig_count, (0.5, 0))
    if mode == "s":
        axs[i, j].set_ylim([-0.3, 0])
        axs[i, j].set_xlim([-0.55, 0])
        axs[i, j].annotate(fig_count, (-0.05, -0.3))
    axs[i, j].axhline(0, color='grey')
    axs[i, j].axvline(color='grey')


def set_scatter_plot_layout():
    """
    Sets layout for scatter plots.
    """
    plt.figure().clear()
    fig = plt.figure(dpi=1200, figsize=(6, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
    return fig, gs.subplots(sharex='col', sharey='row')


def save_scatter_plot(fig, axs, title: str, filepath: str):
    """
    Prepares scatter plots for exports and saves figure as PNG.
    """
    flatten_axis(axs)
    fig.text(0.5, 0.9, title, fontsize=12, ha='center')
    fig.text(0.5, 0, "Scaled Confidence Direction", ha='center')
    fig.text(0, 0.5, "Normalized Explainability Score", va='center', rotation='vertical')
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")


def class_scatter_plot(df_list, title, filepath: str, colors: list):
    """
    Creates scatter plot for classifications plots.
    """
    fig, axs = set_scatter_plot_layout()
    i, j, fig_count = 0, 0, 1

    for df_1, df_2 in zip(df_list[0], df_list[1]):
        df_oj_0_s, df_oj_0_ns, df_oj_1_s, df_oj_1_ns, df_sj_0_s, df_sj_0_ns, df_sj_1_s, df_sj_1_ns = get_df_slices(
            [df_1, df_2])
        tuples = [(df_oj_0_s, "o", colors[2]), (df_oj_0_ns, "o", colors[0]), (df_oj_1_s, "+", colors[2]),
                  (df_oj_1_ns, "+", colors[3]), (df_sj_0_s, "o", colors[0]), (df_sj_0_ns, "o", colors[3]),
                  (df_sj_1_s, "+", colors[3]), (df_sj_1_ns, "+", colors[3])]
        for tup in tuples:
            scatter_axis_1(axs, i, j, fig_count, tup[0], tup[1], matplotlib.colors.ListedColormap(tup[2]), 0.3)
        j += 1
        fig_count += 1
        if j == 2:
            i = 1
            j = 0

    save_scatter_plot(fig, axs, title, filepath)


def trend_scatter_plot(df_list, title, filepath: str, colors: list, mode):
    """
    Creates scatter plot for trend plots.
    """
    fig, axs = set_scatter_plot_layout()

    i, j, fig_count = 0, 0, 1
    for df in df_list:
        df_0_s, df_0_ns, df_1_s, df_1_ns = get_df_slices([df])
        for tup in [(df_0_s, "o", colors[1]), (df_0_ns, "o", colors[0]), (df_1_s, "+", colors[1]),
                    (df_1_ns, "+", colors[0])]:
            scatter_axis_2(axs, i, j, fig_count, tup[0], tup[1], matplotlib.colors.ListedColormap(tup[2]), 0.3,
                           mode)

        j += 1
        fig_count += 1
        if j == 2:
            i = 1
            j = 0

    save_scatter_plot(fig, axs, title, filepath)


def set_violin_colors(violin, colors):
    """
    Sets non default colors for violin plots.
    """
    j = 0
    for k, pc in enumerate(violin["bodies"], 1):
        pc.set_facecolor(colors[j])
        pc.set_edgecolor(colors[j])
        j += 1
        if j == 3:
            j = 0
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        vp = violin[partname]
        vp.set_edgecolor("black")
        if partname == 'cmeans':
            vp.set_edgecolor("red")
        if partname == 'cmedians':
            vp.set_edgecolor("orange")
        vp.set_linewidth(.5)


def violin_axs_1(axs, i, df, cols, pos, colors, annotation):
    """
    Creates violins for layout 1.
    """
    violin = axs[i].violinplot(np.array(df[cols].dropna().values, dtype=float), positions=pos, showmeans=True,
                               showmedians=True)
    set_violin_colors(violin, colors)
    axs[i].set_xticks(pos, [label.replace(" ", "\n") for label in LABELS], fontsize=6)
    axs[i].yaxis.grid(True)
    axs[i].annotate(annotation, (-0.1, 0), fontsize=7)


def violin_axs_2(axs, i, j, df, cols, pos, colors, annotation):
    """
    Creates violins for layout 2.
    """
    violin = axs[i, j].violinplot(np.array(df[cols].dropna().values, dtype=float), positions=pos, showmeans=True,
                                  showmedians=True)
    set_violin_colors(violin, colors)
    axs[i, j].set_xticks(pos, [label.replace(" ", "\n") for label in LABELS[1:]], fontsize=6)
    axs[i, j].yaxis.grid(True)
    axs[i, j].annotate(annotation, (-0.1, 0), fontsize=7)


def set_violin_layout(fig, x, y, df_list, cols_score, colors):
    """
    Sets layout for violin plots and creates violins.
    """
    gs = fig.add_gridspec(x, y, hspace=0.1, wspace=0.1)
    axs = gs.subplots(sharex='col', sharey='row')

    i = 0
    if x == 1:
        ind = np.arange(3)
        for df in df_list:
            violin_axs_1(axs, i, df, cols_score[0], ind, colors[0], i + 1)
            violin_axs_1(axs, i, df, cols_score[1], ind, colors[1], i + 1)
            i += 1
        fig.text(0.5, -0.15, "Explainability Label", ha='center')
        fig.text(0, 0.5, "IAA Score", va='center', rotation='vertical')
    if x == 2:
        ind = np.arange(2)
        j = 0
        fig_count = 1
        for df in df_list:
            violin_axs_2(axs, i, j, df.drop("index", axis=1).drop_duplicates().reset_index(), cols_score[0], ind,
                         colors[0], fig_count)
            j += 1
            fig_count += 1
            if j == 2:
                i += 1
                j = 0
        fig.text(0.5, 0, "Explainability Label", ha='center')
        fig.text(0, 0.5, "IAA Score", va='center', rotation='vertical')

    flatten_axis(axs)


def violin_plot(x: int, y: int, df_list: list, iaa_list: list, cols_score: list, legend_texts: list, title: str,
                filepath: str, legendpath: str, annotation: str, colors: list):
    """
    Creates Violin plot for each annotation combination, explainability label and two IAA metrics.
    Saves figure and legend as PNG.
    """
    fig = plt.figure(dpi=1200, figsize=(y * 2, x * 2))
    if len(df_list) != 0:
        set_violin_layout(fig, x, y, df_list, cols_score, colors)

        fig.text(0.5, 0.9, title, fontsize=12, ha='center')

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.figure().clear()
        patches = []
        i = 0
        for color_list in colors:
            patches.append([Patch(color=color, label=legend_texts[i]) for color in color_list])
            i += 1
        plt.gca()
        legend = plt.legend(handles=patches, labels=legend_texts, ncol=2, fontsize=12,
                            handler_map={list: HandlerTuple(None)})
        export_legend(legend, legendpath.format(1))
        patches = []
        for i in range(0, 3):
            patches.append(Patch(color="white", label=f"{i + 1} : {annotation.format(iaa_list[i])}"))
        legend = plt.legend(handles=patches, fontsize=12)
        export_legend(legend, legendpath.format(2))


def explanation_histogram(lang, hist_df_1: pd.DataFrame, hist_df_2: pd.DataFrame, filepath: str):
    """
    Creates explanations distribution histogram with normal distribution line.
    Saves Figure as PNG.
    """
    colors = [COLORS["red"], COLORS["purple"], COLORS["blue"], COLORS["green"]]
    fig, axs = plt.subplots(dpi=1200, nrows=2, ncols=1)
    data = [hist_df_1[[f"score_{i}" for i in range(0, 4)]], hist_df_2[[f"score_{i}" for i in range(0, 4)]]]
    titles = ["Model", "Human"]
    fig.suptitle(f"Histogram of the Explanation Accuracy Score {lang.upper()}")
    for i in range(0, 2):
        mu, std = norm.fit(data[i])
        x = np.linspace(0, 2, 100)
        p = norm.pdf(x, mu, std)
        ax = axs[i].hist(data[i], bins=25, histtype='bar', color=colors)
        axs[i].plot(x, p, 'k', linewidth=1)
        axs[i].grid(axis="y")
        axs[i].set_xlim([0, 2])

        legend = axs[i].legend(ax, labels=["Normal Distribution"] + [f"{i} Sentence Explanation" for i in [2, 4, 6, 8]],
                               title=f"{titles[i]}-Near", fontsize=6, loc="upper left")
        plt.setp(legend.get_title(), fontsize='xx-small')

    plt.savefig(filepath, bbox_inches="tight")


def multilingual_ann_plot(df_list: list):
    """
    Prepares and creates annotation token distribution multilingual plots.
    """
    df = df_list[0].merge(df_list[1], on="index", how="inner", suffixes=(f'_{LANGUAGES[0]}', f'_{LANGUAGES[1]}'))
    df = df.merge(df_list[2], on="index", how="inner").rename(columns={'mean_token': f'mean_token_{LANGUAGES[2]}',
                                                                       "error": f"error_{LANGUAGES[2]}"})
    df.drop([f"label_{LANGUAGES[0]}", f"label_{LANGUAGES[1]}", "index"], axis=1, inplace=True)
    errorbars = df.set_index("label")[[f"error_{l}" for l in LANGUAGES]].T
    errorbars.index = [f"mean_token_{l}" for l in LANGUAGES]
    distribution_plot(len(LABELS_OCCLUSION), df.set_index("label")[[f"mean_token_{l}" for l in LANGUAGES]].T,
                      rows=[f"mean_token_{lang}" for lang in LANGUAGES],
                      ax_labels=["Explainability Labels", "Number of Tokens"],
                      x_labels=LABELS_OCCLUSION,
                      legend_texts=tuple([f"Mean Number of Tokens {i.upper()}" for i in LANGUAGES]),
                      ylim=[],
                      errorbars=errorbars,
                      title="Token Distribution of Annotation Labels in Gold Standard Dataset.",
                      filepath=f"plots/ann_mean_tokens_exp_labels_gold.png")


def create_lc_flipped_plot(lang: str, lc_df: pd.DataFrame, cols: list, label_texts: list,
                           legend_texts: list, title: str, filepath: str):
    """
    Prepares and creates lower court flipped distribution plots.
    Creates and saves lower court flipped distribution table.
    """
    flipped_df_list = []
    for df in [lc_df[lc_df["prediction"] == p] for p in [0, 1]]:
        flipped_df_list.append(preprocessing.group_by_flipped(df, cols[0]))
    lc_flipped_bar_plot(lang, flipped_df_list, col_x=cols[0], col_y_1=cols[1],
                        col_y_2=cols[2],
                        label_texts=label_texts, legend_texts=legend_texts,
                        title=title, filepath=filepath.format(1))
    # Lower Court flipped table
    flipped_dict = {}
    i = 0
    for df in flipped_df_list:
        df["has_flipped_abs"] = df["has_flipped"] * df["id"]
        flipped_dict[i] = df.to_dict()
        i += 1
    preprocessing.write_json(f"{lang}/quantitative/lc_flipped.json", flipped_dict)


def create_lc_effect_plot(lc_df_dict: dict, cols: list,
                          label_texts: list, legend_texts: list, xlim: list, title: str, filepath: str):
    """
    Prepares and creates lower court effect plots for each language.
    """

    for key in lc_df_dict:
        if key in LANGUAGES:
            mean_pos_df_list_0, mean_neg_df_list_0, mean_pos_df_list_1, mean_neg_df_list_1 = [], [], [], []
            for occlusion_df in lc_df_dict[key]:
                occlusion_df_0 = occlusion_df[occlusion_df["prediction"] == 0]
                occlusion_df_1 = occlusion_df[occlusion_df["prediction"] == 1]
                for tup in [(mean_pos_df_list_0, mean_neg_df_list_0, occlusion_df_0),
                            (mean_pos_df_list_1, mean_neg_df_list_1, occlusion_df_1)]:
                    mean_pos_df = preprocessing.get_one_sided_effect_df(tup[2][tup[2]["confidence_direction"] > 0],
                                                                        lc_df_dict[f"{key}_mu"], cols[0], "pos")
                    mean_neg_df = preprocessing.get_one_sided_effect_df(tup[2][tup[2]["confidence_direction"] < 0],
                                                                        lc_df_dict[f"{key}_mu"], cols[0], "neg")
                    tup[0].append(mean_pos_df)
                    tup[1].append(mean_neg_df)

            horizontal_bar_effect_plot(pd.concat(mean_pos_df_list_0), pd.concat(mean_neg_df_list_0),
                                       pd.concat(mean_pos_df_list_1), pd.concat(mean_neg_df_list_1),
                                       col_y=cols[1],
                                       col_x=cols[0],
                                       label_texts=label_texts,
                                       legend_texts=legend_texts,
                                       xlim=xlim,
                                       title=title.format(key.upper()),
                                       filepath=filepath.format(key))


def create_multilingual_occ_plot(df_dict: dict):
    """
    Prepares and creates language occlusion Dataframes for plots.
    """
    mean_df_list = []
    error_df_list = []
    for key in df_dict:
        df = pd.concat(df_dict[key]).set_index(pd.Index(LANGUAGES))
        preprocessing.write_csv(f"tables/occ_mean_chunk_length_{key}.csv", df)
        mean_df_list.append(df[LABELS_OCCLUSION[1:]])
        error_df_list.append(df[[f"{label}_error" for label in LABELS_OCCLUSION[1:]]])
    occ_distribution_plot(len(LABELS_OCCLUSION[1:]), mean_df_list,
                          rows=LANGUAGES,
                          ax_labels=["Explainability Labels", "Number of Tokens"],
                          x_labels=[label.replace(' ', '\n') for label in LABELS_OCCLUSION[1:]],
                          legend_texts=tuple([f"Mean Chunk Length {i.upper()}" for i in LANGUAGES]),
                          ylim=[0, 120],
                          errorbar_list=error_df_list,
                          title=f"Chunk Length per Explainability Label in Occlusion Experiment",
                          filepath=f"plots/occ_mean_chunk_length.png",
                          legendpath=f"plots/occ_mean_chunk_length_legend.png")


def create_occ_scatter_plot(occlusion_df_dict):
    """
    Prepares and creates four different scatter plots for each language (Trend plots for both explainability labels,
    correct and incorrect classification plots).
    Saves legends as PNG.
    """
    for l in LANGUAGES:
        correct_df_list = []
        incorrect_df_list = []
        for key in occlusion_df_dict[l]:
            if key.startswith("c", 0):
                correct_df_list.append(occlusion_df_dict[l][key])  # Appends all correct o_j and s_j
            if key.startswith("f", 0):
                incorrect_df_list.append(occlusion_df_dict[l][key])
            if key.startswith("s", 0):
                trend_scatter_plot(occlusion_df_dict[l][key],
                                   f"Trend Supports Judgement (effect on confidence) {l.upper()}",
                                   filepath=f"plots/occ_{key}_effect_{l}.png",
                                   colors=[[COLORS['dark blue'], COLORS["yellow"], COLORS["red"]],
                                           ["#ff8800", COLORS["light purple"], COLORS["dark green"]]],
                                   mode="s")
                scatter_legend(colors=[COLORS['dark blue'], "#ff8800"],
                               labels=[f"{LABELS_OCCLUSION[1]} significant", LABELS_OCCLUSION[1]],
                               filename=f"plots/occ_{key}_effect_legend.png")
            if key.startswith("o", 0):
                trend_scatter_plot(occlusion_df_dict[l][key],
                                   f"Trend Opposes Judgement (effect on confidence) {l.upper()}",
                                   filepath=f"plots/occ_{key}_effect_{l}.png",
                                   colors=[[COLORS['dark blue'], COLORS["yellow"], COLORS["red"]],
                                           ["#ff8800", COLORS["light purple"], COLORS["dark green"]]],
                                   mode="o")
                scatter_legend(colors=[COLORS['red'], COLORS["dark green"]],
                               labels=[f"{LABELS_OCCLUSION[2]} significant", LABELS_OCCLUSION[2]],
                               filename=f"plots/occ_{key}_effect_legend.png")

        # Correct classification plots
        labels = [f"{label} significant" for label in [LABELS_OCCLUSION[2], LABELS_OCCLUSION[1]]]
        class_scatter_plot(correct_df_list,
                           f"Impact of Occluded Sentences (Correctly Classified) {l.upper()}",
                           filepath=f"plots/occ_correct_classification_{l}.png",
                           colors=[[COLORS["red"]], [COLORS['dark blue']], [COLORS["dark green"]], ["#ff8800"]])
        scatter_legend([COLORS['red'], COLORS['dark blue'], COLORS["dark green"], "#ff8800"],
                       [LABELS_OCCLUSION[2], LABELS_OCCLUSION[1]] + labels,
                       filename=f"plots/occ_correct_classification_legend.png")
        # False classification plots
        labels = [f"{label} significant" for label in
                  [LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]]]

        class_scatter_plot(incorrect_df_list,
                           f"Impact of Occluded Sentences (Incorrectly Classified) {l.upper()}",
                           filepath=f"plots/occ_false_classification_{l}.png",
                           colors=[[COLORS['dark blue'], COLORS['yellow']],
                                   [COLORS['yellow'], COLORS["red"]],
                                   ["#ff8800", COLORS["light purple"]],
                                   [COLORS["light purple"], COLORS["dark green"]]
                                   ])
        scatter_legend([COLORS["yellow"], COLORS['dark blue'], COLORS['red'], COLORS["light purple"]],
                       [LABELS_OCCLUSION[3], LABELS_OCCLUSION[1], LABELS_OCCLUSION[2]] + labels,
                       filename=f"plots/occ_false_classification_legend.png")
