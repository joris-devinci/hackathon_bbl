# measurement lvl Single
# | nominal/binary | simple bar-chart            | simple_bar_chart(df, col)
# | ordinal/binary | stacked bar-chart           | stacked_bar_chart(df, col)      
# | scale          | histogram                   | histogram(df, col)        
# | scale          | box-plot                    | box_plot(df, col)     


# measurement lvl Double
# binary == nominal
# 
# | nominal + nominal   | clustered bar-chart                     | clustered_bar_chart(df, [col, col])
# | ordinal + nominal   | multiple stacked bar-chart              | multiple_stacked_bar_chart(df, [col, col])       
# | ordinal + ordinal   | heat-map                                | heat_map(df, [col, col])
# | scale + nominal     | split histogram or split box-plot       | split_histogram(df, [col, col])                     
# | scale + ordinal     | split histogram or split box-plot       | split_box_plot(df, [col, col])              
# | scale + scale       | scatterplot                             | scatterplot(df, [col, col])

# Imports
import pandas as pd
import matplotlib.pyplot as plt
import stikpetP as ps
import math
from IPython.display import display
# ---------------

def simple_bar_chart(df:pd.DataFrame, selected_col:str, x:int=5, y:int=3) -> None:
    # Nominal/Binary
    df[selected_col].value_counts().plot.bar(ylabel='Value', figsize=(x, y));
    plt.show()

def stacked_bar_chart(df:pd.DataFrame, selected_col:str, x:int=5, y:int=3) -> None:
    # Ordinal/Binary
    ps.vi_bar_stacked_single(df[selected_col])
    plt.show()

def histogram(df:pd.DataFrame, selected_col:str, x:int=5, y:int=3) -> None:
    # Scale
    df[selected_col].plot.hist(edgecolor='black', figsize=(x, y));
    plt.xlabel(selected_col)
    plt.ylabel("count")
    plt.show()

def box_plot(df:pd.DataFrame, selected_col:str, x:int=5, y:int=3) -> None:
    # Scale
    df[selected_col].plot.box(vert=False, figsize=(x, y));
    plt.show()

def clustered_bar_chart(df:pd.DataFrame, selected_cols:list[str], x:int=8, y:int=3) -> None:
    # 1: Nominal, 2: Nominal
    cross_table = pd.crosstab(df[selected_cols[0]], df[selected_cols[1]], normalize='columns')*100
    ax = cross_table.plot(kind="bar", figsize=(x, y))

    ax.legend(
        title="teach_motivate",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

def multiple_stacked_bar_chart(df:pd.DataFrame, selected_cols:list[str], x:int=8, y:int=3) -> None:
    # 1: Ordinal/Nominal, 2: Ordinal/Nominal
    cross_table = pd.crosstab(df[selected_cols[0]], df[selected_cols[1]], normalize='columns')*100
    ax = cross_table.T.plot(kind="barh", figsize=(x, y))

    ax.legend(
        title="teach_motivate",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()
    plt.show()

def heat_map(df:pd.DataFrame, selected_cols:list[str]) -> None:
    cross_tab = pd.crosstab(survey_adj_df[ordinal_var1], survey_adj_df[ordinal_var2])
    heat_map = cross_tab.style.background_gradient(cmap='Blues')
    display(heat_map)

def split_histogram(df:pd.DataFrame, selected_cols:list[str], x:int=6, y:int=6) -> None:
    # 1: Scale, 2: Nominal
    df.plot.hist(column=[selected_cols[0]], edgecolor='black', by=[selected_cols[1]], sharex=True, figsize=(x, y))
    plt.show()

def split_box_plot(df:pd.DataFrame, selected_cols:list[str], x:int=6, y:int=6) -> None:
    # 1: Scale, 2: Nominal
    df.plot.box(column=[selected_cols[0]], by=[selected_cols[1]], sharex=True, vert=False, figsize=(x, y))
    plt.show()

def scatterplot(df:pd.DataFrame, selected_cols:list[str], x:int=6, y:int=5) -> None:
    # 1: Scale, 2: Scale
    plt.figure(figsize=(x, y))
    plt.scatter(df[selected_cols[0]], df[selected_cols[1]])
    plt.xlabel(selected_cols[0])
    plt.ylabel(selected_cols[1])
    plt.show()