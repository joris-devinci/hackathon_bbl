import math
import matplotlib.pyplot as plt
import pandas as pd
import stikpetP as ps
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import seaborn as sns
from scipy.stats.mstats import zscore
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def plot_bar_charts(dataframe:pd.DataFrame, plots_per_row:int=4, max_label_length:int=20, top_n:int=20) -> None:

    df = dataframe.select_dtypes(include=['category','object','bool'])

    total_plots = len(df.columns)
    if total_plots == 0:
        return
        
    n_rows = math.ceil(total_plots / plots_per_row)
    
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 4, n_rows * 2))
    axes = axes.flatten()
    
    for i, col in enumerate(df):

        if top_n is not None:
            value_counts = df[col].value_counts(sort=False).head(top_n)
        else:
            value_counts = df[col].value_counts(sort=False)
        
        value_counts.plot(kind='barh', ax=axes[i])
        axes[i].set_title(col)
        
        new_labels = [label[:max_label_length] + '…' if len(label) > max_label_length else label
                      for label in value_counts.index.astype(str)]
        axes[i].set_yticklabels(new_labels)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return



def plot_box_plots(dataframe:pd.DataFrame, plots_per_row:int=4, max_label_length:int=20) -> None:
    
    df = dataframe.select_dtypes(include=['number'], exclude=['bool'])

    total_plots = len(df.columns)
    if total_plots == 0:
        return
        
    n_rows = math.ceil(total_plots / plots_per_row)
    
    fig, axes = plt.subplots(n_rows, plots_per_row, figsize=(plots_per_row * 4, n_rows * 2))
    axes = axes.flatten()

    for i, col in enumerate(df):
        
        df.boxplot(col, grid=False, vert=False, ax=axes[i])
        axes[i].set_title(col)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    return



def remove_outliers(dataframe: pd.DataFrame, outlier_factor:float=1.5) -> pd.DataFrame:

    new_df = dataframe.copy()
    num_df = dataframe.select_dtypes(include=['number'], exclude=['bool'])

    quartiles = num_df.quantile(q=[0.25, 0.75], axis=0).transpose()
        
    IQR = quartiles[0.75] - quartiles[0.25]
    
    low_threshold = quartiles[0.25] - outlier_factor * IQR
    high_threshold = quartiles[0.75] + outlier_factor * IQR
    
    for col in num_df:
        
        new_df.loc[new_df[col] < low_threshold[col], col] = float('nan')
        new_df.loc[new_df[col] > high_threshold[col], col] = float('nan')

    return new_df



def moct(
    dataframe:pd.DataFrame, 
    binary_list:list[str]|None=None, 
    nominal_list:list[str]|None=None,
    ordinal_list:list[str]|None=None,
    scaled_list:list[str]|None=None
) -> pd.DataFrame: # Cols(Field, Measure, Value)

    binary_list = binary_list or []
    nominal_list = nominal_list or []
    ordinal_list = ordinal_list or []
    scaled_list = scaled_list or []
    
    fields = []
    measures = []
    values = []

    neutral_list = ['Not at all', 'not so much', 'neutral', 'somewhat', 'Very much']
    agree_list = ["Fully Disagree", "Disagree", "Neither disagree nor agree", "Agree", "Fully agree"]

    def append_values(col:str, measure:str, value) -> None:
        nonlocal fields, measures, values
        fields.append(col)
        measures.append(measure)
        values.append(value)

    for col in dataframe:
        
        if col in binary_list + nominal_list: # Mode
            mode_series = dataframe[col].mode()
            if not mode_series.empty:
                append_values(col, "mode", mode_series.iloc[0])
    
        elif col in ordinal_list: # Mode, Medium
            mode_series = dataframe[col].mode()
            if not mode_series.empty:
                append_values(col, "mode", mode_series.iloc[0])
            
            unique_set = set(dataframe[col].dropna().unique())
            
            if unique_set.issubset(set(agree_list)):
                median = ps.me_median(dataframe[col], levels={"Fully Disagree":1, "Disagree":2, "Neither disagree nor agree":3, "Agree":4, "Fully agree":5})[1]
                # mean = ps.me_mean(dataframe[col], levels={"Fully Disagree":1, "Disagree":2, "Neither disagree nor agree":3, "Agree":4, "Fully agree":5})
                append_values(col, "median", median)
                # append_values(col, "mean", mean)
            elif unique_set.issubset(set(neutral_list)):
                median = ps.me_median(dataframe[col], levels={"Not at all":1, "not so much":2, "neutral":3, "somewhat":4, "Very much":5})[1]
                append_values(col, "median", median)
    
        elif col in scaled_list: # Mode, Median, Mean
            mode_series  = dataframe[col].mode()
            median = dataframe[col].median()
            mean = dataframe[col].mean()
            if not mode_series.empty:
                append_values(col, "mode", mode_series.iloc[0])
            append_values(col, "median", median)
            append_values(col, "mean", mean)

    result = pd.DataFrame({
        "field": fields,
        "measure": measures,
        "value": values
    })
    return result



def check_suitable_regression(dataframe:pd.DataFrame, min_var:int=10, n_col_crit:int=5) -> list[str]:
    '''
    Check Suitable Regression
    -------------------------
    This function checks if a given dataframe might be suitable for regression analysis. With 'suitable' it is meant that there are at least a given number (n_col_crit) columns that have numeric values, and in each there are at least a given number (min_var) of different values.

    Parameters
    ----------
    dataframe : pandas dataframe
        the dataframe to be checked
    min_var : int
        the minimum number of unique scores in each field
    n_col_crit : int
        the minimum number of numeric columns that meet the min_var criteria

    Return
    ------
    A printed statement if the dataframe is suitable and
    the suitable columns as a list.
    
    '''
    # select only the numeric fields
    df_num = dataframe.select_dtypes(include='number', exclude='bool')

    # check for each if it has at least min_var different scores
    suitable_columns = []
    for col in df_num.columns:
        if len(set(df_num[col])) >= min_var:
            suitable_columns.append(col)

    if len(suitable_columns) >= n_col_crit:
        print('enough suitable columns')
        return suitable_columns
    else:
        print('insufficient suitable columns, need different dataset')
        return []


def bivariate_regression(train_df:pd.DataFrame, test_df:pd.DataFrame, X_label:str, Y_label:str, visuals:bool=True, remove_outliers_mode:str='train') -> dict[str, float]:
    
    df_train_sub = train_df[[X_label, Y_label]].dropna()
    df_test_sub = test_df[[X_label, Y_label]].dropna()
    
    if remove_outliers_mode in ["train", "both"]:
        df_train_sub = remove_outliers(df_train_sub).dropna()
    if remove_outliers_mode in ["test", "both"]:
        df_test_sub = remove_outliers(df_test_sub).dropna()
    
    if df_train_sub.empty or df_test_sub.empty:
        print("Not enough data after cleaning to run regression.")
        return

    model = LinearRegression().fit(df_train_sub[[X_label]], df_train_sub[Y_label])
    b0 = model.intercept_
    b1 = model.coef_[0]
    
    y_hat_train = model.predict(df_train_sub[[X_label]])
    y_hat_test = model.predict(df_test_sub[[X_label]])
    
    r2_train = metrics.r2_score(df_train_sub[[Y_label]], y_hat_train) 
    r2_test = metrics.r2_score(df_test_sub[[Y_label]], y_hat_test) 
    
    #show visuals if requested
    if visuals:
        # print the coefficient and intercept
        print("gradient (b1): ", b1)
        print("constant (b0): ", b0)

        # create the plots
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))

        # Training data plot
        axes[0].scatter(df_train_sub[X_label], df_train_sub[Y_label], color='blue')
        axes[0].plot(df_train_sub[X_label], y_hat_train, color='red')
        axes[0].set_xlabel(X_label)
        axes[0].set_ylabel(Y_label)
        axes[0].set_title(f"Training Data (r² = {r2_train:.2f})")

        # Testing data plot
        axes[1].scatter(df_test_sub[X_label], df_test_sub[Y_label], color='blue')
        axes[1].plot(df_test_sub[X_label], y_hat_test, color='red')
        axes[1].set_xlabel(X_label)
        axes[1].set_ylabel(Y_label)
        axes[1].set_title(f"Testing Data (r² = {r2_test:.2f})")

        # Adjust layout to avoid overlap
        plt.tight_layout()
        plt.show()
    
    # dictionary with intercept, coefficient, and determination coefficient
    results = {'b0':b0, 'b1':b1, 'r2_train':r2_train, 'r2_test':r2_test}
            
    return results

def check_corr(train_df:pd.DataFrame, focus_col:str, selected_cols:list[str]=None):

    if not selected_cols:
        selected_cols = check_suitable_regression(train_df)
        if not selected_cols:
            print("No colomns selected and found")
        
    correlation_matrix = train_df[selected_cols].corr()
    
    determination_coeffs = correlation_matrix[focus_col]**2
    determination_coeffs = determination_coeffs.sort_values(ascending=False)

    determination_df = determination_coeffs.to_frame(name='R²')
    
    plt.figure(figsize=(3, 10))
    sns.heatmap(determination_df, annot=True, cmap='coolwarm', fmt=".2f", cbar=True, linewidths=1, linecolor='black')
    plt.title("Determination Coeff. Heatmap for Overall")
    plt.show()


def multivariate_regression(df_train:pd.DataFrame, df_test:pd.DataFrame=None, X_labels:list[str]=None, y_label:str=None, delete_outliers:bool=True) -> tuple:

    if not y_label:
        print("Your missing y label")
    if not X_labels:
        X_labels = df_train.columns.drop(y_label).tolist()
    
    # DATA PREPARE
    if df_test is not None:
        df_test_sub = df_test[X_labels + [y_label]].dropna()

    df_train_sub = df_train[X_labels + [y_label]].dropna()    
    if delete_outliers:
        df_train_sub = remove_outliers(df_train_sub).dropna()

    # LINIEARREGRESSION
    model = LinearRegression().fit(df_train_sub[X_labels], df_train_sub[y_label])
    bx = model.coef_
    b0 = model.intercept_
    
    y_hat_train = model.predict(df_train_sub[X_labels])
    r2_train = metrics.r2_score(df_train_sub[y_label], y_hat_train)
    
    if df_test is not None:
        y_hat_test = model.predict(df_test_sub[X_labels])
        r2_test = metrics.r2_score(df_test_sub[y_label], y_hat_test)

    # ADJUST R2
    n = len(df_train_sub)
    p = len(X_labels)
    adj_r2 = 1 - (1 - r2_train)*(n - 1)/(n - p - 1)
    
    # VIFS
    df_train_const = sm.add_constant(df_train_sub[X_labels])
    vifs = [variance_inflation_factor(df_train_const, i) for i in range(df_train_const.shape[1])]

    # STANDARDIZED COEFFICIENTS
    z_data = zscore(df_train_sub)
    z_df = pd.DataFrame(z_data, columns=df_train_sub.columns, index=df_train_sub.index)
    model = LinearRegression().fit(z_df[X_labels], z_df[y_label])
    bx_z = model.coef_
    b0_z = model.intercept_

    # PART 5: COLLECTING THE RESULTS
    #------------------------------
    # dataframe with labels as index and coefficient values
    coeff_df = pd.DataFrame(list(bx), index = X_labels, columns=['Coefficient'])
    # add also the standardized coefficients, VIFs and the constant
    coeff_df['std. coeff.'] = bx_z
    coeff_df['VIF'] = vifs[1:len(vifs)]
    coeff_df.loc['constant'] = [b0, 'n.a.', vifs[0]]
    #the other values as a dictionary
    results = {'adj_r2':adj_r2, 'r2_train':r2_train, 'r2_test':'n.a.'}
    if df_test is not None:
        results['r2_test'] = r2_test
    
    return coeff_df, results, model


def cl_plot_cm(
    actual_data:pd.Series, 
    predicted_data:pd.Series, 
    labels:list[str]=None, 
    title:str="No title given"
) -> None:
    
    if labels is None:
        labels = actual_data.cat.categories
    k = len(labels)
    
    # create the confusion matrix with sklearn make sure to set labels=labels
    cm = confusion_matrix(actual_data, predicted_data, labels=labels)
    
    # Get the current figure *if it exists* — without creating a new one
    fig = plt._pylab_helpers.Gcf.get_active()
    new_fig = fig is None  # True if no active figure
    
    # Only make a new figure if none exists
    if new_fig:  # if no active subplot
        plt.figure(figsize=(3, 3))
    
    # create the seaborn heatmap of the confusion matrix    
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
    
    # set the x and y labels
    plt.xlabel("predicted value")
    plt.ylabel("actual value")
    
    # set the x and y ticks
    plt.xticks(np.arange(k)+1/k, labels)
    plt.yticks(np.arange(k)+1/k, labels)
    
    # add a title, but only if provided
    plt.title(title)
    
    # Only show the plot if it's a standalone figure
    if new_fig:
        plt.tight_layout()
        plt.show()




def cl_dummies(
    train_data:pd.Series, 
    test_data:pd.Series=None,
    y_label:str=None,
    show_cm:bool=True
) -> pd.DataFrame:

    strats=['most_frequent', 'uniform', 'stratified']
    acs = []
    kappas = []
    
    train_data = train_data[y_label]
    
    k = len(train_data)
    n_classes = train_data.nunique()
    y_cats = train_data.cat.categories
    
    if show_cm:
        plt.figure(figsize=(3*4, 4))

    if test_data is None:
        results_data = train_data
    else:
        results_data = test_data[y_label]

    for ind, s in enumerate(strats):

        model = DummyClassifier(strategy=s)
        model_fit = model.fit(train_data, train_data)
        y_hat = model.predict(results_data)
        
        ac = metrics.accuracy_score(results_data, y_hat)
        kappa = metrics.cohen_kappa_score(results_data, y_hat)
        
        acs.append(ac)
        kappas.append(kappa)
        
        # add visual of confusion matrix
        if show_cm:
            plt.subplot(1, 3, ind+1)
            cl_plot_cm(
                actual_data=results_data,
                predicted_data=y_hat,
                labels=y_cats,
                title=f"{s}\n acc. sc. = {round(ac, 4)}"
            )
            
    if show_cm:
        plt.tight_layout()
        plt.show()

    # return the accuracy scores and kappas as a dataframe with the model as index
    return pd.DataFrame({'accuracy':acs, 'Cohen k':kappas}, index=strats)




def cl_cnb(
    train_df:pd.DataFrame,
    test_df:pd.DataFrame=None, 
    X_labels:list[str]=[], 
    y_label:str='', 
    show_cm:bool=True
) -> tuple:
    testing = True if test_df is not None else False
    
    train_sub = train_df[X_labels + [y_label]].dropna()
    if testing:
        test_sub = test_df[X_labels + [y_label]].dropna()
    
    coding = {} 
    for col in train_sub:
        train_sub[col] = pd.Categorical(train_sub[col])
        cats = train_sub[col].cat.categories
        coding[col] = cats
        if testing:
            test_sub[col] = pd.Categorical(test_sub[col], categories=cats)
    
    train_copy = train_sub.copy()
    if testing:
        test_copy = test_sub.copy()
       
    for col in X_labels:
        train_copy[col] = train_copy[col].cat.codes
        if testing:
            test_copy[col] = test_copy[col].cat.codes

    X_train = train_copy[X_labels]
    y_train = train_copy[y_label]
    
    cnb = CategoricalNB().fit(X_train, y_train)
    y_hat_train = cnb.predict(X_train)
    
    acc_score_train = metrics.accuracy_score(y_train, y_hat_train)
    print(f"accuracy score = {acc_score_train}")
    cohen_kappa_train = metrics.cohen_kappa_score(y_train, y_hat_train)
    print(f"cohen kappa score = {cohen_kappa_train}")
    
    cm = confusion_matrix(y_train, y_hat_train, labels=coding[y_label])
    
    if testing:
        X_test = test_copy[X_labels]
        y_test = test_copy[y_label]

        y_hat_test = cnb.predict(X_test)
        
        acc_score_test = metrics.accuracy_score(y_test, y_hat_test)
        print(f"accuracy score = {acc_score_test}")
        cohen_kappa_test = metrics.cohen_kappa_score(y_test, y_hat_test)
        print(f"cohen kappa score = {cohen_kappa_test}")
        
    results = {}
    results["accuracy_train"] = acc_score_train
    results["kappa_train"] = cohen_kappa_train
    if testing:
        results["accuracy_test"] = acc_score_test
        results["kappa_test"] = cohen_kappa_test
    
    # show the confusion matrix
    if show_cm:
        y_cats = train_sub[y_label].cat.categories
        
        if testing:
            plt.figure(figsize=(3*4, 4))
            plt.subplot(1, 3, 1)
            cl_plot_cm(actual_data=y_train, predicted_data=y_hat_train, labels=y_cats, title='training')
            plt.subplot(1, 3, 2)
            cl_plot_cm(actual_data=y_test, predicted_data=y_hat_test, labels=y_cats, title='test')
            plt.tight_layout()
            plt.show()
        else:
            cl_plot_cm(actual_data=y_train, predicted_data=y_hat_train, labels=y_cats, title='training')
        
    # return the coding, model and a dictionary of measures as pandas dataframe
    return coding, cnb, pd.DataFrame(results, index=[0])




def classification(
    model,
    train_df:pd.DataFrame, 
    test_df:pd.DataFrame=None, 
    X_labels:list[str]=[], 
    y_label:str='', 
    show_cm:bool=True
) -> pd.DataFrame:
    testing = True if test_df is not None else False
    
    train_df_sub = train_df[X_labels + [y_label]].dropna()
    if testing:
        test_df_sub = test_df[X_labels + [y_label]].dropna()

    x_train = train_df_sub[X_labels]
    y_train = train_df_sub[y_label]
    if testing:
        x_test = test_df_sub[X_labels]
        y_test = test_df_sub[y_label]

    model = model.fit(x_train, y_train)
    
    y_hat_train = model.predict(x_train)
    if testing:
        y_hat_test = model.predict(x_test)
    
    acc_train = metrics.accuracy_score(y_train, y_hat_train)
    kappa_train = metrics.cohen_kappa_score(y_train, y_hat_train)
    if testing:
        acc_test = metrics.accuracy_score(y_test, y_hat_test)
        kappa_test = metrics.cohen_kappa_score(y_test, y_hat_test)

    hashmap = {"accuracy": [acc_train], "cohen_kappa": [kappa_train]}
    index = ["train"]
    
    if testing:
        hashmap["accuracy"].append(acc_test)
        hashmap["cohen_kappa"].append(kappa_test)
        index.append("test")
    
    results = pd.DataFrame(hashmap, index=index)
    
    # show the confusion matrix
    if show_cm:
        
        if testing:
            plt.figure(figsize=(3*4, 4))
            plt.subplot(1, 2, 1)
        else:
            plt.figure(figsize=(4, 4))
        cl_plot_cm(
            actual_data=pd.Series(y_train, dtype="category"),
            predicted_data=y_hat_train,
            title='train data'
        )
        if testing:
            plt.subplot(1, 2, 2)
            cl_plot_cm(
                actual_data=pd.Series(y_test, dtype="category"),
                predicted_data=y_hat_test, 
                title='test data'
            )
        plt.show()
    
    return results




def cl_plot_svm(X, y, model):
    # Keep original DataFrame for column names
    X_arr = np.array(X)

    # --- handle y for plotting colors ---
    if hasattr(y, "cat"):  # categorical dtype
        y_num = y.cat.codes.replace(-1, np.nan)
        label_names = list(y.cat.categories)
    else:
        unique_labels = pd.unique(y)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_num = pd.Series(y).map(label_map)
        label_names = list(unique_labels)

    y_num = np.array(y_num)
    y_label = y.name
    y = np.array(y)

    # --- scatter plot of data points ---
    scatter = plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_num, s=30, cmap=plt.cm.Paired)

    # --- legend for categories (match scatter colors exactly) ---
    cmap = scatter.cmap
    norm = scatter.norm
    handles = [
        plt.Line2D([], [], color=cmap(norm(i)), marker='o', linestyle='',
                   markersize=8, label=str(label))
        for i, label in enumerate(label_names)
    ]
    plt.legend(handles=handles, title=y_label)

    # --- plot the decision function ---
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 200)
    yy = np.linspace(ylim[0], ylim[1], 200)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # keep column names to avoid sklearn warning
    xy_df = pd.DataFrame(xy, columns=X.columns)
    Z = model.decision_function(xy_df)    
    if Z.ndim > 1:
        Z = model.predict(xy_df)
        # convert predictions to numeric codes
        Z = pd.Series(Z).map({label: i for i, label in enumerate(label_names)})
        Z = np.array(Z).reshape(XX.shape)
        ax.contourf(XX, YY, Z, alpha=0.2, cmap=plt.cm.Paired)
    else:
        Z = Z.reshape(XX.shape)
    
        # --- plot decision boundary and margins ---
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                   alpha=0.5, linestyles=['--', '-', '--'],
                   linewidths=1)

    # --- highlight support vectors ---
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')

    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('SVM Decision Boundary and Support Vectors')
    plt.show()





def clu_elbow_plot(
    data: pd.DataFrame,
    model,
    k_range: tuple,
    figure_size: tuple = (20, 3)
) -> None:

    metrics = ["distortion", "silhouette", "calinski_harabasz"]
    _, ax = plt.subplots(1, len(metrics), figsize=figure_size)
    
    for i, m in enumerate(metrics):
        
        visualizer = KElbowVisualizer(model, k=(2,10), metric=m, ax=ax[i])
        ax[i] = visualizer.fit(data)
        plt.title(f"Data - {m} score")

    plt.show()




def clu_hca_linkages(
    x_df:pd.DataFrame,
    k:int,
    fig_size:tuple=(5,5)
) -> None:
    
    _, axs = plt.subplots(2, 2, constrained_layout=True, figsize=fig_size)
    linkages = ['single', 'average', 'complete', 'ward']

    columns = list(x_df.columns)
    
    for i, link in enumerate(linkages):
        hca_model = AgglomerativeClustering(linkage=link, n_clusters=k)
        hca_model.fit(x_df)
        clusters = hca_model.labels_
        
        axs[i // 2, i % 2].scatter(x_df[columns[0]], x_df[columns[1]], c=clusters, cmap='rainbow')
        axs[i // 2, i % 2].set_title(link)
    
    plt.show()




def clu_dbscan(
    data:pd.DataFrame,
    X_labels:list[str]|None=None,
    eps:float=1,
    min_samples:int=4,
    use_knee:bool=True,
    plot_knee:bool=True,
    fig_size:tuple=(3, 3)
) -> None:
    # check if labels have been provided, if not use column titles
    if X_labels is None:
        X_labels = list(data.columns)

    X = data[X_labels]
    
    # if use_knee create the knee plot
    if use_knee:
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
        distances, indices = nbrs.kneighbors(X)
        distance_desc = sorted(distances[:,min_samples-1], reverse=True)
        
        kneedle = KneeLocator(range(1,len(distance_desc)+1), distance_desc, curve="convex", direction="decreasing")
        kneedle.plot_knee(figsize=(3, 3), xlabel = 'nr. of points', ylabel='epsilon: distance to k-neighbours')

        n_points_knee = kneedle.knee
        eps = kneedle.knee_y
        
        print('at epsilon', eps, 'we only have', n_points_knee, 'points that could end up being noise')
        print(f"now we use the {eps} as epsilon value for our DBSCAN")
    
    # run dbscan
    model = DBSCAN(eps=eps, min_samples=min_samples, algorithm='brute')
    model.fit(X)

    labels = model.labels_

    noise = X[labels < 0]
    clusters = X[labels >= 0]

    # plot results
    plt.figure(figsize=(3, 3))
    plt.scatter(clusters[X_labels[0]], clusters[X_labels[1]], c=labels[labels >= 0], cmap='rainbow')
    plt.scatter(noise[X_labels[0]], noise[X_labels[1]], color='black', label='noise')
    plt.legend()
    plt.xlabel(X_labels[0])
    plt.ylabel(X_labels[1])
    plt.show()




def rec_n_corr(
    data:pd.DataFrame, 
    liked:str, 
    long:bool=True, 
    columns:list[str]=['title', 'user', 'rating'],
    how_plot:bool=True
) -> pd.DataFrame:

    if long:
        target_df = data[data[columns[0]] == liked][[columns[1], columns[2], columns[0]]]
        target_df = target_df.rename(columns={columns[2]:f"target_{columns[2]}", columns[0]:f"target_{columns[0]}"})
    
        merged_df = data.merge(target_df, on=columns[1])
    
        counts = merged_df.dropna(subset=[columns[2], f"target_{columns[2]}"]) \
                  .groupby(columns[0])[columns[2]].count() \
                  .to_frame(name='n corr')
        counts.columns = ['n corr']
        
    else:
        target_df = data[liked].dropna()

        merged_df = data[data.index.isin(target_df.index)]
        merged_df = merged_df.dropna(axis=1, how='all')

        counts = merged_df.count()
        counts = counts.to_frame(name="n corr")

    counts.sort_values(by='n corr', ascending=True, inplace=True)
        
    if how_plot:
        plt.figure(figsize=(4,3))
        plt.bar(range(1, len(counts)+1), counts['n corr'])
        plt.ylabel('n corr')
        
        plt.ylim(0, 100)
        plt.show()
        
    return counts




def rec_single_title(
    data:pd.DataFrame, 
    liked:str, 
    counts_df:pd.DataFrame, 
    long:bool=True, 
    top:int=5, 
    min_n_cor:int=25, 
    columns:list[str]=['title', 'user', 'rating']
):
    if long:
        filtered = counts_df[counts_df['n corr'] >= min_n_cor]
        filtered_df = data[data[columns[0]].isin([liked] + list(filtered.index))]
        wide_df = filtered_df.pivot_table(columns=columns[0], values=columns[2], index=columns[1])
    else:
        remove = counts_df[counts_df['n corr'] < min_n_cor]
        wide_df = data.drop(columns=list(remove.index))

    liked_scores = wide_df[liked]
    correlations = wide_df.corrwith(liked_scores).sort_values(ascending=False)

    return correlations[0:top]




def rec_single_user_long(
    data:pd.DataFrame,
    user_id:int,
    top:int=5,
    min_n_cor:int=5,
    min_n_ratings:int=10,
    columns:list[str]=["title", "user", "rating"]
) -> list:

    item_col, user_col, rating_col = columns

    n_ratings = data.groupby(item_col)[[rating_col]].count()
    to_drop = list(n_ratings[n_ratings[rating_col] < min_n_ratings].index)

    filtered = data[~data[item_col].isin(to_drop)]

    means_t = filtered.groupby(item_col)[rating_col].transform('mean')
    means = filtered.groupby(item_col)[rating_col].mean()

    filtered = filtered.copy()
    filtered.loc[:, 'rating_adjusted'] = filtered[rating_col] - means_t

    user_data = filtered[filtered[user_col] == user_id]
    user_items = user_data.dropna()[item_col].unique()

    potential_items = filtered[item_col].unique()[~pd.Series(filtered[item_col].unique()).isin(user_items)]

    results = []
    for item in potential_items:
        filtered2 = filtered[filtered[item_col].isin(np.append(user_items, item))]
        
        target_ratings = filtered2[filtered2[item_col] == item][[user_col, rating_col]]
        target_ratings = target_ratings.rename(columns={rating_col: 'corr'}) 
        target_ratings[target_ratings[user_col] == user_id]
        
        merged_df = filtered2.merge(target_ratings, on=user_col)
        
        n_ratings_in_corr = merged_df.groupby(item_col)[[rating_col]].count()
        to_drop = list(n_ratings_in_corr[n_ratings_in_corr[rating_col] < min_n_corr].index)
        filtered3 = merged_df[~merged_df[item_col].isin(to_drop)]

        correlations = filtered3.dropna().groupby(item_col)[[rating_col, 'corr']].corr().unstack()[rating_col]['corr']
        
        merged_df2 = pd.merge(user_data, correlations, left_on=item_col, right_index=True, how='left').dropna()
        
        merged_df2['weighted'] = merged_df2['rating_adjusted']*merged_df2['corr']
        
        mean_weight = sum(merged_df2['weighted'])/sum(abs(merged_df2['corr']))
        
        res = mean_weight + means[item]
        results.append([res, item])

    results = sorted(results, reverse=True)
    return results[:top]



def rec_single_user_wide(
    data:pd.DataFrame,
    user_id:int,
    top:int=5,
    min_n_cor:int=5,
    min_n_ratings:int=10,
    large_data:bool=False
) -> pd.DataFrame:
    filtered = data.drop(columns=data.columns[data.count() < min_n_ratings])

    means = filtered.mean()
    adj = filtered - means

    user_data = adj.loc[user_id].dropna()
    user_items = user_data.index
    
    potential_items = adj.drop(columns=user_items).columns

    if not large_data:
        correlations = adj.corr(min_periods=min_n_cor)

    results = []
    for item in potential_items:

        if large_data:
            correlations_dict = {}
            for i in user_items:
                r = adj[[item, i]].corr(min_periods=min_n_cor).iloc[0,1]
                correlations_dict[i] = r
            
            correlations = pd.DataFrame.from_dict(correlations_dict, orient='index')
            sim_temp = correlations.dropna()
        else:
            sim_temp = correlations[item][user_items].dropna()
        
        merged_df = pd.concat([sim_temp, user_data], axis=1).dropna()
        merged_df.columns = ['corr', 'user adj. rating']
        
        if merged_df.empty:
            continue
        merged_df['weighted score'] = merged_df['corr'] * merged_df['user adj. rating']
        
        mean_weight = merged_df['weighted score'].sum() / merged_df['corr'].abs().sum()
        res = mean_weight + means[item]
        
        results.append([res, item])

    results = sorted(results, reverse=True)
    return results[:top] 