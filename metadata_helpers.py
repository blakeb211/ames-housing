""" Helper file with functions for saving best parameters and rmse scores. """

import pandas as pd
import os


def save_score(pipe, rmse):
    """ Save the grid_search.best_params_ dictionary and rmse to a file.
    Load local scores.csv dataframe.Save or replace entry with the same 
    key as param_dict."""
    model_name = list(pipe.named_steps.keys())[-1]
    param_dict = pipe._final_estimator.get_params()

    # Prepare new row for the dataframe
    col_names = "name", "rmse", "params"
    new_row = pd.Series([model_name, rmse, str(param_dict)])
    new_row = pd.DataFrame(new_row).T
    new_row.columns = col_names

    # Load existing dataframe if csv file exists
    scores_filename = "scores.csv"
    df = pd.DataFrame(columns=col_names)
    if os.path.exists(scores_filename):
        df = pd.read_csv(scores_filename)

    if df['name'].str.contains(model_name).any():
        # overwrite existing row for that model
        search_results = df[df.name == model_name]
        assert len(search_results) == 1
        idx_for_replace = search_results.index
        df.iloc[idx_for_replace] = new_row
    else:
        df = pd.concat([df, new_row], axis=0)

    df.to_csv(scores_filename, index=False)



if __name__ == "__main__":
    pass
    # save_score({"linear__l1_ratio": 0.14},36000)
