def cria_feature(df):

    import pandas as pd

    df['Torque * Rotational speed'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']

    df['Torque * Tool wear'] = df['Torque [Nm]'] * df['Tool wear [min]']

    return df

def get_rank_by_product_id(top,product_id):

    if product_id in top['Product ID'].values:

        return top.loc[top['Product ID'] == product_id, 'Rank'].values[0]
    else:

        return 51
    

def custom_encode(top,product_ids):

    import numpy as np

    result = []
    result.append([get_rank_by_product_id(top,product_id) for product_id in product_ids])

    return np.array(result).reshape(-1, 1)