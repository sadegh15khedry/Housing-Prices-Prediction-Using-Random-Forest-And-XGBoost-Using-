import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_name):
    return pd.read_csv('../datasets/'+file_name)


def split_data(df, feature_column, label_column, test_size=0.15,  val_size=0.15, random_state=50):
    # Split the data into train+val and test
    x_train_val, x_test, y_train_val, y_test = train_test_split(df[feature_column], df[label_column], 
                                                                test_size=test_size, random_state=random_state)
    # Calculate the relative validation size with respect to the train+val set
    val_size_relative = val_size / (1 - test_size)
    
    # Split the train+val set into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, 
                                                      test_size=val_size_relative, random_state=random_state)
    
    return x_train, x_test, x_val, y_train, y_val, y_test



def normalize_data(df, method, normalization_columns):
    if method == 'max_abs':
        for column in normalization_columns:
            df[column] = df[column] / df[column].abs().max()
    elif method == 'min_max':
        for column in normalization_columns: 
            df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())     
    elif method == 'z_score':
        for column in normalization_columns: 
            df[column] = (df[column] - df[column].mean()) / df[column].std()
    elif method == 'robust':
        for column in normalization_columns:
            df[column] = (df[column] - df[column].median()) / (df[column].quantile(0.75) - df[column].quantile(0.25))
    elif method == 'log':
        for column in normalization_columns:
            df[column] = np.log1p(df[column])  # log1p is used to avoid log(0)
    elif method == 'l2':
        df = df.apply(lambda x: x / np.sqrt(np.sum(np.square(x))), axis=1)
    
    return df   