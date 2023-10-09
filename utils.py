# Convert pandas dataframe into compressed sparse row (CSR) format
def pandas_df_to_csr(
    df, 
    USER_COL = "UserId", 
    ITEM_COL = "ProductId", 
    RATING_COL = "Rating"
):
    from pandas.api.types import CategoricalDtype
    from scipy import sparse

    users = df[USER_COL].unique()
    items = df[ITEM_COL].unique()
    shape = (len(users), len(items))

    # Create indices for users and movies
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
    df['UserIndex'] = df[USER_COL].astype(user_cat).cat.codes
    df['ItemIndex'] = df[ITEM_COL].astype(item_cat).cat.codes
    user_map = df[["UserIndex", USER_COL]].drop_duplicates().sort_values(['UserIndex']).set_index('UserIndex', verify_integrity=True)
    item_map = df[["ItemIndex", ITEM_COL]].drop_duplicates().sort_values(['ItemIndex']).set_index('ItemIndex', verify_integrity=True)

    # Conversion via COO matrix
    coo = sparse.coo_matrix((df[RATING_COL], (df['UserIndex'], df['ItemIndex'])), shape=shape)
    csr = coo.tocsr()
    return user_map, item_map, csr

# Normalize list of values into a specified range (default 0-1)
def normalize(arr, t_min=0, t_max=1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)   
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

# Sort a given dictionary by value in descending order
def sort_by_score(x):
    return dict(sorted(x.items(), reverse=True, key=lambda item: item[1]))

# Evaluation metrics
def mean_absolute_error(actual, predicted):
    return np.abs(actual - predicted).mean()

def root_mean_square_error(actual, predicted):
    return np.sqrt(((actual - predicted)**2)).mean()

def precision_at_k(actual, predicted, k):
    return len(
        set(actual) & set(predicted[:k])
    )/k

def recall_at_k(actual, predicted, k):
    return len(
        set(actual) & set(predicted[:k])
    )/len(actual)

def f1_acore_at_k(actual, predicted, k):
    p = precision_at_k(actual, predicted, k)
    r = recall_at_k(actual, predicted, k)
    if p + r == 0:
        return 0
    return 2*(p*r)/(p+r)

# Pretty print
def pretty(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))
