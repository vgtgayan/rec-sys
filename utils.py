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