def check_krona_dim(self, krona_dim, in_features, out_features):
    # make sure that krona_dim is divisible by both in_features and out_features
    min_dim = min(in_features, out_features)
    krona_dim_left, krona_dim_right = krona_dim, krona_dim
    while min_dim % krona_dim_left and min_dim % krona_dim_right:
        krona_dim_left -= 1
        krona_dim_right += 1
    if min_dim % krona_dim_left == 0:
        krona_dim = krona_dim_left
    elif min_dim % krona_dim_right == 0:
        krona_dim = krona_dim_right
