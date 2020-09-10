def calculate_weighted_sentiment(x):
    n = len(x)
    positive_weights = [0.25, 0.5, 0.75] + [1]*n
    negative_weights = [-0.25, -0.5, -0.75] + [-1]*n
    n_positives = x.count('positive')
    n_negatives = x.count('negative')
    weighted_sum = sum(positive_weights[0:n_positives]) + sum(negative_weights[0:n_negatives])
    return weighted_sum

def aggregate_sentiment_by_business(data):
    data_ = data \
        .groupby('business_id') \
        ['predicted'] \
        .apply(list) \
        .reset_index()
    data_['sentiment_score'] = data_['predicted'].apply(calculate_weighted_sentiment)
    data_['sentiment_n_sentences'] = data_['predicted'].apply(len)
    data_ = data_.drop(['predicted'], axis=1)
    return data_

def aggregate_sentiment_by_business_allergy(data, allergen_cols):
    data_ = data \
        [['business_id', 'predicted'] + allergen_cols] \
        .melt(
            id_vars=['business_id', 'predicted'],
            value_vars=allergen_cols,
            var_name='allergen',
            value_name='is_allergen'
            )
    data_ = data_[data_['is_allergen'].astype(int)==1]
    data_ = data_ \
        .groupby(['business_id', 'allergen']) \
        ['predicted'] \
        .apply(list) \
        .reset_index()
    data_['sentiment_score'] = data_['predicted'].apply(calculate_weighted_sentiment)
    data_['sentiment_n_sentences'] = data_['predicted'].apply(len)
    data_ = data_.drop(['predicted'], axis=1)
    return data_
