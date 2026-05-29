import numpy as np

def create_features(df):

    df['lag_1'] = df['total_cases'].shift(1)
    df['lag_2'] = df['total_cases'].shift(2)
    df['lag_3'] = df['total_cases'].shift(3)

    # Rolling statistics
    df['rolling_mean_3'] = (
        df['total_cases']
        .rolling(3)
        .mean()
    )

    df['rolling_std_3'] = (
        df['total_cases']
        .rolling(3)
        .std()
    )

    # Month feature
    df['month'] = (
        (df['weekofyear'] // 4) + 1
    )

    # Seasonal feature
    df['season'] = np.where(
        df['month'].isin([6,7,8,9]),
        1,
        0
    )

    # Temperature interaction
    df['temp_humidity_interaction'] = (
        df['reanalysis_air_temp_k'] *
        df['reanalysis_relative_humidity_percent']
    )

    df = df.dropna()

    return df

