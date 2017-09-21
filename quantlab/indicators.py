def HA(df):
    df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4

    nt = namedtuple('nt', ['Open','Close'])
    previous_row = nt(df.ix[0,'Open'],df.ix[0,'Close'])
    i = 0
    for row in df.itertuples():
        ha_open = (previous_row.Open + previous_row.Close) / 2
        df.ix[i,'HA_Open'] = ha_open
        previous_row = nt(ha_open, row.HA_Close)
        i += 1

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    return df