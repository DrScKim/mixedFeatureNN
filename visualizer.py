
import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

df = pd.read_csv('./output.csv')

print(df.groupby(['Query'], as_index=False).mean())
print(df.groupby(['Query'], as_index=False).group_keys)
print(df.groupby(['Query','OriginalTitle'], as_index=False).mean().to_string())
#print(df.groupby(['Query','OriginalTitle'], as_index=False).var().to_string())
#print(df['Query'])

#d = df.loc[df['Query']=='군기저귀']
df2 = (df.groupby(['Query','OriginalTitle'], as_index=False).mean())


for query, title, meanPrice in zip(df2['Query'], df2['OriginalTitle'], df2['CandidatePrice']):
    print(query)
    print(title)
    print(meanPrice)
    #df3 = df.query("Query == '%s' and OriginTitle = '%s'" % (query, title))
    df3 = df[df.Query == query][df.OriginalTitle == title]
    print(df3['CandidatePrice'])
    sr = (df3['CandidatePrice'] - meanPrice)/meanPrice
    print(df)
    df.append(sr)
