from visualizer.plotlyInit import *

import pandas as pd
pd.options.display.float_format = '{:.4f}'.format


targetQuery = ['냉장고'] #if you want to check another queries, you reset the list.

df = pd.read_csv('./output.csv')

print(df.groupby(['Query'], as_index=False).mean())
print(df.groupby(['Query'], as_index=False).group_keys)
print(df.groupby(['Query','OriginalTitle'], as_index=False).mean().to_string())
df2 = (df.groupby(['Query','OriginalTitle'], as_index=False).mean())

groupedQuery = df.groupby(['Query'], as_index=False)
print(groupedQuery)

fig = dict()
fig['data']=list()

i=0
for query, title in zip(df2['Query'], df2['OriginalTitle']):
    if query not in targetQuery:
        continue
    i+=1
    df3 = df[df.Query == query][df.OriginalTitle == title]
    print(df3.to_string())
    data = dict()
    data['x'] = df3.OriginalPrice  # pd.Series(list(groupedQuery.groups.keys()))#list(groupedQuery.groups.keys()).index(query)
    data['y'] = df3.CandidatePrice
    data['text'] = df3.CandidateTitle
    data['mode'] = 'markers'
    data['name'] = title
    fig['data'].append(data)
fig['layout'] = dict({
        'xaxis': {'title': 'Original Price'},
        'yaxis': {'title': "Candidate Price"}
    })
print(BASE_DIR)
plot(fig, filename=BASE_DIR+'test.html')
