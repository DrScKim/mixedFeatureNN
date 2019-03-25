from visualizer.plotlyInit import *

import pandas as pd
pd.options.display.float_format = '{:.4f}'.format

df = pd.read_csv('./output.csv')

print(df.groupby(['Query'], as_index=False).mean())
print(df.groupby(['Query'], as_index=False).group_keys)
print(df.groupby(['Query','OriginalTitle'], as_index=False).mean().to_string())

df2 = (df.groupby(['Query','OriginalTitle'], as_index=False).mean())

'''
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
'''
import pandas as pd
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
df = pd.read_csv('http://www.stat.ubc.ca/~jenny/notOcto/STAT545A/examples/gapminder/data/gapminderDataFiveYear.txt', sep='\t')
df2007 = df[df.year==2007]
df1952 = df[df.year==1952]
df.head(2)

fig = {
    'data': [
  		{
  			'x': df2007.gdpPercap,
        	'y': df2007.lifeExp,
        	'text': df2007.country,
        	'mode': 'markers',
        	'name': '2007'},
        {
        	'x': df1952.gdpPercap,
        	'y': df1952.lifeExp,
        	'text': df1952.country,
        	'mode': 'markers',
        	'name': '1952'}
    ],
    'layout': {
        'xaxis': {'title': 'GDP per Capita', 'type': 'log'},
        'yaxis': {'title': "Life Expectancy"}
    }
}
print(BASE_DIR)
plot(fig, filename=BASE_DIR+'test.html')
# IPython notebook
# py.iplot(fig, filename='pandas/multiple-scatter')

#url = py.plot(fig, filename='pandas/multiple-scatter')