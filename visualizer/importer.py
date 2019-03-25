
###
import util.configurations as config

config = config.parser()
plotly_username=config['PLOTLY']['plotly_username']
plotly_apikey=config['PLOTLY']['plotly_apikey']
###

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = BASE_DIR + '/templates/'
BASIC_HEATMAP_PATH = TEMPLATE_DIR+'heatmap.html'
BASIC_BARCHART_PATH = TEMPLATE_DIR+'/featureEngineering/featureImportance.html'
BASIC_PREC_RECALL_PATH = TEMPLATE_DIR+'/featureEngineering/prec_rec.html'
BASIC_LINE_CHART_PATH = TEMPLATE_DIR+'/featureEngineering/line_chart.html'
BASIC_DOT_PLOT_PATH = TEMPLATE_DIR+'/featureEngineering/dot_plot.html'

try:
    import plotly
    import plotly.plotly as py
    import plotly.tools as tls
    import plotly.graph_objs as go
    from plotly import __version__
    from plotly.offline import plot
    print(__version__)
    plotly.tools.set_credentials_file(username=plotly_username, api_key=plotly_apikey)
    import numpy as np
except:
    print('Not installed plotly')