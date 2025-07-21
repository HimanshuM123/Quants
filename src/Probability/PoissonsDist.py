# lambda =20 -> average customers per hr

import numpy as np
import plotly.graph_objects as go
from scipy.stats import stats, poisson

# generate pmf
k = np.arange(5, 41)
y = poisson.pmf(k, mu=20)

# create plot
fig = go.Figure(data=go.Scatter(x=k, y=y, mode='lines'))
fig.update_layout(title='Poisson Distribution',
                  xaxis_title='k: Number of shops visited an hr',
                  yaxis_title='Probability',
                  font=dict(size=18),
                  width=650,
                  title_x=0.5,
                  height=400,
                  template='simple_white'
                  )

fig.show()
