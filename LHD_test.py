import numpy as np
import pandas as pd

import utils

nb_past_steps=2
nb_future_steps=2

df=pd.Series([1,3,7,2,15,50,44,12,22,33],index=[1,6,11,16,13,15,20,25,30,35])

dt = df.index.to_series().diff().dropna()

idx_breaks = np.argwhere(dt!=5)

nd=df.values
consecutive_segments1 = np.split(nd, (idx_breaks+1).flatten())

consecutive_segments2 = [c for c in consecutive_segments1 if len(c) >= 
                         nb_past_steps+nb_future_steps]

sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments2]

xss = [sup[0] for sup in sups]
yss = [sup[1] for sup in sups]

xs = np.concatenate(xss)
ys = np.concatenate(yss)

x=np.expand_dims(xs, axis=2)
y=np.expand_dims(ys, axis=1)

print(x)
print(y)