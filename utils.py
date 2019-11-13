import numpy as np

def dataframe_to_supervised(df, nb_past_steps, nb_steps_in_future):
    xs = []
    ys = []
    for column in df.columns:
        sequence = df[column].as_matrix()
        x, y = sequence_to_supervised(sequence, nb_past_steps,
                nb_steps_in_future)
        x = x.reshape(x.shape+(1,))
        y = y.reshape(y.shape+(1,))
        xs.append(x)
        ys.append(y)

    return np.concatenate(xs, axis=2), np.concatenate(ys, axis=1)[:,0]

def rolling_window(sequence, window):
    """Splits the sequence into window sized chunks by moving the window along
    the sequence."""
    shape = sequence.shape[:-1] + (sequence.shape[-1] - window + 1, window)
    strides = sequence.strides + (sequence.strides[-1],)
    return np.lib.stride_tricks.as_strided(sequence, shape=shape, strides=strides)

def sequence_to_supervised(data, nb_past_steps, nb_steps_in_future): # data is 1-D array
    """Computes feature and target data for the sequence. The features are the
    number of past steps used, and the target is the specified number of steps
    into the future."""
    x = rolling_window(data, nb_past_steps)
    y = data[nb_past_steps+nb_steps_in_future-1::1]

    x, y = zip(*zip(x, y))

    return np.array(x), np.array(y)

def sequence_to_supervised_all(data, nb_past_steps, nb_future_steps):
    """Computes feature and target data for the sequence. The features are the
    number of past steps used, and the targets are all the future values until
    the number of steps into the future specified."""
    x = rolling_window(data, nb_past_steps)
    y = rolling_window(data, nb_future_steps)

    x, y = zip(*zip(x[:-nb_future_steps], y[nb_past_steps:]))

    return np.array(x), np.array(y)

def series_to_segments(data, least_length, max_length):
    #nan_index=np.argwhere(data!=data)
    #segments_temp = np.split(data, nan_index.flatten())
    nan_index=np.argwhere(data!=data)
    error_index=np.where(np.abs(np.diff(data))>=40)[0]+1
    break_index = np.sort(np.unique(np.concatenate((nan_index.flatten(),error_index.flatten()))))
    segments_temp = np.split(data, break_index)
    segments_temp = [c[1:] for c in segments_temp]
    segments = []
    for c in segments_temp:
        if len(c) > max_length:
            seq_num = int(len(c)/max_length)
            idx_breaks = np.linspace(max_length, max_length*seq_num, seq_num, dtype=int)
            d = np.split(c, idx_breaks.flatten())
            for e in d[:-1]:
                segments.append(e)
            if len(d[-1]) >= least_length:
                segments.append(d[-1])
        elif len(c) >= least_length:
            segments.append(c)
    return segments