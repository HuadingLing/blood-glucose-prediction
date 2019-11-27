#import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import utils

# 每个患者对应一个xml文件
sheet = ["Baseline", "3M", "6M", "9M", "12M", "15M", "18M", "21M", "24M", "27M", "30M", "33M", "36M"]
data_start_pos = [7, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

def load_glucose_dataset(xlsx_path, nb_past_steps, nb_future_steps, train_fraction,
        valid_fraction, test_fraction, sheet_pos, patient_id, max_length):
    xs, ys = load_glucose_data(xlsx_path, nb_past_steps, nb_future_steps, sheet_pos, patient_id, max_length)
    ys = np.expand_dims(ys, axis=1) # why expand_dims?

    x_train, x_valid, x_test = split_data(xs, train_fraction,
            valid_fraction, test_fraction)
    y_train, y_valid, y_test = split_data(ys, train_fraction,
            valid_fraction, test_fraction)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def load_glucose_data(xlsx_path, nb_past_steps, nb_future_steps, sheet_pos, patient_id, max_length):
    df_glucose_level = load_ohio_series(xlsx_path, sheet_pos, patient_id)
    
    nd_glucose_level = df_glucose_level.values
    consecutive_segments=utils.series_to_segments(nd_glucose_level, nb_past_steps+nb_future_steps, max_length)
    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments]

    xss = [sup[0] for sup in sups] # list of array
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys # why expand_dims?
'''
def load_glucose_data(xml_path, nb_past_steps, nb_future_steps):
    df_glucose_level = load_ohio_series(xml_path, "glucose_level", "value")
    dt = df_glucose_level.index.to_series().diff().dropna() 
    # 对index进行差分：
    # dt.index[0]=df_glucose_level.index[1],dt.index[0]
    # dt.value[0]=df_glucose_level.index[1]-df_glucose_level.index[0]
    # ...
    # dt.len比df_glucose_level.len少1
    # 结合以下部分，举例：
    # df=pd.Series([1,3,7,15,50],index=[1,6,11,13,18])
    # dt=pd.Series([5,5,2,5],index=[6,11,13,18])
    # idx_breaks=array([[2]], dtype=int64)
    # nd=[1,3,7,15,50]
    # consecutive_segments1=[array([1, 3], dtype=int64), array([ 7, 15, 50], dtype=int64)]
    
    
    idx_breaks = np.argwhere(dt!=pd.Timedelta(5, 'm')) # 列出dt.value!=5的所有位置，（定义为数据断点）

    # It would be possible to load more features here
    nd_glucose_level = df_glucose_level.values
    
    
    #consecutive_segments = np.split(nd_glucose_level, idx_breaks.flatten()) # 左开右闭切分,fuck you,你的代码写错了
    consecutive_segments = np.split(nd_glucose_level, (idx_breaks+1).flatten())

    # 保留足够长的连续数据段
    consecutive_segments = [c for c in consecutive_segments if len(c) >=
            nb_past_steps+nb_future_steps]

    sups = [utils.sequence_to_supervised(c, nb_past_steps, nb_future_steps) for
            c in consecutive_segments] # sups is list of tuple (x,y)

    xss = [sup[0] for sup in sups] # list of array
    yss = [sup[1] for sup in sups]

    xs = np.concatenate(xss)
    ys = np.concatenate(yss)

    return np.expand_dims(xs, axis=2), ys # why expand_dims?
'''

def split_data(xs, train_fraction, valid_fraction, test_fraction):
    n = len(xs)
    nb_train = int(np.ceil(train_fraction*n))
    nb_valid = int(np.ceil(valid_fraction*n))
    i_end_train = nb_train
    i_end_valid = nb_train + nb_valid

    return np.split(xs, [i_end_train, i_end_valid])
    # 三部分依次以测量时间顺序来拆分？竟不是随机拆分？
    # 不打乱顺序直接划分？Are you kidding me?

def load_ohio_series(xlsx_path, sheet_pos, patient_id):
    read_data = pd.read_excel(xlsx_path, sheet_name=sheet[sheet_pos])
    #Sheet0 = unprocessed[sheet_pos]
    series = read_data.iloc[patient_id-1,data_start_pos[sheet_pos]:]
    return series

'''
def load_ohio_series(xml_path, variate_name, attribute, time_attribue="ts"):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root:
        if child.tag == variate_name: #一个child对应一列测量值？？？
            dates = []
            values = []
            for event in child:
                ts = event.attrib[time_attribue]
                date = pd.to_datetime(ts, format='%d-%m-%Y %H:%M:%S')
                date = date.replace(second=0) # 把秒置为零
                value = float(event.attrib[attribute]) # 测量值
                dates.append(date)
                values.append(value)
                
            # 取出所有dates和values之后
            index = pd.DatetimeIndex(dates) # 将datetime类型转化为index类型
            series = pd.Series(values, index=index)
            return series
'''