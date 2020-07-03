
# Creating an NLP Neural Network to Predict Category of Wine & Discover Unique Keywords

Flatiron School - Cohort 100719PT

Instructor - James Irving

By Acusio Bivona

## Introduction: For this project, I will be using a deep NLP network and a random forest model on a dataset consisting of professional wine reviews in an attempt to try and determine if specific keywords are affiliated with cheap or expensive wine. One important thing to note is that the price point for cheap vs. expensive wine was arbitrarily chosen based on the distribution of the data. 

> If such results are determined, then I believe a use case for these results would be that a winery can use the specific keywords in their marketing efforts to both portray a 'fancier' feel to cheap wines, as well as provide a 'friendlier' feel to expensive wines. I also believe that wineries can take these keywords and use them in wine production, such as finding unique flavors and aromas for each group and experimenting with their recipes to provide different products that can be more appealing to more groups. 

# Obtain and investigate data


```python
import pandas as pd
df = pd.read_csv("winemag-data-130k-v2.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>NaN</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>NaN</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>NaN</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>NaN</td>
      <td>Alexander Peartree</td>
      <td>NaN</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 129971 entries, 0 to 129970
    Data columns (total 14 columns):
    Unnamed: 0               129971 non-null int64
    country                  129908 non-null object
    description              129971 non-null object
    designation              92506 non-null object
    points                   129971 non-null int64
    price                    120975 non-null float64
    province                 129908 non-null object
    region_1                 108724 non-null object
    region_2                 50511 non-null object
    taster_name              103727 non-null object
    taster_twitter_handle    98758 non-null object
    title                    129971 non-null object
    variety                  129970 non-null object
    winery                   129971 non-null object
    dtypes: float64(1), int64(2), object(11)
    memory usage: 13.9+ MB



```python
df.drop(['Unnamed: 0'], axis=1, inplace=True)
```


```python
df.isna().sum()
```




    country                     63
    description                  0
    designation              37465
    points                       0
    price                     8996
    province                    63
    region_1                 21247
    region_2                 79460
    taster_name              26244
    taster_twitter_handle    31213
    title                        0
    variety                      1
    winery                       0
    dtype: int64




```python
def fill_cols_na(df, column):
    
    """This function can be used to fill in columns with a literal 'N/A', if it is necessary
    
    Parameters:
    
    df - dataframe to pull columns from
    
    column - can be a single column or list of columns"""
        
    df2 = df.copy()
    
    df2[column] = df2[column].fillna('N/A')
    
    return df2
```


```python
df = fill_cols_na(df, ['country', 'designation', 'province', 'region_1', 'region_2', 'taster_name',
                      'taster_twitter_handle', 'variety'])
```


```python
pd.set_option('display.max_rows', 500)
```


```python
df['price'].value_counts()
```




    20.0      6940
    15.0      6066
    25.0      5805
    30.0      4951
    18.0      4883
    12.0      3934
    40.0      3872
    35.0      3801
    13.0      3549
    16.0      3547
    10.0      3439
    22.0      3357
    50.0      3334
    14.0      3215
    45.0      3135
    17.0      3053
    28.0      2942
    24.0      2826
    19.0      2816
    60.0      2277
    11.0      2058
    55.0      1981
    32.0      1963
    38.0      1728
    23.0      1715
    26.0      1706
    65.0      1614
    75.0      1403
    42.0      1403
    36.0      1392
    29.0      1387
    9.0       1339
    48.0      1309
    21.0      1232
    27.0      1193
    70.0      1100
    34.0      1069
    39.0       924
    8.0        892
    80.0       881
    33.0       668
    90.0       665
    85.0       655
    44.0       586
    100.0      585
    49.0       585
    37.0       527
    52.0       517
    7.0        433
    46.0       430
    58.0       413
    95.0       397
    54.0       384
    125.0      328
    43.0       311
    56.0       291
    31.0       276
    59.0       263
    120.0      262
    62.0       253
    150.0      239
    47.0       239
    68.0       222
    110.0      200
    69.0       195
    41.0       194
    53.0       187
    64.0       186
    72.0       178
    57.0       144
    130.0      141
    105.0      133
    79.0       128
    78.0       125
    6.0        120
    66.0       118
    115.0      114
    140.0      112
    135.0      110
    63.0        98
    200.0       94
    67.0        89
    99.0        84
    175.0       82
    51.0        79
    82.0        75
    89.0        70
    145.0       67
    74.0        65
    92.0        61
    160.0       60
    61.0        59
    73.0        53
    88.0        52
    77.0        52
    250.0       51
    98.0        48
    5.0         46
    86.0        45
    84.0        44
    225.0       41
    76.0        39
    96.0        37
    300.0       36
    165.0       36
    155.0       34
    170.0       34
    94.0        33
    87.0        32
    180.0       31
    93.0        30
    112.0       29
    103.0       28
    450.0       27
    97.0        27
    83.0        27
    108.0       27
    190.0       26
    195.0       26
    350.0       21
    210.0       21
    71.0        20
    107.0       19
    185.0       19
    102.0       18
    81.0        17
    149.0       17
    400.0       17
    260.0       17
    104.0       16
    106.0       16
    101.0       15
    240.0       15
    275.0       15
    500.0       14
    118.0       14
    109.0       14
    128.0       14
    111.0       13
    114.0       13
    235.0       13
    220.0       13
    113.0       13
    129.0       12
    290.0       12
    230.0       12
    169.0       12
    325.0       11
    116.0       11
    117.0       11
    132.0       11
    4.0         11
    91.0        10
    138.0       10
    119.0        8
    199.0        8
    330.0        8
    142.0        8
    139.0        8
    320.0        8
    127.0        8
    215.0        8
    122.0        8
    255.0        7
    124.0        7
    179.0        7
    126.0        7
    365.0        7
    163.0        6
    143.0        6
    460.0        6
    299.0        6
    440.0        6
    236.0        5
    550.0        5
    168.0        5
    800.0        5
    141.0        5
    154.0        5
    133.0        5
    280.0        5
    148.0        5
    159.0        5
    279.0        5
    249.0        5
    137.0        5
    375.0        5
    134.0        4
    775.0        4
    380.0        4
    600.0        4
    625.0        4
    245.0        4
    360.0        4
    131.0        4
    152.0        4
    147.0        4
    166.0        4
    146.0        4
    270.0        4
    227.0        4
    144.0        4
    164.0        3
    530.0        3
    305.0        3
    475.0        3
    295.0        3
    286.0        3
    595.0        3
    262.0        3
    430.0        3
    136.0        3
    123.0        3
    153.0        3
    495.0        3
    850.0        3
    158.0        3
    162.0        3
    312.0        3
    650.0        3
    237.0        3
    167.0        3
    1100.0       2
    1500.0       2
    243.0        2
    390.0        2
    156.0        2
    476.0        2
    121.0        2
    391.0        2
    580.0        2
    151.0        2
    194.0        2
    370.0        2
    359.0        2
    187.0        2
    1000.0       2
    310.0        2
    303.0        2
    171.0        2
    333.0        2
    520.0        2
    204.0        2
    398.0        2
    2000.0       2
    307.0        2
    196.0        2
    399.0        2
    510.0        2
    469.0        2
    193.0        2
    181.0        2
    224.0        2
    161.0        2
    184.0        2
    208.0        2
    314.0        2
    265.0        2
    182.0        2
    315.0        2
    219.0        2
    248.0        2
    770.0        2
    285.0        2
    174.0        2
    231.0        2
    2500.0       2
    292.0        2
    316.0        2
    214.0        2
    304.0        2
    188.0        2
    259.0        2
    685.0        2
    197.0        2
    455.0        2
    282.0        1
    328.0        1
    202.0        1
    698.0        1
    474.0        1
    257.0        1
    239.0        1
    468.0        1
    420.0        1
    412.0        1
    1300.0       1
    376.0        1
    525.0        1
    900.0        1
    176.0        1
    886.0        1
    790.0        1
    980.0        1
    207.0        1
    253.0        1
    540.0        1
    234.0        1
    172.0        1
    395.0        1
    222.0        1
    323.0        1
    351.0        1
    486.0        1
    258.0        1
    1200.0       1
    288.0        1
    367.0        1
    750.0        1
    425.0        1
    183.0        1
    415.0        1
    496.0        1
    357.0        1
    271.0        1
    306.0        1
    848.0        1
    780.0        1
    973.0        1
    617.0        1
    467.0        1
    252.0        1
    272.0        1
    767.0        1
    319.0        1
    177.0        1
    764.0        1
    191.0        1
    198.0        1
    410.0        1
    317.0        1
    226.0        1
    639.0        1
    630.0        1
    301.0        1
    757.0        1
    322.0        1
    385.0        1
    203.0        1
    672.0        1
    710.0        1
    2013.0       1
    209.0        1
    388.0        1
    268.0        1
    229.0        1
    281.0        1
    448.0        1
    238.0        1
    369.0        1
    428.0        1
    269.0        1
    276.0        1
    3300.0       1
    480.0        1
    212.0        1
    293.0        1
    675.0        1
    157.0        1
    273.0        1
    463.0        1
    545.0        1
    588.0        1
    206.0        1
    419.0        1
    426.0        1
    1125.0       1
    189.0        1
    499.0        1
    205.0        1
    451.0        1
    660.0        1
    355.0        1
    560.0        1
    228.0        1
    353.0        1
    335.0        1
    470.0        1
    242.0        1
    575.0        1
    574.0        1
    612.0        1
    445.0        1
    569.0        1
    932.0        1
    216.0        1
    247.0        1
    1900.0       1
    602.0        1
    820.0        1
    Name: price, dtype: int64




```python
df['price'].describe()
```




    count    120975.000000
    mean         35.363389
    std          41.022218
    min           4.000000
    25%          17.000000
    50%          25.000000
    75%          42.000000
    max        3300.000000
    Name: price, dtype: float64




```python
df['price'].median()
```




    25.0




```python
df['price'].mode()
```




    0    20.0
    dtype: float64



After viewing the descriptive statistics, I believe that the best course of action for filling the missing values is to use the median.


```python
df['price'].fillna(df['price'].median(), inplace=True)
```


```python
df.isna().sum()
```




    country                  0
    description              0
    designation              0
    points                   0
    price                    0
    province                 0
    region_1                 0
    region_2                 0
    taster_name              0
    taster_twitter_handle    0
    title                    0
    variety                  0
    winery                   0
    dtype: int64



Although neural networks and random forests are strong against outliers, the graph below shows that the data as it currently stands has a very non-normal distribution. Therefore, I will use z-score to remove the outliers because I believe it will improve potential class imbalance and model performance.


```python
df['price'].hist(bins='auto')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdcf3d77320>




```python
import numpy as np
from scipy.stats import zscore
good_rows = np.abs(zscore(df['price'].values))<=3
df[good_rows]['price'].hist(bins='auto')
df=df[good_rows]
```


![png](output_20_0.png)


Also, given the large numbers of different values, I will be binning my data into two bins, with the cutoff point being at 25 dollars. This will also make the classification process more efficient.


```python
cut_labels = [0, 1]
cut_bins = [0,25,np.inf]
ax = df['price'].hist(bins='auto')
[ax.axvline(val) for val in cut_bins]
df['price category'] = pd.cut(df['price'], bins=cut_bins, labels=cut_labels)
```


![png](output_22_0.png)



```python
df['price category'].value_counts(normalize=True)
```




    0    0.545806
    1    0.454194
    Name: price category, dtype: float64



# Text preprocessing for modeling


```python
import numpy as np
np.random.seed(0)
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import string
```

> This function will create the data in such a way that it can be used during the modeling steps.


```python
def clean_description(corpus):
    """This function will be used to clean up df['description'] so that it can be used for modeling and
    visualization.
    
    Parameters:
    
    corpus - body of text that needs to be cleaned."""
    
    #Creates an extensive stopwords list
    stopwords_list = stopwords.words('english')
    additional_punc = ['“','”','...','``',"''",'’',"'s", ' ', "n't",'wine','flavor', 'flavors']
    stopwords_list+=string.punctuation
    stopwords_list.extend(additional_punc)
    
    #Tokenizes the words in the corpus
    tokens = word_tokenize(corpus)
    
    #Uses list comprehension to create the list of clean words
    clean_words = [word.lower() for word in tokens if word.lower() not in stopwords_list]
    return clean_words
```


```python
df['clean description'] = df['description'].apply(clean_description)
```


```python
#Verify that the function was applied correctly
df['clean description'][0]
```




    ['aromas',
     'include',
     'tropical',
     'fruit',
     'broom',
     'brimstone',
     'dried',
     'herb',
     'palate',
     'overly',
     'expressive',
     'offering',
     'unripened',
     'apple',
     'citrus',
     'dried',
     'sage',
     'alongside',
     'brisk',
     'acidity']



# Modeling: Neural Network


```python
target = df['price category']
```


```python
total_vocabulary = df['clean description']
```


```python
len(total_vocabulary)
print('There are {} unique tokens in the dataset.'.format(len(total_vocabulary)))
```

    There are 128749 unique tokens in the dataset.



```python
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
```

    Using TensorFlow backend.
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:517: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:518: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:519: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:520: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:521: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])



```python
y = to_categorical(target).copy()
X = total_vocabulary.copy()
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
```


```python
tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_tr = sequence.pad_sequences(X_train_seq, maxlen=100)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_te = sequence.pad_sequences(X_test_seq, maxlen=100)
```


```python
#Created so that if validation accuracy does not improve after 2 epochs, the neural network will stop running.
#This allows for more efficiency and less memory usage.
from keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor='val_acc', patience=2)
```


```python
embedding_size = 128
model = Sequential()
model.add(Embedding(20000, embedding_size)) 
model.add(LSTM(25, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, None, 128)         2560000   
    _________________________________________________________________
    lstm_1 (LSTM)                (None, None, 25)          15400     
    _________________________________________________________________
    global_max_pooling1d_1 (Glob (None, 25)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 25)                0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 50)                1300      
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 50)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 102       
    =================================================================
    Total params: 2,576,802
    Trainable params: 2,576,802
    Non-trainable params: 0
    _________________________________________________________________



```python
model_1 = model.fit(X_tr, y_train, epochs=5, batch_size=256, validation_split=0.25, callbacks=[callback])
```

    Train on 72420 samples, validate on 24141 samples
    Epoch 1/5
    72420/72420 [==============================] - 46s 636us/step - loss: 0.5751 - acc: 0.6964 - val_loss: 0.4906 - val_acc: 0.7606
    Epoch 2/5
    72420/72420 [==============================] - 45s 619us/step - loss: 0.4738 - acc: 0.7841 - val_loss: 0.4896 - val_acc: 0.7609
    Epoch 3/5
    72420/72420 [==============================] - 46s 641us/step - loss: 0.4267 - acc: 0.8065 - val_loss: 0.5045 - val_acc: 0.7569
    Epoch 4/5
    72420/72420 [==============================] - 48s 660us/step - loss: 0.3877 - acc: 0.8264 - val_loss: 0.5260 - val_acc: 0.7623
    Epoch 5/5
    72420/72420 [==============================] - 46s 638us/step - loss: 0.3522 - acc: 0.8450 - val_loss: 0.5668 - val_acc: 0.7582



```python
y_hat_test = model.predict(X_te)
y_hat_test
```




    array([[0.21202105, 0.9614685 ],
           [0.36852294, 0.77205086],
           [0.26606086, 0.87020785],
           ...,
           [0.00421688, 0.999998  ],
           [0.9421987 , 0.05555815],
           [0.4853678 , 0.5467239 ]], dtype=float32)




```python
#Takes the above step and makes it so that we can see how the neural network predicted the classification of
#each value in the test set.
y_hat_test = y_hat_test.argmax(axis=1)
y_hat_test
```




    array([1, 1, 1, ..., 1, 0, 1])




```python
import matplotlib.pyplot as plt
def plot_confusion_matrix(conf_matrix, classes = None, normalize=True,
                          title='Confusion Matrix', cmap="Blues",
                          print_raw_matrix=False,
                          fig_size=(4,4)):
    """
    Function code borrowed with permission from
    https://github.com/jirvingphd/fsds_pt_100719_cohort_notes/blob/master/sect_42_tuning_neural_networks-Full_SG.ipynb
    
    Check if Normalization Option is Set to True. 
    If so, normalize the raw confusion matrix before visualizing
    #Other code should be equivalent to your previous function.
    Note: Taken from bs_ds and modified
    - Can pass a tuple of (y_true,y_pred) instead of conf matrix.
    """
    import itertools
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn.metrics as metrics
    
    ## make confusion matrix if given tuple of y_true,y_pred
    if isinstance(conf_matrix, tuple):
        y_true = conf_matrix[0].copy()
        y_pred = conf_matrix[1].copy()
        
        if y_true.ndim>1:
            y_true = y_true.argmax(axis=1)
        if y_pred.ndim>1:
            y_pred = y_pred.argmax(axis=1)
            
            
        cm = metrics.confusion_matrix(y_true,y_pred)
    else:
        cm = conf_matrix
        
    ## Generate integer labels for classes
    if classes is None:
        classes = list(range(len(cm)))  
        
    ## Normalize data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt='.2f'
    else:
        fmt= 'd'
        
        
    fontDict = {
        'title':{
            'fontsize':16,
            'fontweight':'semibold',
            'ha':'center',
            },
        'xlabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'ylabel':{
            'fontsize':14,
            'fontweight':'normal',
            },
        'xtick_labels':{
            'fontsize':10,
            'fontweight':'normal',
    #             'rotation':45,
            'ha':'right',
            },
        'ytick_labels':{
            'fontsize':10,
            'fontweight':'normal',
            'rotation':0,
            'ha':'right',
            },
        'data_labels':{
            'ha':'center',
            'fontweight':'semibold',

        }
    }

    # Create plot
    fig,ax = plt.subplots(figsize=fig_size)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,**fontDict['title'])
    plt.colorbar()

    tick_marks = classes#np.arange(len(classes))

    plt.xticks(tick_marks, classes, **fontDict['xtick_labels'])
    plt.yticks(tick_marks, classes,**fontDict['ytick_labels'])

    # Determine threshold for b/w text
    thresh = cm.max() / 2.

    # fig,ax = plt.subplots()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 color='darkgray',**fontDict['data_labels']) #color="white" if cm[i, j] > thresh else "black"

    plt.tight_layout()
    plt.ylabel('True label',**fontDict['ylabel'])
    plt.xlabel('Predicted label',**fontDict['xlabel'])

    if print_raw_matrix:
        print_title = 'Raw Confusion Matrix Counts:'
        print('\n',print_title)
        print(conf_matrix)


    fig = plt.gcf()
    return fig



def plot_keras_history(history,figsize_1=(6,4),
    figsize_2=(8,6)):
    """Plots keras history and returns fig"""
    
    ## Make a df from history
    if isinstance(history,dict)==False:
        history=history.history
    plot_df = pd.DataFrame(history)
    plot_df['Epoch'] = range(1,len(plot_df)+1)
    plot_df.set_index('Epoch',inplace=True)
    ## Get cols for acc vs loss
    acc_cols = list(filter(lambda x: 'acc' in x, plot_df.columns))
    loss_cols = list(filter(lambda x: 'loss' in x, plot_df.columns))   
    
    ## Set figsizes based on number of keys
    if len(acc_cols)>1:
        figsize=figsize_2
    else:
        figsize=figsize_1

    ## Make figure and axes
    fig,ax = plt.subplots(nrows=2,figsize=figsize,sharex=True)
    
    ## Plot Accuracy cols in plot 1
    plot_df[acc_cols].plot(ax=ax[0])
    ax[0].set(ylabel='Accuracy')
    ax[0].set_title('Training Results')

    ## Plot loss cols in plot 2
    plot_df[loss_cols].plot(ax=ax[1])
    ax[1].set(ylabel='Loss')
    ax[1].set_xlabel('Epoch #')


#     ## Change xaxis locators 
#     [a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(len(plot_df),integer=True)) for a in ax]
#     [a.set_xlim((1,len(plot_df)+1)) for a in ax]
    plt.tight_layout()
    
    return fig
```


```python
def evaluate_model(y_true, y_pred,history=None):
    """
    Function code borrowed with permission from
    https://github.com/jirvingphd/fsds_pt_100719_cohort_notes/blob/master/sect_39_NLP_finding_trump_SG.ipynb
    
    Evaluates neural network using sklearn metrics"""
    from sklearn import metrics
    if y_true.ndim>1:
        y_true = y_true.argmax(axis=1)
    if y_pred.ndim>1:
        y_pred = y_pred.argmax(axis=1)   
#     try:    
    if history is not None:
        plot_keras_history(history)
        plt.show()
#     except:
#         pass
    
    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)
    try:
        print(metrics.classification_report(y_true,y_pred))
        
        fig = plot_confusion_matrix((y_true,y_pred))
        plt.show()
    except Exception as e:
        print(f"[!] Error during model evaluation:\n\t{e}")


```


```python
evaluate_model(y_test, y_hat_test, model_1)
```


![png](output_45_0.png)


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               0       0.77      0.77      0.77     17568
               1       0.73      0.72      0.73     14620
    
        accuracy                           0.75     32188
       macro avg       0.75      0.75      0.75     32188
    weighted avg       0.75      0.75      0.75     32188
    



![png](output_45_2.png)


## Evaluation: The neural network was able to correctly classify the cheap wine bottles (0) 77% percent of the time and correctly identified the expensive wine bottles (1) 72% of the time. We also can see from the keras history graphs that the point of convergence did not occur too soon, despite the neural network only needing 3 epochs to process everything. Overall, I got a 75% testing accuracy, which I believe can be further improved with continued tuning.

# Modeling: Random Forest

> The question may be asked, why is a random forest being used when a neural network has already been performed for the purpose of prediction? The reason is neural networks, with all their power, are very challenging in providing meaningful insights into what their results mean, aside from a classification report. I'm running the random forest to get insights into both which words were used and the frequency of those words when making predictions. 


```python
y = df['price category'].copy()
X = df['description'].copy()
```


```python
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
```

Confirm that data in train and test group is divided similarly


```python
y_train.value_counts(normalize=True)
```




    0    0.54581
    1    0.45419
    Name: price category, dtype: float64




```python
y_test.value_counts(normalize=True)
```




    0    0.545793
    1    0.454207
    Name: price category, dtype: float64



The text data must be vectorized so that it can be usable data in this machine learning process.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
stopwords_list = stopwords.words('english')
additional_punc = ['“','”','...','``',"''",'’',"'s", 'wine', 'flavor', 'flavors']
stopwords_list+=string.punctuation
stopwords_list.extend(additional_punc)
vectorizer = TfidfVectorizer(stop_words=stopwords_list)
```


```python
tf_idf_train = vectorizer.fit_transform(X_train)
tf_idf_test = vectorizer.transform(X_test)
```

I used GridSearch in order to find the optimal hyperparameters for my random forest. The results of the gridsearch are displayed in line 15-20 in the cell below. The code was as follows:

    from sklearn.model_selection import GridSearchCV
    params  = {'criterion':['gini','entropy'],
          'max_depth':[3,5,10,50,100,None],
          'class_weight':['balanced',None],
           'bootstrap':[True ,False],
          'min_samples_leaf':[1,2,3,4],}
          
    rf_clf = RandomForestClassifier()
    grid = GridSearchCV(rf_clf,params,return_train_score=False, scoring='recall_weighted',n_jobs=-1)
    
    grid.fit(tf_idf_train,y_train)
    print(grid.best_score_)
    grid.best_params_

    Gridsearch Results
    {'bootstrap': False,
     'class_weight': None,
     'criterion': 'entropy',
     'max_depth': None,
     'min_samples_leaf': 1}


```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(criterion='entropy', bootstrap=False)
rf_clf.fit(tf_idf_train, y_train)
y_hat_train = rf_clf.predict(tf_idf_train)
y_hat_test = rf_clf.predict(tf_idf_test)
```


```python
from sklearn import metrics

def evaluate_model(y_test,y_hat_test,X_test,clf=None,
                  scoring=metrics.recall_score,verbose=False):
    
    """This function will create and return a classification report and confusion matrix.
    
    Function code borrowed with permsission from
    https://github.com/jirvingphd/fsds_pt_100719_cohort_notes/blob/master/sect_39_NLP_finding_trump_SG.ipynb
    
    Parameters:
    
    y_test - The y testing data
    
    y_hat_test - The y prediction data
    
    X_test - The X testing data
    
    clf - Name of your model
    """

    print(metrics.classification_report(y_test,y_hat_test))
    metrics.plot_confusion_matrix(clf,X_test,y_test,normalize='true',
                                 cmap='Blues')
    plt.show()
    if verbose:
        print("MODEL PARAMETERS:")
        print(pd.Series(rf.get_params()))
        
```


```python
evaluate_model(y_test, y_hat_test, tf_idf_test, rf_clf)
```

                  precision    recall  f1-score   support
    
               0       0.77      0.84      0.80     17568
               1       0.78      0.70      0.74     14620
    
        accuracy                           0.78     32188
       macro avg       0.78      0.77      0.77     32188
    weighted avg       0.78      0.78      0.78     32188
    



![png](output_60_1.png)


> Below I will discover the most important overall words for the model. This is a key insight for any recommendations.


```python
vectorizer.get_feature_names()[:10]
```




    ['000', '008', '01', '02', '03', '030', '035', '04', '05', '056']




```python
with plt.style.context('seaborn-talk'):

    importance = pd.Series(rf_clf.feature_importances_,vectorizer.get_feature_names())
    importance.sort_values().tail(20).plot(kind='barh',figsize=(10,10))
    plt.title('Most Important Words')
```


![png](output_63_0.png)


## Evaluation: In terms of classifying, we got an interesting result in that the random forest was better at predicting the classification of the cheaper bottles of wine than the neural network, but was quite a bit worse at classifying the expensive bottles. But, the real value of this model is in the graph above, where we can see which words were the most important when trying to classify each wine bottle. An important note, however, is that the graph above does not specify which words belong to which class.

# Create Meaningful Visuals

### Create frequency distributions for whole dataset and for each class

**Whole dataset**


```python
corpus = df['description']
','.join(corpus)
tokens = word_tokenize(','.join(corpus))
stopwords_list = stopwords.words('english')
additional_punc = ['“','”','...','``',"''",'’',"'s", ' ', "n't",'wine','flavor', 'flavors']
stopwords_list+=string.punctuation
stopwords_list.extend(additional_punc)
stopped_tokens = [word.lower() for word in tokens if word.lower() not in stopwords_list]
freq = FreqDist(stopped_tokens)
freq.most_common(20)
```




    [('fruit', 43391),
     ('aromas', 39242),
     ('palate', 37094),
     ('acidity', 31404),
     ('tannins', 27567),
     ('drink', 27364),
     ('cherry', 26978),
     ('ripe', 26578),
     ('black', 24934),
     ('finish', 21879),
     ('red', 18555),
     ('notes', 18256),
     ('spice', 17922),
     ('rich', 16896),
     ('nose', 16801),
     ('fresh', 16593),
     ('oak', 15655),
     ('berry', 15282),
     ('dry', 15010),
     ('plum', 13935)]



**Cheap Wine**


```python
df2 = df.loc[df['price category'] == 0]
df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
      <th>price category</th>
      <th>clean description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>Italy</td>
      <td>Aromas include tropical fruit, broom, brimston...</td>
      <td>Vulkà Bianco</td>
      <td>87</td>
      <td>25.0</td>
      <td>Sicily &amp; Sardinia</td>
      <td>Etna</td>
      <td>N/A</td>
      <td>Kerin O’Keefe</td>
      <td>@kerinokeefe</td>
      <td>Nicosia 2013 Vulkà Bianco  (Etna)</td>
      <td>White Blend</td>
      <td>Nicosia</td>
      <td>0</td>
      <td>[aromas, include, tropical, fruit, broom, brim...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>Portugal</td>
      <td>This is ripe and fruity, a wine that is smooth...</td>
      <td>Avidagos</td>
      <td>87</td>
      <td>15.0</td>
      <td>Douro</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Quinta dos Avidagos 2011 Avidagos Red (Douro)</td>
      <td>Portuguese Red</td>
      <td>Quinta dos Avidagos</td>
      <td>0</td>
      <td>[ripe, fruity, smooth, still, structured, firm...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>US</td>
      <td>Tart and snappy, the flavors of lime flesh and...</td>
      <td>N/A</td>
      <td>87</td>
      <td>14.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Rainstorm 2013 Pinot Gris (Willamette Valley)</td>
      <td>Pinot Gris</td>
      <td>Rainstorm</td>
      <td>0</td>
      <td>[tart, snappy, lime, flesh, rind, dominate, gr...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>US</td>
      <td>Pineapple rind, lemon pith and orange blossom ...</td>
      <td>Reserve Late Harvest</td>
      <td>87</td>
      <td>13.0</td>
      <td>Michigan</td>
      <td>Lake Michigan Shore</td>
      <td>N/A</td>
      <td>Alexander Peartree</td>
      <td>N/A</td>
      <td>St. Julian 2013 Reserve Late Harvest Riesling ...</td>
      <td>Riesling</td>
      <td>St. Julian</td>
      <td>0</td>
      <td>[pineapple, rind, lemon, pith, orange, blossom...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>Spain</td>
      <td>Blackberry and raspberry aromas show a typical...</td>
      <td>Ars In Vitro</td>
      <td>87</td>
      <td>15.0</td>
      <td>Northern Spain</td>
      <td>Navarra</td>
      <td>N/A</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Tandem 2011 Ars In Vitro Tempranillo-Merlot (N...</td>
      <td>Tempranillo-Merlot</td>
      <td>Tandem</td>
      <td>0</td>
      <td>[blackberry, raspberry, aromas, show, typical,...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>129956</td>
      <td>New Zealand</td>
      <td>The blend is 44% Merlot, 33% Cabernet Sauvigno...</td>
      <td>Gimblett Gravels Merlot-Cabernet Sauvignon-Malbec</td>
      <td>90</td>
      <td>19.0</td>
      <td>Hawke's Bay</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>Joe Czerwinski</td>
      <td>@JoeCz</td>
      <td>Esk Valley 2011 Gimblett Gravels Merlot-Cabern...</td>
      <td>Bordeaux-style Red Blend</td>
      <td>Esk Valley</td>
      <td>0</td>
      <td>[blend, 44, merlot, 33, cabernet, sauvignon, 2...</td>
    </tr>
    <tr>
      <td>129957</td>
      <td>Spain</td>
      <td>Lightly baked berry aromas vie for attention w...</td>
      <td>Crianza</td>
      <td>90</td>
      <td>17.0</td>
      <td>Northern Spain</td>
      <td>Rioja</td>
      <td>N/A</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Viñedos Real Rubio 2010 Crianza  (Rioja)</td>
      <td>Tempranillo Blend</td>
      <td>Viñedos Real Rubio</td>
      <td>0</td>
      <td>[lightly, baked, berry, aromas, vie, attention...</td>
    </tr>
    <tr>
      <td>129963</td>
      <td>Israel</td>
      <td>A bouquet of black cherry, tart cranberry and ...</td>
      <td>Oak Aged</td>
      <td>90</td>
      <td>20.0</td>
      <td>Galilee</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>Mike DeSimone</td>
      <td>@worldwineguys</td>
      <td>Dalton 2012 Oak Aged Cabernet Sauvignon (Galilee)</td>
      <td>Cabernet Sauvignon</td>
      <td>Dalton</td>
      <td>0</td>
      <td>[bouquet, black, cherry, tart, cranberry, clov...</td>
    </tr>
    <tr>
      <td>129964</td>
      <td>France</td>
      <td>Initially quite muted, this wine slowly develo...</td>
      <td>Domaine Saint-Rémy Herrenweg</td>
      <td>90</td>
      <td>25.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Ehrhart 2013 Domaine Saint-Rémy Herren...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Ehrhart</td>
      <td>0</td>
      <td>[initially, quite, muted, slowly, develops, im...</td>
    </tr>
    <tr>
      <td>129970</td>
      <td>France</td>
      <td>Big, rich and off-dry, this is powered by inte...</td>
      <td>Lieu-dit Harth Cuvée Caroline</td>
      <td>90</td>
      <td>21.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Schoffit 2012 Lieu-dit Harth Cuvée Car...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Schoffit</td>
      <td>0</td>
      <td>[big, rich, off-dry, powered, intense, spicine...</td>
    </tr>
  </tbody>
</table>
<p>70272 rows × 15 columns</p>
</div>




```python
corpus = df2['description']
','.join(corpus)
tokens = word_tokenize(','.join(corpus))
stopwords_list = stopwords.words('english')
additional_punc = ['“','”','...','``',"''",'’',"'s", ' ', "n't",'wine','flavor', 'flavors']
stopwords_list+=string.punctuation
stopwords_list.extend(additional_punc)
stopped_tokens_0 = [word.lower() for word in tokens if word.lower() not in stopwords_list]
freq = FreqDist(stopped_tokens_0)
freq.most_common(20)
```




    [('aromas', 22594),
     ('fruit', 22319),
     ('palate', 19588),
     ('acidity', 18595),
     ('ripe', 14005),
     ('drink', 13616),
     ('finish', 12581),
     ('tannins', 12177),
     ('cherry', 11441),
     ('fresh', 11314),
     ('black', 9730),
     ('notes', 9663),
     ('red', 9650),
     ('berry', 8993),
     ('crisp', 8810),
     ('dry', 8807),
     ('nose', 8793),
     ('spice', 8692),
     ('apple', 8513),
     ('fruits', 8149)]



**Expensive wine**


```python
df3 = df.loc[df['price category'] == 1]
df3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>description</th>
      <th>designation</th>
      <th>points</th>
      <th>price</th>
      <th>province</th>
      <th>region_1</th>
      <th>region_2</th>
      <th>taster_name</th>
      <th>taster_twitter_handle</th>
      <th>title</th>
      <th>variety</th>
      <th>winery</th>
      <th>price category</th>
      <th>clean description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>4</td>
      <td>US</td>
      <td>Much like the regular bottling from 2012, this...</td>
      <td>Vintner's Reserve Wild Child Block</td>
      <td>87</td>
      <td>65.0</td>
      <td>Oregon</td>
      <td>Willamette Valley</td>
      <td>Willamette Valley</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Sweet Cheeks 2012 Vintner's Reserve Wild Child...</td>
      <td>Pinot Noir</td>
      <td>Sweet Cheeks</td>
      <td>1</td>
      <td>[much, like, regular, bottling, 2012, comes, a...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>France</td>
      <td>This has great depth of flavor with its fresh ...</td>
      <td>Les Natures</td>
      <td>87</td>
      <td>27.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Jean-Baptiste Adam 2012 Les Natures Pinot Gris...</td>
      <td>Pinot Gris</td>
      <td>Jean-Baptiste Adam</td>
      <td>1</td>
      <td>[great, depth, fresh, apple, pear, fruits, tou...</td>
    </tr>
    <tr>
      <td>11</td>
      <td>France</td>
      <td>This is a dry wine, very spicy, with a tight, ...</td>
      <td>N/A</td>
      <td>87</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Leon Beyer 2012 Gewurztraminer (Alsace)</td>
      <td>Gewürztraminer</td>
      <td>Leon Beyer</td>
      <td>1</td>
      <td>[dry, spicy, tight, taut, texture, strongly, m...</td>
    </tr>
    <tr>
      <td>12</td>
      <td>US</td>
      <td>Slightly reduced, this wine offers a chalky, t...</td>
      <td>N/A</td>
      <td>87</td>
      <td>34.0</td>
      <td>California</td>
      <td>Alexander Valley</td>
      <td>Sonoma</td>
      <td>Virginie Boone</td>
      <td>@vboone</td>
      <td>Louis M. Martini 2012 Cabernet Sauvignon (Alex...</td>
      <td>Cabernet Sauvignon</td>
      <td>Louis M. Martini</td>
      <td>1</td>
      <td>[slightly, reduced, offers, chalky, tannic, ba...</td>
    </tr>
    <tr>
      <td>16</td>
      <td>Argentina</td>
      <td>Baked plum, molasses, balsamic vinegar and che...</td>
      <td>Felix</td>
      <td>87</td>
      <td>30.0</td>
      <td>Other</td>
      <td>Cafayate</td>
      <td>N/A</td>
      <td>Michael Schachner</td>
      <td>@wineschach</td>
      <td>Felix Lavaque 2010 Felix Malbec (Cafayate)</td>
      <td>Malbec</td>
      <td>Felix Lavaque</td>
      <td>1</td>
      <td>[baked, plum, molasses, balsamic, vinegar, che...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>129965</td>
      <td>France</td>
      <td>While it's rich, this beautiful dry wine also ...</td>
      <td>Seppi Landmann Vallée Noble</td>
      <td>90</td>
      <td>28.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Rieflé-Landmann 2013 Seppi Landmann Va...</td>
      <td>Pinot Gris</td>
      <td>Domaine Rieflé-Landmann</td>
      <td>1</td>
      <td>[rich, beautiful, dry, also, offers, considera...</td>
    </tr>
    <tr>
      <td>129966</td>
      <td>Germany</td>
      <td>Notes of honeysuckle and cantaloupe sweeten th...</td>
      <td>Brauneberger Juffer-Sonnenuhr Spätlese</td>
      <td>90</td>
      <td>28.0</td>
      <td>Mosel</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>Anna Lee C. Iijima</td>
      <td>N/A</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef) 2013 ...</td>
      <td>Riesling</td>
      <td>Dr. H. Thanisch (Erben Müller-Burggraef)</td>
      <td>1</td>
      <td>[notes, honeysuckle, cantaloupe, sweeten, deli...</td>
    </tr>
    <tr>
      <td>129967</td>
      <td>US</td>
      <td>Citation is given as much as a decade of bottl...</td>
      <td>N/A</td>
      <td>90</td>
      <td>75.0</td>
      <td>Oregon</td>
      <td>Oregon</td>
      <td>Oregon Other</td>
      <td>Paul Gregutt</td>
      <td>@paulgwine</td>
      <td>Citation 2004 Pinot Noir (Oregon)</td>
      <td>Pinot Noir</td>
      <td>Citation</td>
      <td>1</td>
      <td>[citation, given, much, decade, bottle, age, p...</td>
    </tr>
    <tr>
      <td>129968</td>
      <td>France</td>
      <td>Well-drained gravel soil gives this wine its c...</td>
      <td>Kritt</td>
      <td>90</td>
      <td>30.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Gresser 2013 Kritt Gewurztraminer (Als...</td>
      <td>Gewürztraminer</td>
      <td>Domaine Gresser</td>
      <td>1</td>
      <td>[well-drained, gravel, soil, gives, crisp, dry...</td>
    </tr>
    <tr>
      <td>129969</td>
      <td>France</td>
      <td>A dry style of Pinot Gris, this is crisp with ...</td>
      <td>N/A</td>
      <td>90</td>
      <td>32.0</td>
      <td>Alsace</td>
      <td>Alsace</td>
      <td>N/A</td>
      <td>Roger Voss</td>
      <td>@vossroger</td>
      <td>Domaine Marcel Deiss 2012 Pinot Gris (Alsace)</td>
      <td>Pinot Gris</td>
      <td>Domaine Marcel Deiss</td>
      <td>1</td>
      <td>[dry, style, pinot, gris, crisp, acidity, also...</td>
    </tr>
  </tbody>
</table>
<p>58477 rows × 15 columns</p>
</div>




```python
corpus = df3['description']
','.join(corpus)
tokens = word_tokenize(','.join(corpus))
stopwords_list = stopwords.words('english')
additional_punc = ['“','”','...','``',"''",'’',"'s", ' ', "n't",'wine','flavor', 'flavors']
stopwords_list+=string.punctuation
stopwords_list.extend(additional_punc)
stopped_tokens_1 = [word.lower() for word in tokens if word.lower() not in stopwords_list]
freq = FreqDist(stopped_tokens_1)
freq.most_common(20)
```




    [('fruit', 21072),
     ('palate', 17506),
     ('aromas', 16648),
     ('cherry', 15537),
     ('tannins', 15390),
     ('black', 15204),
     ('drink', 13748),
     ('acidity', 12809),
     ('ripe', 12573),
     ('oak', 9818),
     ('rich', 9394),
     ('finish', 9298),
     ('spice', 9230),
     ('red', 8905),
     ('notes', 8593),
     ('nose', 8008),
     ('blackberry', 7334),
     ('cabernet', 6860),
     ('dark', 6822),
     ('blend', 6469)]



### Create Wordclouds & Bigrams

> I created two functions that will generate wordclouds and bigrams that can be used to make for the whole dataset and for each classification.


```python
from wordcloud import WordCloud
def create_wordcloud(tokens):
    """
    Create a WordCloud
    
    Parameters:
    
    tokens - Some form of tokenized text data
    
    """
    
    wordcloud = WordCloud(stopwords='stopwords_list', collocations=False, max_words=40)
    wordcloud.generate(','.join(tokens))
    plt.figure(figsize = (12, 12), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis('off')
```


```python
create_wordcloud(stopped_tokens)
```


![png](output_78_0.png)



```python
create_wordcloud(stopped_tokens_0)
```


![png](output_79_0.png)



```python
create_wordcloud(stopped_tokens_1)
```


![png](output_80_0.png)



```python
import nltk
def create_bigram(token_data):
    """
    Create a Bigram
    
    Parameters:
    
    tokens - tokenized data
    
    """
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    word_finder = nltk.BigramCollocationFinder.from_words(token_data)

    words_scored = word_finder.score_ngrams(bigram_measures.raw_freq)
    top_words = pd.DataFrame.from_records(words_scored,columns=['Words','Frequency']).head(20)
    return top_words
```


```python
create_bigram(stopped_tokens).style.hide_index().set_caption('All Reviews')
```




<style  type="text/css" >
</style><table id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65e" ><caption>All Reviews</caption><thead>    <tr>        <th class="col_heading level0 col0" >Words</th>        <th class="col_heading level0 col1" >Frequency</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow0_col0" class="data row0 col0" >('black', 'cherry')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow0_col1" class="data row0 col1" >0.00237452</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow1_col0" class="data row1 col0" >('drink', 'now.')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow1_col1" class="data row1 col1" >0.00186115</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow2_col0" class="data row2 col0" >('cabernet', 'sauvignon')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow2_col1" class="data row2 col1" >0.00158051</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow3_col0" class="data row3 col0" >('palate', 'offers')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow3_col1" class="data row3 col1" >0.00137223</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow4_col0" class="data row4 col0" >('pinot', 'noir')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow4_col1" class="data row4 col1" >0.000983706</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow5_col0" class="data row5 col0" >('black', 'pepper')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow5_col1" class="data row5 col1" >0.000848764</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow6_col0" class="data row6 col0" >('white', 'pepper')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow6_col1" class="data row6 col1" >0.000792376</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow7_col0" class="data row7 col0" >('black', 'currant')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow7_col1" class="data row7 col1" >0.000781619</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow8_col0" class="data row8 col0" >('nose', 'palate')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow8_col1" class="data row8 col1" >0.000757499</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow9_col0" class="data row9 col0" >('finish', 'drink')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow9_col1" class="data row9 col1" >0.000749677</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow10_col0" class="data row10 col0" >('black', 'fruit')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow10_col1" class="data row10 col1" >0.000748047</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow11_col0" class="data row11 col0" >('cabernet', 'franc')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow11_col1" class="data row11 col1" >0.000748047</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow12_col0" class="data row12 col0" >('ready', 'drink')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow12_col1" class="data row12 col1" >0.000724905</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow13_col0" class="data row13 col0" >('red', 'cherry')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow13_col1" class="data row13 col1" >0.000712845</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow14_col0" class="data row14 col0" >('red', 'berry')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow14_col1" class="data row14 col1" >0.000699807</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow15_col0" class="data row15 col0" >('firm', 'tannins')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow15_col1" class="data row15 col1" >0.000693288</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow16_col0" class="data row16 col0" >('palate', 'delivers')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow16_col1" class="data row16 col1" >0.000662323</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow17_col0" class="data row17 col0" >('black', 'fruits')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow17_col1" class="data row17 col1" >0.000657108</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow18_col0" class="data row18 col0" >('green', 'apple')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow18_col1" class="data row18 col1" >0.000653848</td>
            </tr>
            <tr>
                                <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow19_col0" class="data row19 col0" >('drink', '2018.')</td>
                        <td id="T_9e35ccc8_bcf2_11ea_8338_a683e7b2e65erow19_col1" class="data row19 col1" >0.000631684</td>
            </tr>
    </tbody></table>




```python
create_bigram(stopped_tokens_0).style.hide_index().set_caption('Cheap Wine Reviews')
```




<style  type="text/css" >
</style><table id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65e" ><caption>Cheap Wine Reviews</caption><thead>    <tr>        <th class="col_heading level0 col0" >Words</th>        <th class="col_heading level0 col1" >Frequency</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow0_col0" class="data row0 col0" >('drink', 'now.')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow0_col1" class="data row0 col1" >0.00258455</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow1_col0" class="data row1 col0" >('black', 'cherry')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow1_col1" class="data row1 col1" >0.00175851</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow2_col0" class="data row2 col0" >('palate', 'offers')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow2_col1" class="data row2 col1" >0.00136917</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow3_col0" class="data row3 col0" >('cabernet', 'sauvignon')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow3_col1" class="data row3 col1" >0.00103953</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow4_col0" class="data row4 col0" >('sauvignon', 'blanc')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow4_col1" class="data row4 col1" >0.000949335</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow5_col0" class="data row5 col0" >('ready', 'drink')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow5_col1" class="data row5 col1" >0.000945441</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow6_col0" class="data row6 col0" >('green', 'apple')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow6_col1" class="data row6 col1" >0.000941548</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow7_col0" class="data row7 col0" >('ready', 'drink.')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow7_col1" class="data row7 col1" >0.000891583</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow8_col0" class="data row8 col0" >('stone', 'fruit')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow8_col1" class="data row8 col1" >0.000837076</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow9_col0" class="data row9 col0" >('red', 'berry')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow9_col1" class="data row9 col1" >0.00082864</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow10_col0" class="data row10 col0" >('berry', 'fruits')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow10_col1" class="data row10 col1" >0.000827342</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow11_col0" class="data row11 col0" >('nose', 'palate')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow11_col1" class="data row11 col1" >0.000815662</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow12_col0" class="data row12 col0" >('crisp', 'acidity')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow12_col1" class="data row12 col1" >0.000795546</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow13_col0" class="data row13 col0" >('tropical', 'fruit')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow13_col1" class="data row13 col1" >0.000774133</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow14_col0" class="data row14 col0" >('finish', 'drink')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow14_col1" class="data row14 col1" >0.000761155</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow15_col0" class="data row15 col0" >('black', 'fruits')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow15_col1" class="data row15 col1" >0.000741039</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow16_col0" class="data row16 col0" >('red', 'fruits')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow16_col1" class="data row16 col1" >0.000731306</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow17_col0" class="data row17 col0" >('white', 'peach')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow17_col1" class="data row17 col1" >0.000726115</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow18_col0" class="data row18 col0" >('black', 'currant')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow18_col1" class="data row18 col1" >0.000713137</td>
            </tr>
            <tr>
                                <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow19_col0" class="data row19 col0" >('pinot', 'noir')</td>
                        <td id="T_d796d9f8_bcf2_11ea_903e_a683e7b2e65erow19_col1" class="data row19 col1" >0.000712488</td>
            </tr>
    </tbody></table>




```python
create_bigram(stopped_tokens_1).style.hide_index().set_caption('Expensive Wine Reviews')
```




<style  type="text/css" >
</style><table id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65e" ><caption>Expensive Wine Reviews</caption><thead>    <tr>        <th class="col_heading level0 col0" >Words</th>        <th class="col_heading level0 col1" >Frequency</th>    </tr></thead><tbody>
                <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow0_col0" class="data row0 col0" >('black', 'cherry')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow0_col1" class="data row0 col1" >0.00299625</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow1_col0" class="data row1 col0" >('cabernet', 'sauvignon')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow1_col1" class="data row1 col1" >0.00212652</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow2_col0" class="data row2 col0" >('palate', 'offers')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow2_col1" class="data row2 col1" >0.00137533</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow3_col0" class="data row3 col0" >('pinot', 'noir')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow3_col1" class="data row3 col1" >0.00125744</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow4_col0" class="data row4 col0" >('black', 'pepper')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow4_col1" class="data row4 col1" >0.00120177</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow5_col0" class="data row5 col0" >('drink', 'now.')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow5_col1" class="data row5 col1" >0.00113104</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow6_col0" class="data row6 col0" >('cabernet', 'franc')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow6_col1" class="data row6 col1" >0.00107275</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow7_col0" class="data row7 col0" >('petit', 'verdot')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow7_col1" class="data row7 col1" >0.000899201</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow8_col0" class="data row8 col0" >('white', 'pepper')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow8_col1" class="data row8 col1" >0.000873004</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow9_col0" class="data row9 col0" >('black', 'currant')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow9_col1" class="data row9 col1" >0.000850737</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow10_col0" class="data row10 col0" >('black', 'fruit')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow10_col1" class="data row10 col1" >0.000840913</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow11_col0" class="data row11 col0" >('french', 'oak')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow11_col1" class="data row11 col1" >0.000798999</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow12_col0" class="data row12 col0" >('palate', 'delivers')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow12_col1" class="data row12 col1" >0.000798999</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow13_col0" class="data row13 col0" >('red', 'cherry')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow13_col1" class="data row13 col1" >0.000784591</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow14_col0" class="data row14 col0" >('firm', 'tannins')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow14_col1" class="data row14 col1" >0.000755119</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow15_col0" class="data row15 col0" >('cherry', 'fruit')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow15_col1" class="data row15 col1" >0.00075119</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow16_col0" class="data row16 col0" >('dark', 'chocolate')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow16_col1" class="data row16 col1" >0.000740056</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow17_col0" class="data row17 col0" >('finish', 'drink')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow17_col1" class="data row17 col1" >0.000738091</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow18_col0" class="data row18 col0" >('nose', 'palate')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow18_col1" class="data row18 col1" >0.000698796</td>
            </tr>
            <tr>
                                <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow19_col0" class="data row19 col0" >('full', 'bodied')</td>
                        <td id="T_05a33e54_bcf3_11ea_98b3_a683e7b2e65erow19_col1" class="data row19 col1" >0.000646403</td>
            </tr>
    </tbody></table>




```python
def transform_format(val):
    if val == 0:
        return 255
    else:
        return 0
    

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
wine_mask = np.array(Image.open(path.join(d,'wine glass mask.png')))


transformed_wine_mask = wine_mask#np.ndarray((wine_mask.shape[0],wine_mask.shape[1]), np.int32)


for i in range(len(wine_mask)):
    transformed_wine_mask[i] = list(map(transform_format, wine_mask[i]))
    
```


```python
transformed_wine_mask
plt.imshow(transformed_wine_mask, cmap='gray')
```


```python
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
wine_mask = np.array(Image.open(path.join(d,'wine glass mask.png')))
wc = WordCloud(background_color='white', stopwords='stopwords_list', collocations=False, mask=transformed_wine_mask,
               contour_color='steelblue')
wc.generate(','.join(stopped_tokens))
wc.to_file(path.join(d, 'wine.png'))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()
```

# Recommendations

> Wineries can take these words and use them to create unique scents, aromas, and tastes in either their pre-existing products or any new ones they may create. They can also boost their products they already have in their marketing knowing that there are specific words that are unqiue to the price point of certain bottles of wine. Words from expensive wines can be used to make cheaper wines seem more valuable, along with words from cheaper wines can be used to make expensive wines more appealing or reach a wider audience.

# Conclusion

> I was able to successfully classify our wine data by its two groups, cheap or expensive, with 77% of the cheap wine being correctly classified and 72% of the expensive wine being correctly classified, while having an overall testing accuarcy of 75%. I was also able to identify that each group contains unique keywords, such as "fresh, berry, & crisp" for the cheap wines, and "oak, rich, & blackberry" for the expensive wines. I believe that these results can provide a lot of unique insight to wineries, especially considering that these unique results came from the thoughts of professional wine reviewers.


```python

```
