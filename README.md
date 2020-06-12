
# Creating an NLP Neural Network to Predict Price of Wine

## Flatiron School - Cohort 100719PT
## Instructor - James Irving
## By Acusio Bivona


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




```python
cut_labels = [1, 2, 3, 4]
cut_bins = [0,10,50,200,3300]
df['Price Category'] = pd.cut(df['price'], bins=cut_bins, labels=cut_labels)
```


```python
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
      <th>Price Category</th>
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
      <td>2</td>
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
      <td>2</td>
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
      <td>2</td>
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
      <td>2</td>
    </tr>
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
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Price Category'].value_counts()
```




    2    103917
    3     19092
    1      6280
    4       682
    Name: Price Category, dtype: int64




```python
df['Price Category'].dtype
```




    CategoricalDtype(categories=[1, 2, 3, 4], ordered=True)




```python
df['Price Category'] = df['Price Category'].astype('int64')
```


```python
df.dtypes
```




    country                   object
    description               object
    designation               object
    points                     int64
    price                    float64
    province                  object
    region_1                  object
    region_2                  object
    taster_name               object
    taster_twitter_handle     object
    title                     object
    variety                   object
    winery                    object
    Price Category             int64
    dtype: object




```python
import matplotlib.pyplot as plt
pc_hist = plt.hist(x=df['Price Category'])
pc_hist.show()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-21-c020b857b894> in <module>
          1 import matplotlib.pyplot as plt
          2 pc_hist = plt.hist(x=df['Price Category'])
    ----> 3 pc_hist.show()
    

    AttributeError: 'tuple' object has no attribute 'show'



![png](output_21_1.png)


# Fun with Word Embeddings


```python
import numpy as np
np.random.seed(0)
from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import string
```


```python
#Model 1
data = df['description'].map(word_tokenize)
```


```python
#Model 1
data[:5]
```




    0    [Aromas, include, tropical, fruit, ,, broom, ,...
    1    [This, is, ripe, and, fruity, ,, a, wine, that...
    2    [Tart, and, snappy, ,, the, flavors, of, lime,...
    3    [Pineapple, rind, ,, lemon, pith, and, orange,...
    4    [Much, like, the, regular, bottling, from, 201...
    Name: description, dtype: object




```python
#Model 2
corpus = df['description'].to_list()
corpus[:10]
```




    ["Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, offering unripened apple, citrus and dried sage alongside brisk acidity.",
     "This is ripe and fruity, a wine that is smooth while still structured. Firm tannins are filled out with juicy red berry fruits and freshened with acidity. It's  already drinkable, although it will certainly be better from 2016.",
     'Tart and snappy, the flavors of lime flesh and rind dominate. Some green pineapple pokes through, with crisp acidity underscoring the flavors. The wine was all stainless-steel fermented.',
     'Pineapple rind, lemon pith and orange blossom start off the aromas. The palate is a bit more opulent, with notes of honey-drizzled guava and mango giving way to a slightly astringent, semidry finish.',
     "Much like the regular bottling from 2012, this comes across as rather rough and tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew.",
     'Blackberry and raspberry aromas show a typical Navarran whiff of green herbs and, in this case, horseradish. In the mouth, this is fairly full bodied, with tomatoey acidity. Spicy, herbal flavors complement dark plum fruit, while the finish is fresh but grabby.',
     "Here's a bright, informal red that opens with aromas of candied berry, white pepper and savory herb that carry over to the palate. It's balanced with fresh acidity and soft tannins.",
     "This dry and restrained wine offers spice in profusion. Balanced with acidity and a firm texture, it's very much for food.",
     "Savory dried thyme notes accent sunnier flavors of preserved peach in this brisk, off-dry wine. It's fruity and fresh, with an elegant, sprightly footprint.",
     "This has great depth of flavor with its fresh apple and pear fruits and touch of spice. It's off dry while balanced with acidity and a crisp texture. Drink now."]




```python
#Model 2
','.join(corpus)[:100]
```




    "Aromas include tropical fruit, broom, brimstone and dried herb. The palate isn't overly expressive, "




```python
#Model 2
freq = FreqDist(','.join(corpus))
freq.most_common(100)
```




    [(' ', 5118835),
     ('e', 2679488),
     ('a', 2197359),
     ('t', 2037048),
     ('i', 1996023),
     ('n', 1802979),
     ('r', 1761599),
     ('s', 1654735),
     ('o', 1553393),
     ('l', 1186824),
     ('h', 1150459),
     ('d', 1012526),
     ('c', 773570),
     ('f', 708088),
     ('u', 643567),
     (',', 570663),
     ('y', 568010),
     ('p', 552391),
     ('m', 475065),
     ('g', 461831),
     ('b', 442672),
     ('w', 441606),
     ('.', 356148),
     ('v', 261179),
     ('k', 224636),
     ('T', 115717),
     ('-', 76476),
     ("'", 58755),
     ('x', 56419),
     ('I', 52415),
     ('0', 49078),
     ('A', 44727),
     ('2', 43243),
     ('S', 42839),
     ('C', 40715),
     ('j', 32435),
     ('D', 31837),
     ('1', 28137),
     ('M', 26631),
     ('B', 24821),
     ('F', 23260),
     ('P', 23235),
     ('z', 20012),
     ('%', 19244),
     ('R', 17730),
     ('q', 15952),
     ('G', 13576),
     ('W', 12670),
     ('V', 12445),
     ('N', 11431),
     ('L', 11052),
     ('5', 10236),
     ('H', 8314),
     ('8', 8177),
     ('O', 7839),
     ('3', 7122),
     ('7', 6481),
     ('E', 6455),
     ('9', 6377),
     ('–', 6334),
     ('6', 6088),
     ('é', 5875),
     ('4', 5629),
     ('—', 4376),
     (')', 4315),
     ('(', 4307),
     (';', 2996),
     ('J', 2968),
     ('Z', 2912),
     ('Y', 2599),
     ('è', 2263),
     ('K', 1641),
     (':', 1438),
     ('U', 1420),
     ('ô', 969),
     ('/', 818),
     ('ü', 802),
     ('Q', 774),
     ('”', 712),
     ('“', 708),
     ('â', 604),
     ('?', 482),
     ('ã', 425),
     ('ñ', 377),
     ('$', 347),
     ('’', 328),
     ('û', 300),
     ('ä', 283),
     ('í', 140),
     ('!', 139),
     ('&', 137),
     ('X', 132),
     ('ç', 131),
     ('É', 73),
     ('à', 69),
     ('"', 56),
     ('á', 52),
     ('ö', 50),
     ('‘', 49),
     ('ê', 46)]




```python
#Model 2
tokens = word_tokenize(','.join(corpus))
tokens[:10]
```




    ['Aromas',
     'include',
     'tropical',
     'fruit',
     ',',
     'broom',
     ',',
     'brimstone',
     'and',
     'dried']




```python
#Model 2
freq = FreqDist(tokens)
freq.most_common(100)
```




    [(',', 569434),
     ('and', 347136),
     ('.', 224766),
     ('of', 172862),
     ('the', 168325),
     ('a', 157612),
     ('with', 115792),
     ('is', 97293),
     ('wine', 78057),
     ('this', 72852),
     ('in', 60103),
     ('flavors', 57513),
     ('to', 55303),
     ('The', 52649),
     ("'s", 51341),
     ('fruit', 43608),
     ('It', 43567),
     ('on', 42999),
     ('it', 42026),
     ('This', 41120),
     ('that', 39338),
     ('palate', 37377),
     ('aromas', 35358),
     ('acidity', 31311),
     ('from', 30013),
     ('but', 29408),
     ('tannins', 27747),
     ('cherry', 26858),
     ('are', 25950),
     ('ripe', 24845),
     ('has', 24599),
     ('black', 24415),
     ('finish', 22083),
     ('A', 21855),
     ('for', 20743),
     ('by', 20593),
     ('Drink', 20406),
     ('%', 19244),
     ('notes', 17933),
     ('spice', 17842),
     ('red', 17826),
     ('as', 17306),
     ('nose', 16862),
     ('its', 16362),
     ('rich', 15963),
     ('an', 15757),
     ('oak', 15432),
     ('berry', 15241),
     ('fresh', 15132),
     ('dry', 13724),
     ('plum', 13714),
     ('fruits', 13286),
     ('blend', 12957),
     ('finish.', 12830),
     ('offers', 12555),
     ('apple', 12397),
     ('blackberry', 12175),
     ('soft', 11955),
     ('white', 11951),
     ('crisp', 11628),
     ('sweet', 11553),
     ('texture', 11544),
     ('through', 11260),
     ('citrus', 10920),
     ('shows', 10558),
     ('Cabernet', 10447),
     ('vanilla', 10196),
     ('dark', 10112),
     ('well', 10035),
     ('while', 10015),
     ('bright', 9721),
     ('light', 9717),
     ('at', 9579),
     ('more', 9535),
     ('pepper', 9251),
     ('full', 9142),
     ('juicy', 9058),
     ('raspberry', 9015),
     ('fruity', 8996),
     ('very', 8698),
     ('green', 8597),
     ('good', 8590),
     ('some', 8487),
     ('firm', 8453),
     ('touch', 8446),
     ('peach', 8234),
     ('lemon', 8128),
     ('now', 8029),
     ('will', 7930),
     ('chocolate', 7829),
     ('character', 7798),
     ('There', 7653),
     ('pear', 7566),
     ('dried', 7559),
     ('drink', 7462),
     ('Sauvignon', 7405),
     ('or', 7396),
     ('balanced', 7369),
     ('out', 7339),
     ('be', 7233)]




```python
#Model 2
stopwords_list = stopwords.words('english')
stopwords_list
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     "she's",
     'her',
     'hers',
     'herself',
     'it',
     "it's",
     'its',
     'itself',
     'they',
     'them',
     'their',
     'theirs',
     'themselves',
     'what',
     'which',
     'who',
     'whom',
     'this',
     'that',
     "that'll",
     'these',
     'those',
     'am',
     'is',
     'are',
     'was',
     'were',
     'be',
     'been',
     'being',
     'have',
     'has',
     'had',
     'having',
     'do',
     'does',
     'did',
     'doing',
     'a',
     'an',
     'the',
     'and',
     'but',
     'if',
     'or',
     'because',
     'as',
     'until',
     'while',
     'of',
     'at',
     'by',
     'for',
     'with',
     'about',
     'against',
     'between',
     'into',
     'through',
     'during',
     'before',
     'after',
     'above',
     'below',
     'to',
     'from',
     'up',
     'down',
     'in',
     'out',
     'on',
     'off',
     'over',
     'under',
     'again',
     'further',
     'then',
     'once',
     'here',
     'there',
     'when',
     'where',
     'why',
     'how',
     'all',
     'any',
     'both',
     'each',
     'few',
     'more',
     'most',
     'other',
     'some',
     'such',
     'no',
     'nor',
     'not',
     'only',
     'own',
     'same',
     'so',
     'than',
     'too',
     'very',
     's',
     't',
     'can',
     'will',
     'just',
     'don',
     "don't",
     'should',
     "should've",
     'now',
     'd',
     'll',
     'm',
     'o',
     're',
     've',
     'y',
     'ain',
     'aren',
     "aren't",
     'couldn',
     "couldn't",
     'didn',
     "didn't",
     'doesn',
     "doesn't",
     'hadn',
     "hadn't",
     'hasn',
     "hasn't",
     'haven',
     "haven't",
     'isn',
     "isn't",
     'ma',
     'mightn',
     "mightn't",
     'mustn',
     "mustn't",
     'needn',
     "needn't",
     'shan',
     "shan't",
     'shouldn',
     "shouldn't",
     'wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't"]




```python
#Model 2
additional_punc = ['“','”','...','``',"''",'’',"'s"]
```


```python
#Model 2
stopwords_list+=string.punctuation
stopwords_list.extend(additional_punc)
stopwords_list
```




    ['i',
     'me',
     'my',
     'myself',
     'we',
     'our',
     'ours',
     'ourselves',
     'you',
     "you're",
     "you've",
     "you'll",
     "you'd",
     'your',
     'yours',
     'yourself',
     'yourselves',
     'he',
     'him',
     'his',
     'himself',
     'she',
     "she's",
     'her',
     'hers',
     'herself',
     'it',
     "it's",
     'its',
     'itself',
     'they',
     'them',
     'their',
     'theirs',
     'themselves',
     'what',
     'which',
     'who',
     'whom',
     'this',
     'that',
     "that'll",
     'these',
     'those',
     'am',
     'is',
     'are',
     'was',
     'were',
     'be',
     'been',
     'being',
     'have',
     'has',
     'had',
     'having',
     'do',
     'does',
     'did',
     'doing',
     'a',
     'an',
     'the',
     'and',
     'but',
     'if',
     'or',
     'because',
     'as',
     'until',
     'while',
     'of',
     'at',
     'by',
     'for',
     'with',
     'about',
     'against',
     'between',
     'into',
     'through',
     'during',
     'before',
     'after',
     'above',
     'below',
     'to',
     'from',
     'up',
     'down',
     'in',
     'out',
     'on',
     'off',
     'over',
     'under',
     'again',
     'further',
     'then',
     'once',
     'here',
     'there',
     'when',
     'where',
     'why',
     'how',
     'all',
     'any',
     'both',
     'each',
     'few',
     'more',
     'most',
     'other',
     'some',
     'such',
     'no',
     'nor',
     'not',
     'only',
     'own',
     'same',
     'so',
     'than',
     'too',
     'very',
     's',
     't',
     'can',
     'will',
     'just',
     'don',
     "don't",
     'should',
     "should've",
     'now',
     'd',
     'll',
     'm',
     'o',
     're',
     've',
     'y',
     'ain',
     'aren',
     "aren't",
     'couldn',
     "couldn't",
     'didn',
     "didn't",
     'doesn',
     "doesn't",
     'hadn',
     "hadn't",
     'hasn',
     "hasn't",
     'haven',
     "haven't",
     'isn',
     "isn't",
     'ma',
     'mightn',
     "mightn't",
     'mustn',
     "mustn't",
     'needn',
     "needn't",
     'shan',
     "shan't",
     'shouldn',
     "shouldn't",
     'wasn',
     "wasn't",
     'weren',
     "weren't",
     'won',
     "won't",
     'wouldn',
     "wouldn't",
     '!',
     '"',
     '#',
     '$',
     '%',
     '&',
     "'",
     '(',
     ')',
     '*',
     '+',
     ',',
     '-',
     '.',
     '/',
     ':',
     ';',
     '<',
     '=',
     '>',
     '?',
     '@',
     '[',
     '\\',
     ']',
     '^',
     '_',
     '`',
     '{',
     '|',
     '}',
     '~',
     '“',
     '”',
     '...',
     '``',
     "''",
     '’',
     "'s"]




```python
#Model 2
stopped_tokens = [word.lower() for word in tokens if word.lower() not in stopwords_list]
stopped_tokens
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
     "n't",
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
     'acidity.',
     'ripe',
     'fruity',
     'wine',
     'smooth',
     'still',
     'structured',
     'firm',
     'tannins',
     'filled',
     'juicy',
     'red',
     'berry',
     'fruits',
     'freshened',
     'acidity',
     'already',
     'drinkable',
     'although',
     'certainly',
     'better',
     '2016.',
     'tart',
     'snappy',
     'flavors',
     'lime',
     'flesh',
     'rind',
     'dominate',
     'green',
     'pineapple',
     'pokes',
     'crisp',
     'acidity',
     'underscoring',
     'flavors',
     'wine',
     'stainless-steel',
     'fermented.',
     'pineapple',
     'rind',
     'lemon',
     'pith',
     'orange',
     'blossom',
     'start',
     'aromas',
     'palate',
     'bit',
     'opulent',
     'notes',
     'honey-drizzled',
     'guava',
     'mango',
     'giving',
     'way',
     'slightly',
     'astringent',
     'semidry',
     'finish.',
     'much',
     'like',
     'regular',
     'bottling',
     '2012',
     'comes',
     'across',
     'rather',
     'rough',
     'tannic',
     'rustic',
     'earthy',
     'herbal',
     'characteristics',
     'nonetheless',
     'think',
     'pleasantly',
     'unfussy',
     'country',
     'wine',
     'good',
     'companion',
     'hearty',
     'winter',
     'stew.',
     'blackberry',
     'raspberry',
     'aromas',
     'show',
     'typical',
     'navarran',
     'whiff',
     'green',
     'herbs',
     'case',
     'horseradish',
     'mouth',
     'fairly',
     'full',
     'bodied',
     'tomatoey',
     'acidity',
     'spicy',
     'herbal',
     'flavors',
     'complement',
     'dark',
     'plum',
     'fruit',
     'finish',
     'fresh',
     'grabby.',
     'bright',
     'informal',
     'red',
     'opens',
     'aromas',
     'candied',
     'berry',
     'white',
     'pepper',
     'savory',
     'herb',
     'carry',
     'palate',
     'balanced',
     'fresh',
     'acidity',
     'soft',
     'tannins.',
     'dry',
     'restrained',
     'wine',
     'offers',
     'spice',
     'profusion',
     'balanced',
     'acidity',
     'firm',
     'texture',
     'much',
     'food.',
     'savory',
     'dried',
     'thyme',
     'notes',
     'accent',
     'sunnier',
     'flavors',
     'preserved',
     'peach',
     'brisk',
     'off-dry',
     'wine',
     'fruity',
     'fresh',
     'elegant',
     'sprightly',
     'footprint.',
     'great',
     'depth',
     'flavor',
     'fresh',
     'apple',
     'pear',
     'fruits',
     'touch',
     'spice',
     'dry',
     'balanced',
     'acidity',
     'crisp',
     'texture',
     'drink',
     'now.',
     'soft',
     'supple',
     'plum',
     'envelopes',
     'oaky',
     'structure',
     'cabernet',
     'supported',
     '15',
     'merlot',
     'coffee',
     'chocolate',
     'complete',
     'picture',
     'finishing',
     'strong',
     'end',
     'resulting',
     'value-priced',
     'wine',
     'attractive',
     'flavor',
     'immediate',
     'accessibility.',
     'dry',
     'wine',
     'spicy',
     'tight',
     'taut',
     'texture',
     'strongly',
     'mineral',
     'character',
     'layered',
     'citrus',
     'well',
     'pepper',
     'food',
     'wine',
     'almost',
     'crisp',
     'aftertaste.',
     'slightly',
     'reduced',
     'wine',
     'offers',
     'chalky',
     'tannic',
     'backbone',
     'otherwise',
     'juicy',
     'explosion',
     'rich',
     'black',
     'cherry',
     'whole',
     'accented',
     'throughout',
     'firm',
     'oak',
     'cigar',
     'box.',
     'dominated',
     'oak',
     'oak-driven',
     'aromas',
     'include',
     'roasted',
     'coffee',
     'bean',
     'espresso',
     'coconut',
     'vanilla',
     'carry',
     'palate',
     'together',
     'plum',
     'chocolate',
     'astringent',
     'drying',
     'tannins',
     'give',
     'rather',
     'abrupt',
     'finish.',
     'building',
     '150',
     'years',
     'six',
     'generations',
     'winemaking',
     'tradition',
     'winery',
     'trends',
     'toward',
     'leaner',
     'style',
     'classic',
     'california',
     'buttercream',
     'aroma',
     'cut',
     'tart',
     'green',
     'apple',
     'good',
     'everyday',
     'sipping',
     'wine',
     'flavors',
     'range',
     'pear',
     'barely',
     'ripe',
     'pineapple',
     'prove',
     'approachable',
     'distinctive.',
     'zesty',
     'orange',
     'peels',
     'apple',
     'notes',
     'abound',
     'sprightly',
     'mineral-toned',
     'riesling',
     'dry',
     'palate',
     'yet',
     'racy',
     'lean',
     'refreshing',
     'easy',
     'quaffer',
     'wide',
     'appeal.',
     'baked',
     'plum',
     'molasses',
     'balsamic',
     'vinegar',
     'cheesy',
     'oak',
     'aromas',
     'feed',
     'palate',
     'braced',
     'bolt',
     'acidity',
     'compact',
     'set',
     'saucy',
     'red-berry',
     'plum',
     'flavors',
     'features',
     'tobacco',
     'peppery',
     'accents',
     'finish',
     'mildly',
     'green',
     'flavor',
     'respectable',
     'weight',
     'balance.',
     'raw',
     'black-cherry',
     'aromas',
     'direct',
     'simple',
     'good',
     'juicy',
     'feel',
     'thickens',
     'time',
     'oak',
     'character',
     'extract',
     'becoming',
     'apparent',
     'flavor',
     'profile',
     'driven',
     'dark-berry',
     'fruits',
     'smoldering',
     'oak',
     'finishes',
     'meaty',
     'hot.',
     'desiccated',
     'blackberry',
     'leather',
     'charred',
     'wood',
     'mint',
     'aromas',
     'carry',
     'nose',
     'full-bodied',
     'tannic',
     'heavily',
     'oaked',
     'tinto',
     'fino',
     'flavors',
     'clove',
     'woodspice',
     'sit',
     'top',
     'blackberry',
     'fruit',
     'hickory',
     'forceful',
     'oak-based',
     'aromas',
     'rise',
     'dominate',
     'finish.',
     'red',
     'fruit',
     'aromas',
     'pervade',
     'nose',
     'cigar',
     'box',
     'menthol',
     'notes',
     'riding',
     'back',
     'palate',
     'slightly',
     'restrained',
     'entry',
     'opens',
     'riper',
     'notes',
     'cherry',
     'plum',
     'specked',
     'crushed',
     'pepper',
     'blend',
     'merlot',
     'cabernet',
     'sauvignon',
     'cabernet',
     'franc',
     'approachable',
     'ready',
     'enjoyed.',
     'ripe',
     'aromas',
     'dark',
     'berries',
     'mingle',
     'ample',
     'notes',
     'black',
     'pepper',
     'toasted',
     'vanilla',
     'dusty',
     'tobacco',
     'palate',
     'oak-driven',
     'nature',
     'notes',
     'tart',
     'red',
     'currant',
     'shine',
     'offering',
     'bit',
     'levity.',
     'sleek',
     'mix',
     'tart',
     'berry',
     'stem',
     'herb',
     'along',
     'hint',
     'oak',
     'chocolate',
     'fair',
     'value',
     'widely',
     'available',
     'drink-now',
     'oregon',
     'pinot',
     'wine',
     'oak-aged',
     'six',
     'months',
     'whether',
     'neutral',
     're-staved',
     'indicated.',
     'delicate',
     'aromas',
     'recall',
     'white',
     'flower',
     'citrus',
     'palate',
     'offers',
     'passion',
     'fruit',
     'lime',
     'white',
     'peach',
     'hint',
     'mineral',
     'alongside',
     'bright',
     'acidity.',
     'wine',
     'geneseo',
     'district',
     'offers',
     'aromas',
     'sour',
     'plums',
     'enough',
     'cigar',
     'box',
     'tempt',
     'nose',
     'flavors',
     'bit',
     'flat',
     'first',
     'acidity',
     'tension',
     'sour',
     'cherries',
     'emerges',
     'midpalate',
     'bolstered',
     'black',
     'licorice.',
     'aromas',
     'prune',
     'blackcurrant',
     'toast',
     'oak',
     'carry',
     'extracted',
     'palate',
     'along',
     'flavors',
     'black',
     'cherry',
     'roasted',
     'coffee',
     'beans',
     'firm',
     'drying',
     'tannins',
     'provide',
     'framework.',
     'oak',
     'earth',
     'intermingle',
     'around',
     'robust',
     'aromas',
     'wet',
     'forest',
     'floor',
     'vineyard-designated',
     'pinot',
     'hails',
     'high-elevation',
     'site',
     'small',
     'production',
     'offers',
     'intense',
     'full-bodied',
     'raspberry',
     'blackberry',
     'steeped',
     'smoky',
     'spice',
     'smooth',
     'texture.',
     'pretty',
     'aromas',
     'yellow',
     'flower',
     'stone',
     'fruit',
     'lead',
     'nose',
     'bright',
     'palate',
     'offers',
     'yellow',
     'apple',
     'apricot',
     'vanilla',
     'delicate',
     'notes',
     'lightly',
     'toasted',
     'oak',
     'alongside',
     'crisp',
     'acidity.',
     'aromas',
     'recall',
     'ripe',
     'dark',
     'berry',
     'toast',
     'whiff',
     'cake',
     'spice',
     'soft',
     'informal',
     'palate',
     'offers',
     'sour',
     'cherry',
     'vanilla',
     'hint',
     'espresso',
     'alongside',
     'round',
     'tannins',
     'drink',
     'soon.',
     'aromas',
     'suggest',
     'mature',
     'berry',
     'scorched',
     'earth',
     'animal',
     'toast',
     'anise',
     'palate',
     'offers',
     'ripe',
     'black',
     'berry',
     'oak',
     'espresso',
     'cocoa',
     'vanilla',
     'alongside',
     'dusty',
     'tannins.',
     'clarksburg',
     'becoming',
     'chenin',
     'blanc',
     'california',
     'bottling',
     'using',
     'fruit',
     'sourced',
     'several',
     'vineyards',
     'area',
     'balanced',
     'trace',
     'sweetness',
     'background',
     '1',
     'residual',
     'sugar',
     'crisp',
     'straightforward',
     'blessed',
     'notes',
     'pear',
     'lime',
     'drink',
     'cold.',
     'red',
     'cherry',
     'fruit',
     'comes',
     'laced',
     'light',
     'tannins',
     'giving',
     'bright',
     'wine',
     'open',
     'juicy',
     'character.',
     'merlot',
     'nero',
     "d'avola",
     'form',
     'base',
     'easy',
     'red',
     'wine',
     'would',
     'pair',
     'fettuccine',
     'meat',
     'sauce',
     'pork',
     'roast',
     'quality',
     'fruit',
     'clean',
     'bright',
     'sharp.',
     'part',
     'extended',
     'calanìca',
     'series',
     'grillo-viognier',
     'blend',
     'shows',
     'aromas',
     'honeysuckle',
     'jasmine',
     'backed',
     'touches',
     'cut',
     'grass',
     'wild',
     'sage',
     'mouth',
     'shows',
     'ripe',
     'yellow-fruit',
     'flavors.',
     'rustic',
     'dry',
     'flavors',
     'berries',
     'currants',
     'licorice',
     'spices',
     'made',
     'cabernet',
     'franc',
     'cabernet',
     'sauvignon.',
     'shows',
     'tart',
     'green',
     'gooseberry',
     'flavor',
     'similar',
     'new',
     'zealand',
     'sauvignon',
     'blanc',
     'notes',
     'include',
     'tropical',
     'fruit',
     'orange',
     'honey',
     'unoaked',
     'splash',
     'muscat',
     'commendable',
     'dryness',
     'acidity.',
     'many',
     'erath',
     '2010',
     'vineyard',
     'designates',
     'strongly',
     'herbal',
     'notes',
     'leaf',
     'herb',
     'create',
     'somewhat',
     'unripe',
     'flavor',
     'impressions',
     'touch',
     'bitterness',
     'finish',
     'fruit',
     'passes',
     'ripeness',
     'sweet',
     'tomatoes.',
     'white',
     'flower',
     'lychee',
     'apple',
     'aromas',
     'carry',
     'mellow',
     'bouquet',
     'chunky-feeling',
     'palate',
     'bears',
     'powdery',
     'sweet',
     'flavors',
     'peach',
     'melon',
     'mixed',
     'greener',
     'notes',
     'grass',
     'lime',
     '80',
     'viognier',
     'component',
     'typical',
     'warm',
     'climate',
     'plump',
     'oily',
     'short',
     'finish',
     'chardonnay',
     'fills',
     'blend.',
     'concentrated',
     'cabernet',
     'offers',
     'aromas',
     'cured',
     'meat',
     'dried',
     'fruit',
     'rosemary',
     'barbecue',
     'spice',
     'teriyaki',
     'sauce',
     'flavors',
     'give',
     'wine',
     'bold',
     'chewy',
     'feel.',
     'inky',
     'color',
     'wine',
     'plump',
     'aromas',
     'ripe',
     'fruit',
     'blackberry',
     'jam',
     'rum',
     'cake',
     'palate',
     'soft',
     'smooth.',
     'part',
     'natural',
     'wine',
     'movement',
     'wine',
     'made',
     'organic',
     'grapes',
     'label',
     'printed',
     'vegetable',
     'ink',
     'recycled',
     'paper',
     'quality',
     'fruit',
     'nice',
     'juicy',
     'palate',
     'bright',
     'berry',
     'flavor',
     'finish.',
     'catarratto',
     'one',
     'sicily',
     'widely',
     'farmed',
     'white',
     'grape',
     'varieties',
     'expression',
     'shows',
     'mineral',
     'note',
     'backed',
     'citrus',
     'almond',
     'blossom',
     'touches.',
     'stiff',
     'tannic',
     'wine',
     'slowly',
     'opens',
     'brings',
     'brambly',
     'berry',
     'flavors',
     'play',
     'along',
     'notes',
     'earthy',
     'herbs',
     'touch',
     'bitterness',
     'tannins.',
     'festive',
     'wine',
     'soft',
     'ripe',
     'fruit',
     'acidity',
     'plus',
     'red',
     'berry',
     'flavor.',
     'clean',
     'brisk',
     'mouthfeel',
     'gives',
     'slightly',
     'oaked',
     'sauvignon',
     'blanc',
     'instant',
     'likeability',
     'dry',
     'rich',
     'streak',
     'honey',
     'sweetens',
     'citrus',
     'pear',
     'tropical',
     'fruit',
     'flavors',
     'pair',
     'asian',
     'fare',
     'ham',
     'green',
     'salad',
     'grapefruit',
     'sections.',
     'berry',
     'aroma',
     'comes',
     'cola',
     'herb',
     'notes',
     'palate',
     'tangy',
     'racy',
     'delivers',
     'raspberry',
     'plum',
     'flavors',
     'modest',
     'finish.',
     'right',
     'starting',
     'blocks',
     'oaky',
     'wine',
     'dripping',
     'caramel',
     'vanilla',
     'notes',
     'texture',
     'midpalate',
     'finessed',
     'graceful',
     'drying',
     'tannins',
     'latch',
     'onto',
     'oak-driven',
     'finish',
     'eccentric',
     'blend',
     '50',
     'tannat',
     '35',
     'petit',
     'verdot',
     '15',
     'pinotage.',
     'spicy',
     'fresh',
     'clean',
     'would',
     ...]




```python
#Model 2
freq = FreqDist(stopped_tokens)
freq.most_common(100)
```




    [('wine', 78233),
     ('flavors', 59986),
     ('fruit', 43832),
     ('aromas', 39546),
     ('palate', 37381),
     ('acidity', 31739),
     ('tannins', 28098),
     ('drink', 27868),
     ('cherry', 27238),
     ('ripe', 26904),
     ('black', 25379),
     ('finish', 22086),
     ('red', 18729),
     ('notes', 18399),
     ('spice', 18135),
     ('rich', 17186),
     ('nose', 16875),
     ('fresh', 16671),
     ('oak', 15860),
     ('berry', 15435),
     ('dry', 15103),
     ('plum', 14061),
     ('soft', 13410),
     ('fruits', 13288),
     ('blend', 12974),
     ('finish.', 12830),
     ('apple', 12718),
     ('offers', 12660),
     ('blackberry', 12657),
     ('crisp', 12639),
     ('white', 12241),
     ('sweet', 12173),
     ('texture', 11562),
     ('shows', 11510),
     ('dark', 11322),
     ('light', 11307),
     ('citrus', 11238),
     ('bright', 10869),
     ('cabernet', 10447),
     ('vanilla', 10441),
     ('well', 10224),
     ('full', 10069),
     ('juicy', 9700),
     ('pepper', 9588),
     ('fruity', 9368),
     ('good', 9357),
     ('raspberry', 9243),
     ('firm', 9124),
     ('green', 8951),
     ('touch', 8447),
     ('peach', 8432),
     ('lemon', 8410),
     ('chocolate', 7930),
     ('dried', 7867),
     ('character', 7799),
     ('pear', 7708),
     ('balanced', 7677),
     ('sauvignon', 7405),
     ('structure', 7171),
     ('spicy', 7099),
     ('now.', 6978),
     ('smooth', 6950),
     ('pinot', 6697),
     ('made', 6428),
     ('concentrated', 6304),
     ('also', 6256),
     ('tannic', 6253),
     ('herb', 6213),
     ('herbal', 6120),
     ('tart', 6087),
     ('like', 6063),
     ('wood', 6009),
     ('hint', 5971),
     ('licorice', 5889),
     ('mineral', 5853),
     ('fine', 5851),
     ('bit', 5815),
     ('still', 5803),
     ('give', 5735),
     ('merlot', 5626),
     ('long', 5624),
     ('creamy', 5614),
     ('currant', 5583),
     ('opens', 5549),
     ('note', 5547),
     ('flavor', 5505),
     ('mouth', 5470),
     ('toast', 5436),
     ('alongside', 5388),
     ('dense', 5366),
     ('orange', 5362),
     ('along', 5339),
     ('clean', 5274),
     ('age', 5249),
     ('lead', 5220),
     ('full-bodied', 5205),
     ('leather', 5164),
     ('savory', 5161),
     ('earthy', 5139),
     ('syrah', 5119)]




```python
#Model 1
model = Word2Vec(data, size=100, window=5, min_count=1, workers=4)
```


```python
#Model 2
model = Word2Vec([stopped_tokens], size=100, window=5, min_count=1, workers=4)
```


```python
#Model 1
model.train(data, total_examples=model.corpus_count, epochs=10)
```




    (41025785, 61357260)




```python
#Model 2
model.train([stopped_tokens], total_examples=model.corpus_count, epochs=10)
```




    (100000, 32518830)




```python
#Model 1
wv = model.wv
```


```python
#Model 2
wv = model.wv
```


```python
#Model 1
wv.most_similar('Aromas')
```




    [('Notes', 0.8065941333770752),
     ('Scents', 0.7904120683670044),
     ('Touches', 0.7267212271690369),
     ('Flavors', 0.7266938090324402),
     ('Hints', 0.7152518630027771),
     ('Highlights', 0.6811310648918152),
     ('Layers', 0.6666620969772339),
     ('aromas', 0.6483620405197144),
     ('Accents', 0.6184842586517334),
     ('Whiffs', 0.5952425003051758)]




```python
#Model 2
wv.most_similar('wine')
```




    [('flavors', 0.9999605417251587),
     ('aromas', 0.9999589323997498),
     ('palate', 0.9999567270278931),
     ('fresh', 0.9999556541442871),
     ('notes', 0.9999524354934692),
     ('finish', 0.9999516010284424),
     ('acidity', 0.9999514818191528),
     ('fruit', 0.9999512434005737),
     ('rich', 0.9999507665634155),
     ('long', 0.999945342540741)]




```python
#Model 1
wv['Aromas']
```




    array([-2.8454772e-01, -4.9646276e-01, -4.1607366e+00,  1.3687874e+00,
            3.7103353e+00, -2.2572200e+00,  1.5280840e+00,  1.0774952e+01,
            5.8199108e-01,  3.9913079e-01,  5.0573599e-01,  5.7638831e+00,
           -1.0738858e+00,  3.5575392e+00, -1.4113132e+00, -3.9413726e+00,
           -4.4100223e+00,  2.3618758e+00,  2.4368601e+00,  2.9416436e-01,
           -1.1382991e+00, -1.4160022e+00, -1.0194013e+00,  4.4622931e+00,
            4.9901233e+00, -5.4468638e-01, -1.0721319e+00, -1.8607612e+00,
           -3.3467453e+00, -4.8381205e+00,  4.0050812e+00,  4.5586586e+00,
            5.3959908e+00,  1.1704580e+00, -2.9011128e+00,  1.8712484e+00,
            6.7603850e-01, -3.1047711e-01,  2.6159806e+00,  5.1789899e+00,
           -4.3956556e+00, -3.6152303e-02,  2.3022845e-01,  7.7908570e-01,
            1.0570222e+00,  1.0256631e+00, -3.3134811e+00, -2.3461936e+00,
            3.5431964e+00, -2.5050921e+00,  1.4910073e+00,  2.3028243e+00,
            4.0623903e+00, -1.4807061e+00,  1.3853774e+00, -9.5884657e-01,
           -4.3523154e-01,  5.8442742e-01, -4.2608304e+00, -1.3126242e+00,
            9.8674643e-01, -8.3644432e-01, -5.8570397e-01, -2.0641305e+00,
            2.8219841e+00,  1.7997749e-01, -1.4399205e+00, -1.6792736e+00,
           -1.4929314e+00, -2.1934102e+00, -6.4961159e-01,  3.4461277e+00,
           -1.8673275e+00,  3.2533655e+00, -6.1784852e-01,  3.8838327e+00,
            3.1070728e+00,  8.5480080e+00,  1.3122096e+00, -4.4085309e-01,
            4.3943644e+00, -1.3378646e+00,  6.2275285e-01, -1.1511211e+00,
            1.1906828e+00,  1.0169150e+00,  4.7072336e-01,  2.0859981e+00,
           -1.8663391e+00, -5.1666390e-02, -1.0829883e+00, -2.0107212e+00,
           -9.7947801e-03, -4.2755194e+00,  2.2179451e+00,  2.5563836e+00,
           -2.0982084e+00, -2.3097272e+00, -1.0918939e+00, -3.1242824e+00],
          dtype=float32)




```python
#Model 2
wv['wine']
```




    array([-7.28988290e-01,  5.89181244e-01, -4.20181185e-01,  6.79549158e-01,
           -5.61286092e-01, -3.03451627e-01, -1.66687425e-02, -5.53961247e-02,
            2.15241551e-01,  8.15211535e-01, -3.36894304e-01, -8.51017296e-01,
           -5.23345947e-01, -1.52614221e-01, -1.26449871e+00, -1.87686563e-01,
            1.13376394e-01,  4.30480361e-01,  5.14692307e-01, -3.60265654e-03,
           -2.00087711e-01,  4.17635232e-01,  4.30520289e-02, -6.54390872e-01,
           -7.38561824e-02, -7.61514843e-01, -2.91326940e-01, -4.29191977e-01,
           -1.50086150e-01, -5.44779003e-04, -2.33520828e-02,  1.39292732e-01,
           -3.78958672e-01,  1.95481911e-01, -6.60543323e-01, -5.05610049e-01,
            3.51299405e-01,  1.43056020e-01, -2.39997923e-01, -1.05621195e+00,
            3.27203989e-01,  1.92553774e-01, -4.11055326e-01,  1.60443261e-01,
            8.49295974e-01, -3.77539128e-01,  2.63945073e-01,  7.00588882e-01,
            7.89521456e-01,  1.42790228e-01, -4.34528619e-01,  2.16291085e-01,
            3.89513433e-01,  7.12460577e-01, -1.59765661e-01,  7.61509597e-01,
            5.50355971e-01,  7.13623881e-01,  1.45635888e-01, -5.47974050e-01,
            7.57722855e-01, -3.34780663e-01,  3.64975929e-01, -2.17997923e-01,
           -2.08977655e-01,  3.80337596e-01, -8.93101335e-01, -9.14730906e-01,
           -6.62377775e-02,  1.01781356e+00, -1.60306588e-01, -6.65664256e-01,
           -1.06292665e+00, -2.28052557e-01, -1.35990822e+00, -2.31734335e-01,
           -2.18401566e-01, -3.11260730e-01, -2.31806815e-01, -5.01776874e-01,
           -2.85131037e-01, -1.30438536e-01, -9.32505786e-01,  3.23792428e-01,
           -7.15391994e-01, -5.66569567e-01,  3.84505808e-01,  1.68744847e-01,
           -5.04178405e-01, -3.59894663e-01,  3.89697969e-01, -2.07055613e-01,
            9.27913904e-01, -1.31096780e-01,  3.29299748e-01,  1.67388916e-02,
            2.07456037e-01, -6.06089175e-01, -5.75950384e-01, -2.03689039e-01],
          dtype=float32)




```python
#Model 1
wv.vectors
```




    array([[-2.7985141e-01,  2.9602352e-01, -1.5445307e+00, ...,
            -1.1343012e+00,  1.9518315e+00, -9.7611606e-01],
           [ 5.9382135e-01, -5.2232675e-02, -5.8637893e-01, ...,
             7.7536911e-01,  1.1432698e+00,  1.5821687e+00],
           [-1.5910634e+00,  2.4604274e-01, -1.3907577e+00, ...,
            -1.3064194e+00,  2.3976212e+00, -6.9682693e-01],
           ...,
           [ 4.1693710e-02, -7.1414458e-03,  4.7542807e-02, ...,
             1.5803667e-02, -5.0320905e-03,  5.4156035e-02],
           [ 4.1727621e-02, -1.5584046e-03,  1.0024369e-01, ...,
             1.4103501e-01, -2.5418842e-02,  1.2796254e-02],
           [ 3.8580362e-02,  5.5177845e-02,  1.3666784e-02, ...,
            -1.0357072e-02,  5.6253565e-03,  7.9172308e-04]], dtype=float32)




```python
#Model 2
wv.vectors
```




    array([[-7.2898829e-01,  5.8918124e-01, -4.2018119e-01, ...,
            -6.0608917e-01, -5.7595038e-01, -2.0368904e-01],
           [-6.6605896e-01,  5.3329480e-01, -3.7432432e-01, ...,
            -5.4986542e-01, -5.2854866e-01, -1.8777938e-01],
           [-6.0834283e-01,  4.9009115e-01, -3.4383330e-01, ...,
            -4.9553579e-01, -4.8041889e-01, -1.6588880e-01],
           ...,
           [ 3.7508595e-03, -2.4629305e-03,  1.9838421e-03, ...,
             3.8793008e-03,  4.9809683e-03,  2.8974307e-03],
           [-1.6139228e-03,  2.0733212e-04,  3.0159059e-03, ...,
             2.2480378e-03, -1.4378247e-03,  3.5166260e-04],
           [-3.3727317e-04, -8.2688195e-05,  2.9322866e-03, ...,
            -9.9405111e-04, -4.0982985e-03,  4.1952282e-03]], dtype=float32)




```python
#Model 1
wv.most_similar(positive=['Aromas', 'Notes'], negative=['Scents'])
```




    [('Flavors', 0.7498737573623657),
     ('Touches', 0.7128608226776123),
     ('Hints', 0.7040060758590698),
     ('Layers', 0.6173896789550781),
     ('Highlights', 0.5982887148857117),
     ('Accents', 0.5878312587738037),
     ('Tones', 0.5302862524986267),
     ('Its', 0.5239870548248291),
     ('Cola', 0.5220677852630615),
     ('aromas', 0.5202503204345703)]




```python
#Model 2
wv.most_similar(positive=['wine', 'flavors'], negative=['finish'])
```




    [('aromas', 0.999920129776001),
     ('notes', 0.9999077916145325),
     ('fruit', 0.9999011158943176),
     ('palate', 0.9998993277549744),
     ('rich', 0.9998965859413147),
     ('fresh', 0.9998900890350342),
     ('acidity', 0.9998890161514282),
     ('long', 0.9998889565467834),
     ('berry', 0.9998863935470581),
     ('nose', 0.999883234500885)]



# Neural Network


```python
target = df['Price Category']
```


```python
#Model 1
total_vocabulary = set(word for description in data for word in description)
```


```python
#Model 2
#total_vocabulary = set(word for description in stopped_tokens/corpus? for word in description)
```


```python
#Model 1
len(total_vocabulary)
print('There are {} unique tokens in the dataset.'.format(len(total_vocabulary)))
```

    There are 51780 unique tokens in the dataset.



```python
#Model 2
len(total_vocabulary)
print('There are {} unique tokens in the dataset.'.format(len(total_vocabulary)))
```


```python
#Model 1
glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    for line in f:
        parts = line.split()
        word = parts[0].decode('utf-8')
        if word in total_vocabulary:
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
```


```python
#Model 2
glove = {}
with open('glove.6B.50d.txt', 'rb') as f:
    for line in f:
        parts = line.split()
        word = parts[0].decode('utf-8')
        if word in total_vocabulary:
            vector = np.array(parts[1:], dtype=np.float32)
            glove[word] = vector
```


```python
glove['aromas']
```




    array([ 1.093   ,  0.65613 , -1.5218  ,  0.69799 ,  0.61011 ,  0.25169 ,
            0.03538 , -0.53179 ,  0.13368 ,  1.57    , -0.38117 ,  0.010478,
            1.1387  , -0.38847 , -0.18242 ,  0.16008 , -0.39453 ,  0.025334,
            0.4808  , -1.7512  , -0.38028 ,  0.52807 ,  0.8787  , -0.47349 ,
           -0.23939 ,  1.2645  , -0.39373 ,  1.4757  ,  0.78596 , -0.51365 ,
            0.62215 , -0.10212 , -0.11154 , -0.036182,  1.4312  , -0.11613 ,
           -1.5928  ,  0.10555 ,  0.7415  , -0.34381 ,  0.058573,  0.22099 ,
           -0.089933, -0.56413 ,  0.96937 ,  1.1901  ,  1.0556  , -0.067241,
           -0.75956 , -0.22039 ], dtype=float32)




```python
glove['wine']
```


```python
class W2vVectorizer(object):
    
    def __init__(self, w2v):
        # Takes in a dictionary of words and vectors as input
        self.w2v = w2v
        if len(w2v) == 0:
            self.dimensions = 0
        else:
            self.dimensions = len(w2v[next(iter(glove))])
    
    # Note: Even though it doesn't do anything, it's required that this object implement a fit method or else
    # it can't be used in a scikit-learn pipeline  
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        return np.array([
            np.mean([self.w2v[w] for w in words if w in self.w2v]
                   or [np.zeros(self.dimensions)], axis=0) for words in X])
```


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
#Model 1
y = to_categorical(target)
X = df['description']
```


```python
#Model 2
y = to_categorical(target)
#X = stopped_tokens/corpus?
```


```python
#Model 1
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
```


```python
#Model 2
X_train,X_test,y_train,y_test = train_test_split(X,y,stratify=y)
```


```python
#Model 1
X_train
```




    105737    88-90 Barrel sample. Very soft, rounded wine, ...
    52637     One of Faiveley's monopoles (wholly owned vine...
    121292    This Bordeaux-style blend is fresh, showing ar...
    106545    Aromas include black berries, coffee, espresso...
    126506    Young and vigorous now, marked by tannins and ...
                                    ...                        
    46670     Black fruit aromas show over toasted vanilla a...
    111406    Aromas of fresh cantaloupe melon and apricots,...
    20379     Solid, ripe, perfumed and tannic, this is a co...
    49605     With its expressive notes of espressso and bla...
    51444     From parcels around the village of Maimbray, t...
    Name: description, Length: 97478, dtype: object




```python
#Model 2
X_train
```


```python
#Model 1
y_train
```




    array([[0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           [0., 0., 1., 0., 0.],
           ...,
           [0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0.]], dtype=float32)




```python
#Model 2
y_train
```


```python
#Model 1
y_test
```




    array([[0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 0., 1., 0.],
           ...,
           [0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0.],
           [0., 0., 1., 0., 0.]], dtype=float32)




```python
#Model 2
y_test
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
#Model 1
embedding_size = 128
model = Sequential()
model.add(Embedding(20000, embedding_size)) #input_length=100?
model.add(LSTM(25, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

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
    dense_2 (Dense)              (None, 5)                 255       
    =================================================================
    Total params: 2,576,955
    Trainable params: 2,576,955
    Non-trainable params: 0
    _________________________________________________________________



```python
#Model 2
embedding_size = 128
model = Sequential()
model.add(Embedding(20000, embedding_size)) #input_length=100?
model.add(LSTM(25, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

model.summary()
```


```python
model_1 = model.fit(X_tr, y_train, epochs=3, batch_size=128, validation_split=0.3)
```

    Train on 68234 samples, validate on 29244 samples
    Epoch 1/3
    68234/68234 [==============================] - 68s 999us/step - loss: 0.7355 - acc: 0.7755 - val_loss: 0.5406 - val_acc: 0.8004
    Epoch 2/3
    68234/68234 [==============================] - 66s 971us/step - loss: 0.5504 - acc: 0.7994 - val_loss: 0.5083 - val_acc: 0.8004
    Epoch 3/3
    68234/68234 [==============================] - 63s 926us/step - loss: 0.5037 - acc: 0.8063 - val_loss: 0.5010 - val_acc: 0.8125



```python
model_2 = model.fit(X_tr, y_train, epochs=3, batch_size=128, validation_split=0.3)
```


```python
#Model 1
y_hat_test = model.predict(X_te)
y_hat_test
```




    array([[2.1163463e-05, 1.4354082e-02, 9.9128073e-01, 2.5182551e-01,
            4.9337628e-03],
           [7.2505786e-06, 1.9325461e-04, 8.3761376e-01, 9.1981906e-01,
            7.8036778e-02],
           [8.3553681e-07, 1.2669052e-02, 9.9913800e-01, 7.6963849e-02,
            2.2124355e-04],
           ...,
           [1.7269737e-07, 3.2510310e-02, 9.9983585e-01, 1.8551691e-02,
            1.7905861e-05],
           [8.8934121e-06, 7.5249918e-02, 9.9856001e-01, 5.0534748e-02,
            3.5000488e-04],
           [7.9983518e-05, 8.1394427e-03, 9.6457791e-01, 5.0883985e-01,
            2.4896331e-02]], dtype=float32)




```python
#Model 2
y_hat_test = model.predict(X_te)
y_hat_test
```


```python
#Model 1
y_hat_test = y_hat_test.argmax(axis=1)#.shape)
y_hat_test
```




    array([2, 3, 2, ..., 2, 2, 2])




```python
#Model 2
y_hat_test = y_hat_test.argmax(axis=1)#.shape)
y_hat_test
```


```python
def plot_confusion_matrix(conf_matrix, classes = None, normalize=True,
                          title='Confusion Matrix', cmap="Blues",
                          print_raw_matrix=False,
                          fig_size=(4,4)):
    """Check if Normalization Option is Set to True. 
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
    """Evaluates neural network using sklearn metrics"""
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


![png](output_83_0.png)


    
    
    ------------------------------------------------------------
    	CLASSIFICATION REPORT:
    ------------------------------------------------------------
                  precision    recall  f1-score   support
    
               1       0.00      0.00      0.00      1570
               2       0.83      0.96      0.89     25979
               3       0.56      0.28      0.37      4773
               4       0.00      0.00      0.00       171
    
        accuracy                           0.81     32493
       macro avg       0.35      0.31      0.31     32493
    weighted avg       0.75      0.81      0.77     32493
    


    /Users/acusiobivona/opt/anaconda3/envs/learn-env/lib/python3.6/site-packages/sklearn/metrics/_classification.py:1272: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
      _warn_prf(average, modifier, msg_start, len(result))



![png](output_83_3.png)



```python
evaluate_model(y_test, y_hat_test, model_2)
```


```python
#Model 1
from wordcloud import WordCloud
wordcloud = WordCloud(stopwords=None,collocations=False)
wordcloud.generate(','.join(total_vocabulary))
plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off')
```




    (-0.5, 399.5, 199.5, -0.5)




![png](output_85_1.png)



```python
#Model 2
wordcloud = WordCloud(stopwords=None,collocations=False)
wordcloud.generate(','.join(stopped_tokens))
plt.figure(figsize = (12, 12), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis('off')
```


```python
#Model 1
import nltk
bigram_measures = nltk.collocations.BigramAssocMeasures()
word_finder = nltk.BigramCollocationFinder.from_words(total_vocabulary)

words_scored = word_finder.score_ngrams(bigram_measures.raw_freq)
top_words = pd.DataFrame.from_records(words_scored,columns=['Words','Frequency']).head(20)
top_words
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
      <th>Words</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>(!, factor)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>1</td>
      <td>(#, necessitates)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>2</td>
      <td>($, Livio)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>3</td>
      <td>(%, coffee/toasty)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>4</td>
      <td>(&amp;, sandy)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>5</td>
      <td>(', Jérémy)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>6</td>
      <td>('', dried-cherry)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>7</td>
      <td>('01, messy)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>8</td>
      <td>('02, lemonade-flavored)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>9</td>
      <td>('03, vineyards—Mark)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>10</td>
      <td>('04, sell-by)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>11</td>
      <td>('04s, Salmon-pink)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>12</td>
      <td>('04—which, sweet-spiced)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>13</td>
      <td>('05, green-plum-)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>14</td>
      <td>('05s, Wonderfully)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>15</td>
      <td>('06, Handles)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>16</td>
      <td>('06s, duck)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>17</td>
      <td>('07, Bojador)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>18</td>
      <td>('07s, now–)</td>
      <td>0.000019</td>
    </tr>
    <tr>
      <td>19</td>
      <td>('08, Liquors)</td>
      <td>0.000019</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Model 2
bigram_measures = nltk.collocations.BigramAssocMeasures()
word_finder = nltk.BigramCollocationFinder.from_words(stopped_tokens)

words_scored = word_finder.score_ngrams(bigram_measures.raw_freq)
top_words = pd.DataFrame.from_records(words_scored,columns=['Words','Frequency']).head(20)
top_words
```


```python
#Model 1
bigram_measures = nltk.collocations.BigramAssocMeasures()

word_pmi_finder = nltk.BigramCollocationFinder.from_words(total_vocabulary)
word_pmi_finder.apply_freq_filter(1)

word_pmi_scored = word_pmi_finder.score_ngrams(bigram_measures.pmi)
pd.DataFrame.from_records(word_pmi_scored,columns=['Words','PMI']).head(20)
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
      <th>Words</th>
      <th>PMI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>(!, factor)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>1</td>
      <td>(#, necessitates)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>2</td>
      <td>($, Livio)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>3</td>
      <td>(%, coffee/toasty)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>4</td>
      <td>(&amp;, sandy)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>5</td>
      <td>(', Jérémy)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>6</td>
      <td>('', dried-cherry)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>7</td>
      <td>('01, messy)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>8</td>
      <td>('02, lemonade-flavored)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>9</td>
      <td>('03, vineyards—Mark)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>10</td>
      <td>('04, sell-by)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>11</td>
      <td>('04s, Salmon-pink)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>12</td>
      <td>('04—which, sweet-spiced)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>13</td>
      <td>('05, green-plum-)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>14</td>
      <td>('05s, Wonderfully)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>15</td>
      <td>('06, Handles)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>16</td>
      <td>('06s, duck)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>17</td>
      <td>('07, Bojador)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>18</td>
      <td>('07s, now–)</td>
      <td>15.660107</td>
    </tr>
    <tr>
      <td>19</td>
      <td>('08, Liquors)</td>
      <td>15.660107</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Model 2
bigram_measures = nltk.collocations.BigramAssocMeasures()

word_pmi_finder = nltk.BigramCollocationFinder.from_words(stopped_vocabulary)
word_pmi_finder.apply_freq_filter(5)

word_pmi_scored = word_pmi_finder.score_ngrams(bigram_measures.pmi)
pd.DataFrame.from_records(word_pmi_scored,columns=['Words','PMI']).head(20)
```


```python

```
