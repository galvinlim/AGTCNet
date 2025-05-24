# AGTCNet: A Graph-Temporal Approach for Principled Motor Imagery EEG Classification

The ***Attentive Graph-Temporal Convolutional Network (AGTCNet)*** is a novel graph-temporal model designed for subject-independent (subject-invariant) and subject-specific (session-invariant) motor imagery EEG (MI-EEG) classification.


## Dependencies

```
pip install -r requirements.txt
```


## MI-EEG Datasets

### BCI Competition IV Dataset 2A (BCICIV2A)

Available in https://bnci-horizon-2020.eu/database/data-sets

### EEG Motor Movement/Imagery Dataset (EEGMMIDB)

Available in https://physionet.org/content/eegmmidb/1.0.0/


## Model Training

To train **AGTCNet**, run the `main.ipynb` notebook and modify the configurations in the 4th cell.

To train **state-of-the-art models**, run the `main_sota.ipynb` notebook and modify the configurations in the 4th and 7th cells.

### Dataset Selection (4th Cell)

```py
DATASET = 'BCICIV2A'        # 'BCICIV2A' | 'EEGMMIDB'
```

### Model Training Framework (4th Cell)

```py
SUBJECT_SELECTION = 'SN'    # 'SL' | 'SM' | 'SN'
SESSION_SELECTION = 'DS'    # 'DS' | 'RS'
FINE_TUNING = False         # for 'SL-DS-FT' using 'SN' Top Models as baseline

VARY_VALID_RUN = False      # True for DS: LOSeO
KFOLD = False               # True for SN: LSSO | False for SN: LOSO 

CLASS = 4                   # (int)
```

### Model Selection (7th Cell)

```py
MODEL_NAME = 'DB-ATCNet'    # 'EEGNet' | 'EEG-TCNet' | 'TCNet-Fusion' | 'ATCNet' | 'DB-ATCNet' 
```

## AGTCNet Performance

<table style="border-collapse: collapse; width: 100%; text-align: center; border-top: 2px solid black; border-bottom: 2px solid black;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left;">Datasets (Classes)</th>
      <th colspan="3" style="text-align: center;">Subject-Independent</th>
      <th colspan="3" style="text-align: center; border-left: 1px solid black;">Subject-Specific</th>
    </tr>
    <tr>
      <th style="text-align: center;">SMA20 Acc (%)</th>
      <th style="text-align: center;">Acc (%)</th>
      <th style="text-align: center;">κ-score</th>
      <th style="text-align: center; border-left: 1px solid black;">SMA20 Acc (%)</th>
      <th style="text-align: center;">Acc (%)</th>
      <th style="text-align: center;">κ-score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;"><strong><em>BCICIV2A (4-Class)</em></strong></td>
      <td>66.82 ± 9.62</td>
      <td>70.20 ± 9.51</td>
      <td>0.60 ± 0.13</td>
      <td style="border-left: 1px solid black;">82.88 ± 6.97</td>
      <td>84.13 ± 6.56</td>
      <td>0.79 ± 0.09</td>
    </tr>
    <tr>
      <td style="text-align: left;"><strong><em>EEGMMIDB (4-Class)</em></strong></td>
      <td>64.14 ± 2.83</td>
      <td>65.44 ± 2.81</td>
      <td>0.54 ± 0.04</td>
      <td style="border-left: 1px solid black;">72.13 ± 16.66</td>
      <td>74.81 ± 15.70</td>
      <td>0.66 ± 0.21</td>
    </tr>
    <tr>
      <td style="text-align: left;"><strong><em>EEGMMIDB (2-Class)</em></strong></td>
      <td>85.22 ± 2.35</td>
      <td>86.61 ± 2.20</td>
      <td>0.73 ± 0.04</td>
      <td style="border-left: 1px solid black;">90.54 ± 10.49</td>
      <td>92.53 ± 9.38</td>
      <td>0.85 ± 0.19</td>
    </tr>
  </tbody>
</table>


## State-of-the-Art Models

The codes to reproduce state-of-the-art models are obtained from the following repositories:

- [EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet)
- [DB-ATCNet](https://github.com/zk-xju/DB-ATCNet)