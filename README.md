## Code for "FMAP: Flow Matching for Human Motion Prediction".

### Data

**Datasets for [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [HumanEva-I](http://humaneva.is.tue.mpg.de/)**:

We adopt the data preprocessing from [GSPS](https://github.com/wei-mao-2019/gsps), which you can refer to [here](https://drive.google.com/drive/folders/1sb1n9l0Na5EqtapDVShOJJ-v6o-GZrIJ) and download all files into the `./data` directory.

Final `./data` directory structure is shown below:

```
data
├── amass_retargeted.npy
├── data_3d_h36m.npz
├── data_3d_h36m_test.npz
├── data_3d_humaneva15.npz
├── data_3d_humaneva15_test.npz
├── data_multi_modal
│   ├── data_candi_t_his25_t_pred100_skiprate20.npz
│   └── t_his25_1_thre0.500_t_pred100_thre0.100_filtered_dlow.npz
└── humaneva_multi_modal
    ├── data_candi_t_his15_t_pred60_skiprate15.npz
    └── t_his15_1_thre0.500_t_pred60_thre0.010_index_filterd.npz
```

### Pretrained Model

We provide pretrained model [Google Drive](https://drive.google.com/drive/folders/18bbfFdvr80SDy_o3_ohZhnH-cFO8o_k3?usp=sharing) .
Please put the 'ckpt_ema_500.pt' (for HumanEva) in the `./results/he/models` and 'ckpt_ema_1000.pt' (for Human36M) in the `./results/h36,/models`.


### Training

For Human3.6M:

The argument '--ode_options' for Human3.6M is recommended as '{"step_size": 0.1}'.
```
python main_fm.py --cfg h36m --mode train
```

For HumanEva-I:

```
python main_fm.py --cfg he --mode train
```

### Evaluation

Evaluate on Human3.6M:

```
python main_fm.py --cfg h36m --mode eval --ckpt ./results/h36m/models/ckpt_ema_1000.pt
```

Evaluate on HumanEva-I:

```
python main_fm.py --cfg he --mode eval --ckpt ./results/he/models/ckpt_ema_500.pt
```

### Acknowledgments

Part of the code is borrowed from the [HumanMAC](https://github.com/LinghaoChan/HumanMAC) repo.

### License

This code is distributed under an [MIT LICENSE](https://github.com/LinghaoChan/HumanMAC/blob/main/LICENSE). Note that our code depends on other libraries and datasets which each have their own respective licenses that must also be followed.

