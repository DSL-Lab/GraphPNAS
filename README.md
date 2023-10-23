# Probabilistic Neural Architecture Search with Deep Graph Generation

## Dependencies
Python 3.8, PyTorch>(1.8.0)

```pip install -r requirements.txt```

## DATASET Preparation
put nasbench101 file to ./data/nasbench_only108.tfrecord
unzip nasbench201 .tar files to data folder: ./data/nasbench_201

## Run Demos
To run NB101 experiments:
  ```python run_nas_multiple.py -c config/nb101.yaml```

To run NB201 experiments:
  ```python run_nas_multiple.py -c config/nb201_*.yaml```


## Cite
Please cite our paper if you use this code in your research work.
```
@article{li2022graphpnas,
  title={GraphPNAS: Learning Distribution of Good Neural Architectures via Deep Graph Generative Models},
  author={Li, Muchen and Liu, Jeffrey Yunfan and Sigal, Leonid and Liao, Renjie},
  journal={arXiv preprint arXiv:2211.15155},
  year={2022}
}
```

## Questions/Bugs
Please submit a Github issue 
