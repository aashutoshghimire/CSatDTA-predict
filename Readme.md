------------
# Run The file
------------
```
python main.py
```
## About
This repo contains a predict function that returns KIBA score for smile string and protein seq input. The model used in this project belongs to CSatDTA.

------------
# Requirements
------------

```
CUDA 11.3 (nvidia-smi) 
python 3.7.7
tensorgpu 2.1.0 
keras==2.3.1
scikit-learn
```

-------------
## Note
------------

If you get this error

```
AttributeError: 'str' object has no attribute 'decode'
```

the solution was downgrading the h5py package (in my case to 2.10.0), apparently putting back only Keras and Tensorflow to the correct versions was not enough.

```
pip install 'h5py==2.10.0' --force-reinstall
```

