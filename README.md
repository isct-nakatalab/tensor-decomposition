# tensor-decomposition
This repository is for tensor decomposition including below models;

`src/model.py`
- Parafac : tensor decomposition model assuming normal distribution
- NonNegativeParafac : tensor decomposition model assuming normal distribution which returns non-negative predicted values
- PoissonParafac : tensor decomposition model assuming poisson distribution
- ZeroInflatedParafac ： tensor decomposition model assuming data to follow zero inflated poisson distribution 


Import the above models with the following command.
```
! pip install -q git+https://$$TOKEN@github.com/tokyotech-nakatalab/tensor_decomposition.git
```


To perform tensor decomposition, specify the model with the corresponding input parameters.
```
model = Parafac(tensor, key, rank=rank, num_step=num_step, logging_step=logging_step)
```

You can earn the predicted tensor and decomposed factors with the following command.
```
pred_tensor = model.predict
pred_matrixlist = model.matrix_list
```

For ZeroInflatedParafac model, you can also earn the predicted probability matrix with the following command.
```
pred_P = model.P
````
