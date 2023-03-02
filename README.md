# Tensor Decomposition
This repository is for tensor decomposition including below models;

`src/tensordec/model.py`
- **Parafac** : tensor decomposition model assuming data to follow normal distribution
- **NonNegativeParafac** : parafac model which returns non-negative predicted values
- **PoissonParafac** : tensor decomposition model assuming data to follow poisson distribution
- **ZeroInflatedPoissonParafac** : tensor decomposition model assuming data to follow zero inflated poisson distribution 


It allows faster decomposition than the existing libraries with the use of Just In Time (JIT) compilation of JAX Python function which enables efficient execution in XLA.

----------------------------

## Importing Tensor Decomposition Models
You can import this repository on your local.
```
! pip install git+https://github.com/tokyotech-nakatalab/tensor-decomposition.git
```
Import the above models with the following command.
```
from tensordec.model import Parafac, NonNegativeParafac, PoissonParafac, ZeroInflatedParafac
```


## Tensor Decomposition
To perform tensor decomposition, specify the model with the corresponding input parameters.
```
model = Parafac(tensor, key, rank=rank, num_step=num_step, logging_step=logging_step)
```

### Checking Predicted Tensor Values
You can earn the predicted tensor and decomposed factors with the following command.
```
pred_tensor = model.predict()
pred_matrixlist = model.matrix_list
```

For ZeroInflatedPoissonParafac model, you can also earn the mixture ratio of the Poisson distribution and a distribution that represents fixed probabilities of zero.
```
pred_P = model.P
````
