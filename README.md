# Tensor Decomposition
This repository is for tensor decomposition including below models;

`src/tensordec/model.py`
- **Parafac** : tensor decomposition model assuming data to follow normal distribution
- **NonNegativeParafac** : tensor decomposition model assuming data to follow normal distribution which returns non-negative predicted values
- **PoissonParafac** : tensor decomposition model assuming data to follow poisson distribution
- **ZeroInflatedParafac** ： tensor decomposition model assuming data to follow zero inflated poisson distribution 

----------------------------

## Installing Tensor Decomposition Models
You can import this repository on your local.
```
! pip install git+https://github.com/tokyotech-nakatalab/tensor-decomposition.git
```
Import the above models with the following command.
```
import tensordec
from tensrodec.model import Parafac, NonNegativeParafac, PoissonParafac, ZeroInflatedParafac
```


## Tensor Decomposition
To perform tensor decomposition, specify the model with the corresponding input parameters.
```
model = Parafac(tensor, key, rank=rank, num_step=num_step, logging_step=logging_step)
```

### Checking Predicted Tensor Values
You can earn the predicted tensor and decomposed factors with the following command.
```
pred_tensor = model.predict
pred_matrixlist = model.matrix_list
```

For ZeroInflatedParafac model, you can also earn the predicted probability matrix with the following command.
```
pred_P = model.P
````
