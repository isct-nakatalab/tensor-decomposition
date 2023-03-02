from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit


class Parafac:
    def __init__(
        self, tensor, key, rank=5, num_step=1000, logging_step=100, init_values=None
    ):
        """
        tensor : jaxlib.xla_extension.Array
        key : jax.random.PRNGKey
        rank : int, rank of decomposition
        num_step : int, number of training steps
        logging_step : int, number of logging steps
        init_values : list[DeviceArray, DeviceArray, ...], initial values of matrix
        """
        self.key = key
        self.input_shape = tensor.shape
        self.rank = rank
        self.num_step = num_step
        self.logging_step = logging_step
        self.start = ord("a")
        self.common_dim = "r"
        self.operation = self.make_operation()

        if init_values is not None:
            self.matrix_list = list(
                [jnp.array(init_values[i]) for i in range(len(self.input_shape))]
            )
        else:
            self.matrix_list = list(
                [
                    jax.random.normal(self.key, shape=(size, rank))
                    for size in self.input_shape
                ]
            )

        self.optimize(tensor, num_step=self.num_step, logging_step=self.logging_step)

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self):
        # reconstruction of tensor from decomposed values
        preds = jnp.einsum(self.operation, *self.matrix_list)
        return preds

    def calc_loss(self, data_tensor, pred_tensor):
        mse_loss = ((pred_tensor - data_tensor) ** 2).sum()
        return mse_loss

    def forward(self, data_tensor):
        """
        data_tensor : Tensor to be decomposed
        """
        # reconstruction of tensor from decomposed values
        pred_tensor = self.predict()
        loss = self.calc_loss(data_tensor, pred_tensor)
        return loss

    def optimize(self, tensor, num_step=1000, logging_step=100):
        @partial(jax.jit, static_argnames=["t"])
        def step(tensor, t, matrix_list):
            RR_list = jnp.array(
                [jnp.dot(M.T, M) for i, M in enumerate(matrix_list) if i != t]
            )

            ztz = RR_list.prod(axis=0)

            ztz_inverse = jnp.linalg.inv(ztz)

            xz = [tensor] + [M for i, M in enumerate(matrix_list) if i != t]
            xz = jnp.einsum(self.operation_lst[t], *xz)
            return jnp.dot(xz, ztz_inverse)

        log_loss = []
        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))
        self.operation_lst = []
        for i in target:
            source = target + "," + ",".join([j + "r" for j in target if i != j])
            operation = source + "->" + i + "r"
            self.operation_lst.append(operation)

        for n_step in range(num_step):
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print(f"step{n_step} loss", loss)

            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list))
                self.matrix_list[tensor_id] = step_out

class NNParafac:
    def __init__(
        self, tensor, key, rank=5, num_step=1000, logging_step=100, init_values=None
    ):
        """
        tensor : jaxlib.xla_extension.Array
        key : jax.random.PRNGKey
        rank : int, rank of decomposition
        num_step : int, number of training steps
        logging_step : int, number of logging steps
        init_values : list[DeviceArray, DeviceArray, ...], initial values of matrix
        """
        self.key = key
        self.input_shape = tensor.shape
        self.rank = rank
        self.num_step = num_step
        self.logging_step = logging_step
        self.start = ord("a")
        self.common_dim = "r"
        self.operation = self.make_operation()

        if init_values is not None:
            self.matrix_list = list(
                [jnp.array(init_values[i]) for i in range(len(self.input_shape))]
            )
        else:
            self.matrix_list = list(
                [
                    jax.random.normal(self.key, shape=(size, rank)) ** 2
                    for size in self.input_shape
                ]
            )

        self.optimize(tensor, num_step=self.num_step, logging_step=self.logging_step)

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self):
        # reconstruction of tensor from decomposed values
        preds = jnp.einsum(self.operation, *self.matrix_list)
        return preds

    def calc_loss(self, data_tensor, pred_tensor):
        mse_loss = ((pred_tensor - data_tensor) ** 2).sum()
        return mse_loss

    def forward(self, data_tensor):
        """
        data_tensor : Tensor to be decomposed
        """
        # reconstruction of tensor from decomposed values
        pred_tensor = self.predict()
        loss = self.calc_loss(data_tensor, pred_tensor)
        return loss

    def optimize(self, tensor, num_step, logging_step):
        @partial(jax.jit, static_argnames=["t"])
        def step(tensor, t, matrix_list):
            RR_list = jnp.array(
                [jnp.dot(M.T, M) for i, M in enumerate(matrix_list) if i != t]
            )

            ztz = RR_list.prod(axis=0)

            xz = [tensor] + [M for i, M in enumerate(matrix_list) if i != t]
            xz = jnp.einsum(self.operation_lst[t], *xz)
            numerator = jnp.clip(xz, a_min=jnp.finfo(float).eps)
            denominator = jnp.dot(matrix_list[t], ztz)
            denominator = jnp.clip(denominator, a_min=jnp.finfo(float).eps)
            factor = matrix_list[t] * numerator / denominator
            return factor

        log_loss = []
        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))
        self.operation_lst = []
        for i in target:
            source = target + "," + ",".join([j + "r" for j in target if i != j])
            operation = source + "->" + i + "r"
            self.operation_lst.append(operation)

        for n_step in range(num_step):
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print(f"step{n_step} loss", loss)

            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list))
                self.matrix_list[tensor_id] = step_out


class PoissonParafac:
    def __init__(
        self, tensor, key, rank=5, num_step=1000, logging_step=100, init_values=None
    ):
        """
        tensor : jaxlib.xla_extension.Array
        key : jax.random.PRNGKey
        rank : int, rank of decomposition
        num_step : int, number of training steps
        logging_step : int, number of logging steps
        init_values : list[DeviceArray, DeviceArray, ...], initial values of matrix
        """
        self.key = key
        self.input_shape = tensor.shape
        self.num_axis = len(self.input_shape)
        self.rank = rank
        self.num_step = num_step
        self.logging_step = logging_step
        self.start = ord("a")
        self.common_dim = "r"
        self.operation = self.make_operation()

        if init_values is not None:
            self.matrix_list = list(
                [jnp.array(init_values[i]) for i in range(len(self.input_shape))]
            )
        else:
            self.matrix_list = list(
                [
                    jax.random.poisson(self.key, lam=1, shape=(size, rank))
                    for size in self.input_shape
                ]
            )

        self.optimize(tensor, num_step=self.num_step, logging_step=self.logging_step)

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self):
        # reconstruction of tensor from decomposed values
        preds = jnp.einsum(self.operation, *self.matrix_list)
        return preds

    def calc_loss(self, data_tensor, pred_tensor):
        mse_loss = ((pred_tensor - data_tensor) ** 2).sum()
        return mse_loss

    def forward(self, data_tensor):
        """
        data_tensor : Tensor to be decomposed
        """
        # reconstruction of tensor from decomposed values
        pred_tensor = self.predict()
        loss = self.calc_loss(data_tensor, pred_tensor)
        return loss

    def optimize(self, tensor, num_step, logging_step):
        @partial(jax.jit, static_argnames=["t"])
        def step(tensor, t, matrix_list):
            n = jnp.einsum(self.sum_operation, *matrix_list)
            d = jnp.einsum(self.operation, *matrix_list).reshape(*self.input_shape, -1)
            d = jnp.clip(d, a_min=jnp.finfo(float).eps)
            lamda = n / d

            numerator = jnp.einsum(self.operation_lst[t], tensor, lamda)
            denominator = jnp.einsum(
                self.sum_operation2, *[M for i, M in enumerate(matrix_list) if i != t]
            )
            denominator = denominator.sum(
                axis=[i for i in range(self.num_axis - 1)]
            ).reshape(1, -1)

            denominator = jnp.clip(denominator, a_min=jnp.finfo(float).eps)
            return numerator / denominator

        log_loss = []
        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))

        self.operation_lst = []
        for i in target:
            operation = target + ", " + target + "r" + "->" + i + "r"
            self.operation_lst.append(operation)

        self.sum_operation = (
            " ,".join(chr(ord("a") + i) + "r" for i in range(self.num_axis))
            + "->"
            + "".join(chr(ord("a") + i) for i in range(self.num_axis))
            + "r"
        )
        self.sum_operation2 = (
            " ,".join(chr(ord("a") + i) + "r" for i in range(self.num_axis - 1))
            + "->"
            + "".join(chr(ord("a") + i) for i in range(self.num_axis - 1))
            + "r"
        )

        for n_step in range(num_step):
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print(f"step{n_step} loss", loss)
            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list))
                self.matrix_list[tensor_id] = step_out

class ZIPParafac:
    def __init__(
        self,
        tensor,
        key,
        rank=5,
        num_step=1000,
        logging_step=100,
        init_values=None,
        P_axis=None,
    ):
        """
        tensor : jaxlib.xla_extension.Array
        key : jax.random.PRNGKey
        rank : int, rank of decomposition
        num_step : int, number of training steps
        logging_step : int, number of logging steps
        init_values : list[DeviceArray, DeviceArray, ...], initial values of matrix
        P_axis : list, axes of probability matrix to take average on 
        """
        self.key = key
        self.input_shape = tensor.shape
        self.num_axis = len(self.input_shape)
        self.P = jnp.zeros(self.input_shape)
        self.rank = rank
        self.num_step = num_step
        self.logging_step = logging_step
        self.P_axis = P_axis
        self.start = ord("a")
        self.common_dim = "r"
        self.operation = self.make_operation()

        if init_values is not None:
            self.matrix_list = list(
                [jnp.array(init_values[i]) for i in range(len(self.input_shape))]
            )
        else:
            self.matrix_list = list(
                [
                    jax.random.gamma(self.key, 1.0, shape=(size, self.rank))
                    for size in self.input_shape
                ]
            )

        self.optimize(
            tensor,
            num_step=self.num_step,
            logging_step=self.logging_step,
            P_axis=self.P_axis,
        )

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self, expected=True):
        # reconstruction of tensor from decomposed values
        preds = jnp.einsum(self.operation, *self.matrix_list)
        if expected:
            preds = preds * (1 - self.P)
        return preds

    def calc_loss(self, data_tensor, pred_tensor, in_sample=False):
        if in_sample:
            p = self.Z
            is_zero = self.zero
        else:
            p = self.P
            is_zero = data_tensor == 0
        # calculate loss
        nonzero_loss = (
            (
                data_tensor
                * (jnp.log(data_tensor + 1e-8) - jnp.log(pred_tensor + 1e-8))
                - data_tensor
                + pred_tensor
                - jnp.log(1 - p + 1e-8)
            )
            * (1 - is_zero)
        ).sum()
        zero_loss = -(
            jnp.log(p + (1 - p) * (jnp.exp(-pred_tensor) + 1e-8)) * is_zero
        ).sum()
        return nonzero_loss + zero_loss

    def forward(self, data_tensor):
        """
        data_tensor : Tensor to be decomposed
        """
        # reconstruction of tensor from decomposed values
        pred_tensor = self.predict(expected=False)
        loss = self.calc_loss(data_tensor, pred_tensor, in_sample=False)
        return loss

    def optimize(self, tensor, num_step=1000, logging_step=100, P_axis=None):
        @partial(jax.jit, static_argnames=["t"])
        def step(tensor, t, matrix_list, Z):
            n = jnp.einsum(self.sum_operation, *matrix_list)
            d = jnp.einsum(self.operation, *matrix_list).reshape(*self.input_shape, -1)
            d = jnp.clip(d, a_min=jnp.finfo(float).eps)
            lamda = n / d

            numerator = jnp.einsum(self.operation_lst[t], (1 - Z) * tensor, lamda)

            denominator = jnp.einsum(
                self.operation_lst2[t],
                1 - Z,
                *[M for i, M in enumerate(matrix_list) if i != t],
            )

            denominator = jnp.clip(denominator, a_min=jnp.finfo(float).eps)
            return numerator / denominator

        @jit
        def update_Z(P, matrix_list):
            likelihood = jnp.exp(-jnp.einsum(self.operation, *matrix_list))
            Z = self.zero * (P / (P + (1 - P) * (likelihood + 1e-8)))
            return Z

        log_loss = []

        self.zero = tensor == 0
        self.Z = self.zero * 1e-8
        self.P = self.Z.mean()

        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))

        self.operation_lst = []
        for i in target:
            operation = target + ", " + target + "r" + "->" + i + "r"
            self.operation_lst.append(operation)

        self.sum_operation = (
            " ,".join(chr(ord("a") + i) + "r" for i in range(self.num_axis))
            + "->"
            + "".join(chr(ord("a") + i) for i in range(self.num_axis))
            + "r"
        )
        self.operation_lst2 = []
        for t, j in enumerate(target):
            operation = (
                target
                + ", "
                + " ,".join(
                    chr(ord("a") + i) + "r" for i in range(self.num_axis) if i != t
                )
                + "->"
                + j
                + "r"
            )
            self.operation_lst2.append(operation)

        for n_step in range(num_step):
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print(f"step{n_step} loss", loss)
            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list), self.Z)
                self.matrix_list[tensor_id] = step_out

            if P_axis:
                self.P = self.Z.mean(axis=P_axis, keepdims=True)
            else:
                self.P = self.Z.mean()
            self.Z = update_Z(self.P, self.matrix_list)
