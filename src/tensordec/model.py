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


@jit
def init_pred_step(lmd, nu):
    zd1_z = (2**nu + 2*lmd) / (2**nu*(1+lmd)+lmd**2)
    r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
    r_zd1 = 6**nu/(3*lmd**2) + 2*3**nu/(3*lmd)

    zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
    zd1_z = zd1_z * zd1_z_rate

    rate_max = jnp.max(zd1_z_rate)

    return r_z, r_zd1, zd1_z, rate_max

@jit
def calc_pred_step(j, r_z, r_zd1, zd1_z, lmd, nu):
    r_z = (j)**nu / lmd * (1 + r_z)
    r_zd1 = ((j)**nu * (j-1) / (lmd*(j))) * (1 + r_zd1)

    zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
    zd1_z = zd1_z * zd1_z_rate

    return r_z, r_zd1, zd1_z, jnp.max(zd1_z_rate)

def calc_pred(lmd, nu, max_step=1000000, truncate_tol=1e-10):
    r_z, r_zd1, zd1_z, rate_max = init_pred_step(lmd, nu)
    begin = 4
    for j in range(begin, max_step+begin):
        r_z, r_zd1, zd1_z, rate_max = calc_pred_step(j, r_z, r_zd1, zd1_z, lmd, nu)
        if rate_max<=(1+truncate_tol):
            break
    return lmd*zd1_z

class COMPParafac:
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
        self.nu = 1.

        self.max_value = int(tensor.max())
        self.optimize(tensor, num_step=self.num_step, logging_step=self.logging_step)

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self):
        lmd = jnp.einsum(self.operation, *self.matrix_list)
        # print(lmd)
        preds = calc_pred(lmd, self.nu, max_step=self.max_step)
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
        l2_weight = 1e-8
        min_nu = 0.5
        self.max_step = int((self.max_value*2)**(1/min_nu))
        log_y_fac = []
        factor = 0
        a = list(range(self.max_value+1))
        a[0] = 1
        for j in a:
            tmp = jnp.log(j)
            factor += tmp
            log_y_fac.append(factor)
        log_y_fac_sum = jnp.array(log_y_fac).take(tensor.astype(int)).sum()
        

        log_loss = []
        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))

        calc_feature_operation = ",".join([target[i]+"r" for i in range(self.num_axis-1)])+"->"+target[:-1]+"r"
        calc_lmd_operation_lst = ["".join([target[i] for i in range(self.num_axis) if i != t])+"r,"+target[t]+"r->"+target for t in range(self.num_axis)]
        calc_weight_operation_lst = [target+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"r"+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"s"+"->"+target[t]+"rs" for t in range(self.num_axis)]
        calc_a_operation_lst = [target+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"r"+"->"+target[t]+"r" for t in range(self.num_axis)]

        @jit
        def init_z_step(lmd, nu):
            zd1_z = (2**nu + 2*lmd) / (2**nu*(1+lmd)+lmd**2)
            zd2_z = 2 / (2**nu*(1 + lmd) + lmd**2)
            r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
            r_zd1 = 6**nu/(3*lmd**2) + 2*3**nu/(3*lmd)
            r_zd2 = 3**nu/(3*lmd)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate

            rates = zd1_z_rate+zd2_z_rate

            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, rates

        @jit
        def calc_z_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu):
            r_z = (j)**nu / lmd * (1 + r_z)
            r_zd1 = ((j)**nu * (j-1) / (lmd*(j))) * (1 + r_zd1)
            r_zd2 = (j**nu)*(j-2)/j/lmd*(1 + r_zd2)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate

            rates = zd1_z_rate+zd2_z_rate

            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, rates

        def calc_zs_iter(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, truncate_tol):
            r_z, r_zd1, r_zd2, zd1_z, zd2_z, rates = calc_z_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu)
            continue_index = rates>(2+truncate_tol)
            num_continue = continue_index.sum()
            if num_continue == 0:
                return zd1_z, zd2_z
            argsort_index = jnp.argsort(-continue_index.astype(int))
            
            inverse_index = argsort_index.argsort()
            r_z_sorted = r_z[argsort_index]
            r_zd1_sorted = r_zd1[argsort_index]
            r_zd2_sorted = r_zd2[argsort_index]
            zd1_z_sorted = zd1_z[argsort_index]
            zd2_z_sorted = zd2_z[argsort_index]
            lmd_sorted = lmd[argsort_index]
            zd1_z_continue, zd2_z_continue = calc_zs_iter(j+1, r_z_sorted[:num_continue], r_zd1_sorted[:num_continue], r_zd2_sorted[:num_continue], zd1_z_sorted[:num_continue], zd2_z_sorted[:num_continue], lmd_sorted[:num_continue], nu, truncate_tol)
            zd1_z = jnp.concatenate([zd1_z_continue, zd1_z_sorted[num_continue:]])[inverse_index]
            zd2_z = jnp.concatenate([zd2_z_continue, zd2_z_sorted[num_continue:]])[inverse_index]
            return zd1_z, zd2_z

        def calc_zs(lmd, nu, max_step=1000000, truncate_tol=1e-10):
            lmd_shape = lmd.shape
            lmd = lmd.reshape(-1)
            r_z, r_zd1, r_zd2, zd1_z, zd2_z, rate_max = init_z_step(lmd, nu)
            begin = 4
            zd1_z, zd2_z = calc_zs_iter(begin, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, truncate_tol)
            zd1_z = zd1_z.reshape(lmd_shape)
            zd2_z = zd2_z.reshape(lmd_shape)
            # for j in tqdm(range(begin, max_step)):
            #     r_z, r_zd1, r_zd2, zd1_z, zd2_z, rate_max = calc_zs_iter(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu)
                # if rate_max<=(2+truncate_tol):
                #     break
            return zd1_z, zd2_z
            
        @partial(jax.jit, static_argnames=["t"])
        def calc_lmd_x(t, matrix_list):
            x = jnp.einsum(calc_feature_operation, *[matrix_list[_] for _ in range(len(matrix_list)) if _!=t])
            lmd = jnp.einsum(calc_lmd_operation_lst[t], x, matrix_list[t])
            return lmd, x
        
        @partial(jax.jit, static_argnames=["t"])
        def calc_next_value(zd1_z, zd2_z, tensor, lmd, matrix_list, t, x):
            weight = -zd2_z + zd1_z**2 - (tensor) / (lmd**2+1e-6)
            weight = jnp.clip(weight, a_max=1e-8)
            weight = jnp.einsum(calc_weight_operation_lst[t], weight, x, x)
            weight = weight - jnp.expand_dims(jnp.eye(weight.shape[1]), 0)*l2_weight

            a = jnp.einsum(calc_a_operation_lst[t], (-zd1_z+(tensor)/(lmd+1e-6)), x) - matrix_list[t]*l2_weight
            delta = jnp.linalg.solve(weight, a)
            out = matrix_list[t] - delta
            out = jnp.clip(out, a_min=0.)
            return out

        def step(tensor, t, matrix_list, nu, max_step=50, truncate_tol=1e-6):
            lmd, x = calc_lmd_x(t, matrix_list)
            zd1_z, zd2_z  = calc_zs(lmd, nu, max_step=max_step, truncate_tol=truncate_tol)
            out = calc_next_value(zd1_z, zd2_z, tensor, lmd, matrix_list, t, x)
            return out
        
        @jit
        def init_znu_step(lmd, nu):
            zd1_z = jnp.log(2)*lmd**2 / (2**nu*(1 + lmd) + lmd**2)
            zd2_z = jnp.log(2)**2*lmd**2 / (2**nu*(1 + lmd) + lmd**2)
            factor = jnp.log(6)
            r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
            r_zd1 = 3**nu/lmd * jnp.log(2)/jnp.log(6)
            r_zd2 = 3**nu/lmd * (jnp.log(2)/jnp.log(6))**2

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate
            
            rate_max = jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)

            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max

        @jit
        def calc_znu_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, factor):
            factor = factor + jnp.log(j)
            r_z = (j)**nu / lmd * (1 + r_z)
            r_zd1 = (j**nu/lmd) * ((factor-jnp.log(j))/factor) * (1 + r_zd1)
            r_zd2 = (j**nu/lmd) * ((factor-jnp.log(j))/factor)**2 * (1 + r_zd2)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate
            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)

        def calc_zs_nu(lmd, nu, max_step=1000000, truncate_tol=1e-10):
            r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max = init_znu_step(lmd, nu)
            begin = 4
            for j in range(begin, max_step):
                r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max = calc_znu_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, factor)
                if rate_max<=(2+truncate_tol):
                    break
            return zd1_z, zd2_z
        
        @jit
        def calc_next_value_nu(zd1_z, zd2_z, lmd, nu):
            a = - zd1_z.sum() - log_y_fac_sum
            w = (-zd2_z + zd1_z**2).sum()
            w = jnp.clip(w, a_max=1e-6)
            delta = a / w
            out = nu - delta
            out = jnp.clip(out, a_min=min_nu, a_max=10.)
            return out

        # @partial(jax.jit, static_argnames=["max_step"])
        def step_nu(tensor, matrix_list, nu, max_step=50, truncate_tol=1e-6):
            lmd = jnp.einsum(self.operation, *matrix_list)
            zd1_z, zd2_z = calc_zs_nu(lmd, nu, max_step=max_step, truncate_tol=truncate_tol)
            out = calc_next_value_nu(zd1_z, zd2_z, lmd, nu)
            return out
        
        for n_step in range(num_step):
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print()
                print(f"step{n_step} loss", loss)
                print()
            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list), self.nu, max_step=self.max_step)
                self.matrix_list[tensor_id] = step_out

            self.nu = step_nu(tensor, tuple(self.matrix_list), self.nu, max_step=self.max_step)



class ZICOMPParafac:
    def __init__(
        self, tensor, key, rank=5, num_step=1000, logging_step=100, init_values=None, mixture_ratio_axis=None,
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
        self.nu = 1.
        self.mixture_ratio = jnp.zeros(self.input_shape)

        self.max_value = int(tensor.max())
        self.optimize(tensor, num_step=self.num_step, logging_step=self.logging_step, mixture_ratio_axis=mixture_ratio_axis)

    def make_operation(self):
        n_factors = len(self.input_shape)
        target = "".join(chr(self.start + i) for i in range(n_factors))
        source = ",".join(i + self.common_dim for i in target)
        operation = source + "->" + target
        return operation

    def predict(self):
        def calc_pred(lmd, nu, max_step=self.max_value*2, truncate_tol=1e-6):
            # print(lmd)
            zd1_z = (2**nu + 2*lmd) / (2**nu*(1+lmd)+lmd**2)
            begin = 3
            for j in range(begin, max_step):
                if j == begin:
                    r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
                    r_zd1 = 6**nu/(3*lmd**2) + 2*3**nu/(3*lmd)
                else:
                    r_z = (j)**nu / lmd * (1 + r_z)
                    r_zd1 = ((j)**nu * (j-1) / (lmd*(j))) * (1 + r_zd1)

                zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
                zd1_z = zd1_z * zd1_z_rate

                if (zd1_z_rate <= (1 + truncate_tol)).all():
                    break
            return lmd*zd1_z

        # reconstruction of tensor from decomposed values
        lmd = jnp.einsum(self.operation, *self.matrix_list)
        # print(lmd)
        preds = calc_pred(lmd, self.nu)
        preds = preds * (1 - self.mixture_ratio)
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

    def optimize(self, tensor, num_step, logging_step, mixture_ratio_axis=None):
        l2_weight = 1e-8
        min_nu = 0.5
        max_step = int((self.max_value*2)**(1/min_nu))
        log_y_fac = []
        factor = 0
        a = list(range(self.max_value+1))
        a[0] = 1
        for j in a:
            tmp = jnp.log(j)
            factor += tmp
            log_y_fac.append(factor)
        log_y_fac = jnp.array(log_y_fac).take(tensor.astype(int))
        

        log_loss = []
        target = "".join(chr(ord("a") + i) for i in range(len(tensor.shape)))

        calc_feature_operation = ",".join([target[i]+"r" for i in range(self.num_axis-1)])+"->"+target[:-1]+"r"
        calc_lmd_operation_lst = ["".join([target[i] for i in range(self.num_axis) if i != t])+"r,"+target[t]+"r->"+target for t in range(self.num_axis)]
        calc_weight_operation_lst = [target+","+target+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"r"+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"s"+"->"+target[t]+"rs" for t in range(self.num_axis)]
        calc_a_operation_lst = [target+","+target+","+"".join([target[i] for i in range(self.num_axis) if i != t])+"r"+"->"+target[t]+"r" for t in range(self.num_axis)]

        zero = tensor == 0
        expectated_mixture_ratio = zero
        self.mixture_ratio = expectated_mixture_ratio.mean(axis=mixture_ratio_axis, keepdims=True)

        @jit
        def calc_z_step(j, z, r_z, lmd, nu):
            r_z = (j)**nu / lmd * (1 + r_z)

            z_rate = (1 + 1/r_z)
            z = z * z_rate

            return z, r_z, z_rate

        def calc_z(lmd, nu, max_step=50, truncate_tol=1e-8):
            z = lmd+1
            r_z = 1/lmd
            for j in range(2, max_step):
                z, r_z, z_rate = calc_z_step(j, z, r_z, lmd, nu)

                if z_rate.max()<(1+truncate_tol):
                    break
            return z

        def update_mixture_ratio(mixture_ratio, matrix_list, max_step=max_step):
            likelihood = 1/calc_z(jnp.einsum(self.operation, *matrix_list), self.nu, max_step=max_step)
            expectated_mixture_ratio = zero * ((mixture_ratio + 1e-8) / (mixture_ratio + (1 - mixture_ratio) * likelihood + 1e-8))
            return expectated_mixture_ratio

        @jit
        def init_z_step(lmd, nu):
            zd1_z = (2**nu + 2*lmd) / (2**nu*(1+lmd)+lmd**2)
            zd2_z = 2 / (2**nu*(1 + lmd) + lmd**2)
            r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
            r_zd1 = 6**nu/(3*lmd**2) + 2*3**nu/(3*lmd)
            r_zd2 = 3**nu/(3*lmd)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate

            rate_max = jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)

            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, rate_max

        @jit
        def calc_zs_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu):
            r_z = (j)**nu / lmd * (1 + r_z)
            r_zd1 = ((j)**nu * (j-1) / (lmd*(j))) * (1 + r_zd1)
            r_zd2 = (j**nu)*(j-2)/j/lmd*(1 + r_zd2)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate
            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)
        
        def calc_zs(lmd, nu, max_step=1000000, truncate_tol=1e-10):
            r_z, r_zd1, r_zd2, zd1_z, zd2_z, rate_max = init_z_step(lmd, nu)
            begin = 4
            for j in range(begin, max_step):
                r_z, r_zd1, r_zd2, zd1_z, zd2_z, rate_max = calc_zs_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu)
                if rate_max<=(2+truncate_tol):
                    break
            return zd1_z, zd2_z
        
        @partial(jax.jit, static_argnames=["t"])
        def calc_lmd_x(t, matrix_list):
            x = jnp.einsum(calc_feature_operation, *[matrix_list[_] for _ in range(len(matrix_list)) if _!=t])
            lmd = jnp.einsum(calc_lmd_operation_lst[t], x, matrix_list[t])
            return lmd, x
        
        @partial(jax.jit, static_argnames=["t"])
        def calc_next_value(zd1_z, zd2_z, tensor, lmd, matrix_list, t, x, expectated_mixture_ratio):
            weight = -zd2_z + zd1_z**2 - (tensor) / (lmd**2+1e-6)
            # weight = jnp.clip(weight, a_max=-1e-2)
            weight = jnp.clip(weight, a_max=0.)
            weight = jnp.einsum(calc_weight_operation_lst[t], weight, 1-expectated_mixture_ratio, x, x)
            weight = weight - jnp.expand_dims(jnp.eye(weight.shape[1]), 0)*l2_weight
            a = jnp.einsum(calc_a_operation_lst[t], (-zd1_z+(tensor)/(lmd+1e-6)), (1-expectated_mixture_ratio), x) - matrix_list[t]*l2_weight
            delta = jnp.linalg.solve(weight, a)
            out = matrix_list[t] - delta
            out = jnp.clip(out, a_min=0.)
            return out

        def step(tensor, t, matrix_list, nu, expectated_mixture_ratio, max_step=50, truncate_tol=1e-6):
            lmd, x = calc_lmd_x(t, matrix_list)
            zd1_z, zd2_z  = calc_zs(lmd, nu, max_step=max_step, truncate_tol=truncate_tol)
            out = calc_next_value(zd1_z, zd2_z, tensor, lmd, matrix_list, t, x, expectated_mixture_ratio)
            return out
        
        @jit
        def init_znu_step(lmd, nu):
            zd1_z = jnp.log(2)*lmd**2 / (2**nu*(1 + lmd) + lmd**2)
            zd2_z = jnp.log(2)**2*lmd**2 / (2**nu*(1 + lmd) + lmd**2)
            factor = jnp.log(6)
            r_z = 6**nu/lmd**3 + 6**nu/lmd**2 + 3**nu/lmd
            r_zd1 = 3**nu/lmd * jnp.log(2)/jnp.log(6)
            r_zd2 = 3**nu/lmd * (jnp.log(2)/jnp.log(6))**2

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate
            
            rate_max = jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)

            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max

        @jit
        def calc_znu_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, factor):
            factor = factor + jnp.log(j)
            r_z = (j)**nu / lmd * (1 + r_z)
            r_zd1 = (j**nu/lmd) * ((factor-jnp.log(j))/factor) * (1 + r_zd1)
            r_zd2 = (j**nu/lmd) * ((factor-jnp.log(j))/factor)**2 * (1 + r_zd2)

            zd1_z_rate = (1 + 1/r_zd1) / (1 + 1/r_z)
            zd1_z = zd1_z * zd1_z_rate

            zd2_z_rate = (1 + 1/r_zd2) / (1 + 1/r_z)
            zd2_z = zd2_z * zd2_z_rate
            return r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, jnp.max(zd1_z_rate)+jnp.max(zd2_z_rate)

        def calc_zs_nu(lmd, nu, max_step=1000000, truncate_tol=1e-10):
            r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max = init_znu_step(lmd, nu)
            begin = 4
            for j in range(begin, max_step):
                r_z, r_zd1, r_zd2, zd1_z, zd2_z, factor, rate_max = calc_znu_step(j, r_z, r_zd1, r_zd2, zd1_z, zd2_z, lmd, nu, factor)
                if rate_max<=(2+truncate_tol):
                    break
            return zd1_z, zd2_z
        
        @jit
        def calc_next_value_nu(zd1_z, zd2_z, lmd, nu, expectated_mixture_ratio):
            a = - ((zd1_z - log_y_fac)*(1-expectated_mixture_ratio)).sum()# - l2_weight*nu
            w = ((-zd2_z + zd1_z**2)*(1-expectated_mixture_ratio)).sum()
            w = jnp.clip(w, a_max=1e-6)
            # w = w - l2_weight
            delta = a / w
            out = nu - delta
            out = jnp.clip(out, a_min=min_nu, a_max=10.)
            return out

        def step_nu(tensor, matrix_list, nu, expectated_mixture_ratio, max_step=50, truncate_tol=1e-6):
            lmd = jnp.einsum(self.operation, *matrix_list)
            zd1_z, zd2_z = calc_zs_nu(lmd, nu, max_step=max_step, truncate_tol=truncate_tol)
            out = calc_next_value_nu(zd1_z, zd2_z, lmd, nu, expectated_mixture_ratio)
            return out
        
        for n_step in range(num_step):
            # print(self.mixture_ratio)
            if n_step % logging_step == 0:
                loss = self.forward(tensor)
                log_loss.append(loss.item())
                print(f"step{n_step} loss", loss)
            for tensor_id in range(len(self.matrix_list)):
                step_out = step(tensor, tensor_id, tuple(self.matrix_list), self.nu, expectated_mixture_ratio, max_step=max_step)
                self.matrix_list[tensor_id] = step_out

            self.nu = step_nu(tensor, tuple(self.matrix_list), self.nu, expectated_mixture_ratio, max_step=max_step)
            expectated_mixture_ratio = update_mixture_ratio(self.mixture_ratio, self.matrix_list)
            if mixture_ratio_axis:
                self.mixture_ratio = expectated_mixture_ratio.mean(axis=mixture_ratio_axis, keepdims=True)
            else:
                self.mixture_ratio = expectated_mixture_ratio.mean()
