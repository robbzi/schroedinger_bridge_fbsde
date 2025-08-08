import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from typing import Callable, Optional


from utils import get_derivatives


class NNAnsatz(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        hidden_size,
        activation,
        learning_rate,  # 0.01
        domain_extrema: torch.Tensor,
        time_grid_tensor: torch.Tensor,
        backward: bool,
        differentiate_through_drift: bool,
        batch_normalization: bool,
        idx: Optional[int] = None,
        name: str = "NNAnsatz",
        exp_output=False,
        # optimizer,
        # lr_scheduler,
        # decayRate,
    ):
        super().__init__()

        # add time dimension
        input_dimension = input_dimension + 1
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.batch_normalization = batch_normalization
        self.domain_extrema = domain_extrema
        self.backward = backward
        self.differentiate_through_drift = differentiate_through_drift
        self.learning_rate = learning_rate
        self.idx = idx
        self.name = name
        # self.time_continuous = time_continuous

        assert time_grid_tensor is not None, (
            "time_grid_tensor must be provided for time_continuous=True"
        )
        # assert time grid tensor is increasing order
        assert torch.all(time_grid_tensor[1:] >= time_grid_tensor[:-1]), (
            "time_grid_tensor must be passed in increasing order"
        )
        if backward:
            # if backward, we want to reverse the time grid tensor
            time_grid_tensor = torch.flip(time_grid_tensor, dims=[0])
        self.time_grid_tensor = time_grid_tensor

        self.exp_output = exp_output

        # self.dummy_layer = nn.Linear(input_dimension, 1000)
        # self.relu = nn.ReLU()
        # self.dummy_layer2 = nn.Linear(1000, output_dimension)

        # Define layers
        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(input_dimension, output_dimension))
        else:
            layers.append(nn.Linear(input_dimension, hidden_size))
            if self.batch_normalization:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation)
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                if self.batch_normalization:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(activation)
            layers.append(nn.Linear(hidden_size, output_dimension))

        self.model = nn.Sequential(*layers)
        # Xavier initialization
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

        # # Setup optimizer
        # self.optimizer = torch.optim.Adam(
        #     self.parameters(), lr=learning_rate, weight_decay=0
        # )  # emphasizing to set weight_decay to 0
        # # self.optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate) # with weight decay
        # decayRate = 0.9993
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=decayRate
        # )

    def forward(self, x):
        if self.exp_output:
            # If exp_output is True, apply exponential to the output
            return torch.exp(self.model(x))
        else:
            # Normal forward pass
            return self.model(x)

    def log_forward(self, x):
        """
        Logarithm of the forward pass to avoid unnecessary computation.
        """
        assert self.exp_output, "This method is only for models with exp_output=True."
        if self.exp_output:
            return self.model(x)

    def grad_forward(self, x, create_graph: bool = False, spacial_only: bool = False):
        """
        Gradient of the forward pass.
        """
        assert not self.exp_output, (
            "This method is only for models with exp_output=False."
        )
        
        #x.requires_grad = True
        y = self.model(x)
        grad = torch.autograd.grad(
            torch.sum(y),
            x,
            create_graph=create_graph,
        )[0]
        if spacial_only:
            return grad[:, 1:]
        else:
            return grad

    def grad_log_forward(self, x):
        """
        Gradient of the logarithm of the forward pass to avoid unnecessary computation.
        """
        assert self.exp_output, "This method is only for models with exp_output=True."
        x.requires_grad = True
        y = self.model(x)
        grad = torch.autograd.grad(torch.sum(y), x)[0]
        return grad

    def drift_sde_solution_func(
        self,
        init_vals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Drift SDE solution function. Assumes particles are transported by the vector field
        given by grad_log_forward.
        """
        # self.model.eval()
        # self.model.to("cuda")
        # TODO: ensure init_vals are generated on cuda, and copy constructed on cuda
        init_vals = init_vals.to("cuda")
        init_vals.requires_grad = True
        # dt can be unevenly spaced
        # TODO: we can save this computation by adding self.dt as a property
        self.time_grid_tensor = self.time_grid_tensor.to(init_vals.device)
        dt = self.time_grid_tensor[1:] - self.time_grid_tensor[:-1]

        # assert dt is only positive or only negative
        assert (torch.all(dt > 0) and not self.backward) or (
            torch.all(dt < 0) and self.backward
        ), (
            f"time grid tensor is not strictly monotonic or incosistent with backward={self.backward}."
        )
        dt = torch.abs(dt)

        # d dimensional noise increments
        noise = torch.randn(
            (
                init_vals.shape[0],
                self.time_grid_tensor.shape[0] - 1,
                init_vals.shape[1],
            ),
            device=init_vals.device,
        )

        # initialize X with zeros
        X = torch.zeros(
            (init_vals.shape[0], self.time_grid_tensor.shape[0], init_vals.shape[1]),
            device=init_vals.device,
        )
        

        # TODO: do we need to move the model to the device? Maybe to it externally
        # # set initial value
        #X[:, 0, :] = init_vals
        # for t in range(1, len(self.time_grid_tensor)):
        #     X_for_forward = X[:, t - 1, :]
        #     # concatenate with time step
        #     X_for_forward = torch.cat(
        #         (
        #             self.time_grid_tensor[t - 1].unsqueeze(0).repeat(X.shape[0], 1),
        #             X_for_forward,
        #         ),
        #         dim=-1,
        #     )
        #     current_dt = dt[None, t - 1, None]
        #     X[:, t, :] = (
        #         X[:, t - 1, :]
        #         + self.grad_forward(
        #             X_for_forward,
        #             create_graph=self.differentiate_through_drift,
        #             spacial_only=True,
        #         )
        #         * current_dt
        #         + torch.sqrt(current_dt) * noise[:, t - 1, :]
        #     )
        # return X

                # Build path without writing into a base tensor referenced by views
        X_list = []
        x_prev = init_vals
        X_list.append(x_prev)

        for t in range(1, len(self.time_grid_tensor)):
            # concat time and state
            t_in = self.time_grid_tensor[t - 1].expand(x_prev.shape[0], 1)
            X_for_forward = torch.cat((t_in, x_prev), dim=-1)

            current_dt = dt[t - 1]
            drift = self.grad_forward(
                X_for_forward,
                create_graph=self.differentiate_through_drift,
                spacial_only=True,
            )
            # Euler–Maruyama step
            x_next = x_prev + drift * current_dt + torch.sqrt(current_dt) * noise[:, t - 1, :]
            X_list.append(x_next)
            x_prev = x_next

        X = torch.stack(X_list, dim=1)
        return X

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = nn.functional.mse_loss(y_hat, y)
    #     # save loss for logging
    #     self.log(
    #         "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     self.log(
    #         "lr",
    #         self.optimizer.param_groups[0]["lr"],
    #         on_step=True,
    #         on_epoch=True,
    #         prog_bar=True,
    #         logger=True,
    #     )
    #     return loss

    # def configure_optimizers(self):
    #     lr_scheduler_config = LRSchedulerConfigType(
    #         scheduler=self.lr_scheduler,
    #         interval="step",
    #         frequency=1,
    #     )

    #     return OptimizerLRSchedulerConfig(
    #         optimizer=self.optimizer,
    #         lr_scheduler = lr_scheduler_config
    #     )


class DummyNNAnsatz(nn.Module):
    """
    Dummy model for initialization of Schrödinger Bridge training.
    if exp_output is True, then we want to call log_forward instead of forward.
    """

    def __init__(self, device, exp_output=False):
        super().__init__()
        self.device = device
        self.exp_output = exp_output

    def forward(self, x):
        if self.exp_output:
            raise NotImplementedError(
                "This model is not meant to be used for forward pass. Use log_forward instead."
            )
        else:
            return torch.zeros((x.shape[0], 1), device=self.device)

    def log_forward(self, x):
        if not self.exp_output:
            raise NotImplementedError(
                "This model is not meant to be used for log_forward pass. Use forward instead."
            )
        else:
            return torch.zeros((x.shape[0], 1), device=self.device)


class LitSchroedingerBridgePINN(pl.LightningModule):
    def __init__(
        self,
        # optimizer,
        # lr_scheduler,
        # decayRate,
        input_dimension: int,
        output_dimension: int,
        n_hidden_layers: int,
        hidden_size: int,
        activation: Callable[[], nn.Module],
        learning_rate: float,
        domain_extrema: torch.Tensor,
        time_grid_tensor: torch.Tensor,
        train_steps: int,
        batches_per_block: int,
        artifacts_dir: str,
        differentiate_through_drift: bool,
        batch_normalization: bool = False,
        exp_output: bool = False,
        train_jointly: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["activation"])

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.domain_extrema = domain_extrema
        self.time_grid_tensor = time_grid_tensor
        self.train_steps = train_steps
        self.batches_per_block = batches_per_block
        self.artifacts_dir = artifacts_dir
        self.differentiate_through_drift = differentiate_through_drift
        self.batch_normalization = batch_normalization
        self.exp_output = exp_output
        self.train_jointly = train_jointly

        self.forward_net = self._create_nn_ansatz(
            name="forward model", idx=0, backward=False
        )
        self.backward_net = self._create_nn_ansatz(
            name="backward model", idx=1, backward=True
        )

        self.training_step_count = 0
        self.block_count = 0
        self.epoch_count = 0

        # TODO the lightning module should not care about physics and should not handle the log probabilities
        # physics parameters
        self.g = 1.0
        self.dim = 1
        cov = 1.0
        self.log_p_0 = (
            lambda x: (-0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi * cov)))
            + -0.5 * torch.sum((x - 3) ** 2 / cov, dim=-1)[:, None]
        )
        self.log_p_T = (
            lambda x: (-0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi * cov)))
            + -0.5 * torch.sum((x) ** 2 / cov, dim=-1)[:, None]
        )

        # not configurable for now
        self.loss_weights = {
            "pde": 1.0,
            "bc": 10.0,
        }
        self.decay_rate = 0.9993

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        opimizer_forward = torch.optim.Adam(
            self.forward_net.parameters(), lr=self.learning_rate
        )

        lr_scheduler_forward = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=opimizer_forward, gamma=self.decay_rate
        )

        lr_scheduler_config_forward = LRSchedulerConfigType(
            scheduler=lr_scheduler_forward,
            interval="step",
            frequency=1,
        )

        opt_fwd = OptimizerLRSchedulerConfig(
            optimizer=opimizer_forward, lr_scheduler=lr_scheduler_config_forward
        )

        optimizer_backward = torch.optim.Adam(
            self.backward_net.parameters(), lr=self.learning_rate
        )

        lr_scheduler_backward = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer_backward, gamma=self.decay_rate
        )

        opt_bwd = OptimizerLRSchedulerConfig(
            optimizer=optimizer_backward,
            lr_scheduler=lr_scheduler_backward,
        )

        # opt_fwd = torch.optim.Adam(self.forward_net.parameters(), lr=self.learning_rate)
        # opt_bwd = torch.optim.Adam(self.backward_net.parameters(), lr=self.learning_rate)
        return [opt_fwd, opt_bwd]

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        assert isinstance(optimizers, list)
        opt_fwd, opt_bwd = optimizers

        # TODO: we want to be able to configure the number of interior and boundary points
        # TODO: we only need forward or backward points depending on the current block
        # when not training jointly
        x_forward_interior, x_forward_bc = (
            batch["forward_interior"],
            batch["forward_bc"],
        )
        x_backward_interior, x_backward_bc = (
            batch["backward_interior"],
            batch["backward_bc"],
        )
        # Determine which network to train based on current block
        block = (self.training_step_count // self.batches_per_block) % 2
        self.training_step_count += 1
        if self.training_step_count % self.batches_per_block == 0:
            self.block_count += 1
            if self.block_count % 2 == 0:
                self.epoch_count += 1

        if self.train_jointly:
            # ===== Joint Optimization =====
            loss_fwd = self._compute_loss_forward(x_forward_interior, x_forward_bc)
            loss_bwd = self._compute_loss_backward(x_backward_interior, x_backward_bc)
            loss_total = loss_fwd + loss_bwd

            opt_fwd.zero_grad()
            opt_bwd.zero_grad()
            self.manual_backward(loss_total)
            opt_fwd.step()
            opt_bwd.step()

            # step scheduler
            schedulers = self.lr_schedulers()
            if isinstance(schedulers, list):
                for scheduler in schedulers:
                    scheduler.step()

            self.log_dict(
                {
                    # get lr with scheduler
                    "lr": opt_fwd.param_groups[0]["lr"],
                    "loss_total": loss_total,
                    "loss_forward": loss_fwd,
                    "loss_backward": loss_bwd,
                },
                prog_bar=True,
            )
        else:
            # ===== Blockwise Alternating Optimization =====
            if block == 0:
                loss_fwd = self._compute_loss_forward(x_forward_interior, x_forward_bc)

                opt_fwd.zero_grad()
                self.manual_backward(loss_fwd)
                opt_fwd.step()

                self.log("loss_forward", loss_fwd, prog_bar=True)
            else:
                loss_bwd = self._compute_loss_backward(
                    x_backward_interior, x_backward_bc
                )

                opt_bwd.zero_grad()
                self.manual_backward(loss_bwd)
                opt_bwd.step()

                self.log("loss_backward", loss_bwd, prog_bar=True)

    def _create_nn_ansatz(self, name: str, idx: int, backward: bool):
        return NNAnsatz(
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            n_hidden_layers=self.n_hidden_layers,
            hidden_size=self.hidden_size,
            activation=self.activation,
            learning_rate=self.learning_rate,
            domain_extrema=self.domain_extrema,
            time_grid_tensor=self.time_grid_tensor,
            backward=backward,
            differentiate_through_drift=self.differentiate_through_drift,
            batch_normalization=self.batch_normalization,
            exp_output=self.exp_output,
            idx=idx,
        )

    def _compute_boundary_loss_forward(self, x_bc):
        """
        Compute the boundary condition loss for the forward model.
        """
        u_bc_forward = self.forward_net(x_bc)
        # TODO: should we freeze the backward network during this computation?
        # self.backward_net.eval()
        u_bc_backward = self.backward_net(x_bc)

        bc_residual = u_bc_forward + u_bc_backward - self.log_p_T(x_bc[:, 1:])

        bc_loss = torch.mean(bc_residual.pow(2))
        return bc_loss
    
    def _compute_loglik_loss_forward(self, x_bc):
        """
        Compute the log likelihood loss for the forward model. Assumes that x_bc is generated
        by the forward model gradient vector field at time T.
        """
        
        bc_loss = - torch.mean(self.log_p_T(x_bc[:, 1:]))

        return bc_loss

    def _compute_pde_loss_forward(self, x_interior):
        # x_interior is of shape (batch_size, d+1) where the first dimension is time
        x_interior.requires_grad_(True)
        u = self.forward_net(x_interior)

        grad_u_t, grad_u_x, laplace_u = get_derivatives(u, x_interior)

        g = self.g
        pde_residual = (
            grad_u_t
            + 0.5 * g**2 * laplace_u
            + 0.5 * g**2 * grad_u_x.sum(dim=-1).unsqueeze(-1) ** 2
        )
        pde_loss = torch.mean(pde_residual.pow(2))
        return pde_loss

    def _compute_loss_forward(self, x_interior, x_bc):
        """
        Compute the loss for the forward model.
        """
        # Compute PDE residual
        pde_loss = self._compute_pde_loss_forward(x_interior)

        # Compute boundary condition loss
        # bc_loss = self._compute_boundary_loss_forward(x_bc)

        # Compute log likelihood loss
        bc_loss = self._compute_loglik_loss_forward(x_bc)

        # Total loss
        total_loss = (
            self.loss_weights["pde"] * pde_loss + self.loss_weights["bc"] * bc_loss
        )

        return total_loss

    def _compute_pde_loss_backward(self, x_interior):
        x_interior.requires_grad_(True)
        u = self.backward_net(x_interior)

        grad_u_t, grad_u_x, laplace_u = get_derivatives(u, x_interior)

        g = self.g
        pde_residual = (
            grad_u_t
            - 0.5 * g**2 * laplace_u
            - 0.5 * g**2 * grad_u_x.sum(dim=-1).unsqueeze(-1) ** 2
        )
        pde_loss = torch.mean(pde_residual.pow(2))
        return pde_loss

    def _compute_boundary_loss_backward(self, x_bc):
        """
        Compute the boundary condition loss for the backward model.
        """
        u_bc_forward = self.forward_net(x_bc)
        # TODO: should we freeze the forward network during this computation?
        # self.forward_net.eval()
        u_bc_backward = self.backward_net(x_bc)

        bc_residual = u_bc_forward + u_bc_backward - self.log_p_0(x_bc[:, 1:])

        bc_loss = torch.mean(bc_residual.pow(2))
        return bc_loss

    def _compute_loglik_loss_backward(self, x_bc):
        """
        Compute the log likelihood loss for the backward model. Assumes that x_bc is generated
        by the backward model gradient vector field at time 0.
        """
        
        bc_loss = - torch.mean(self.log_p_0(x_bc[:, 1:]))

        return bc_loss
    
    def _compute_loss_backward(self, x_interior, x_bc):
        """
        Compute the loss for the backward model.
        """
        # Compute PDE residual
        pde_loss = self._compute_pde_loss_backward(x_interior)

        # Compute boundary condition loss
        # bc_loss = self._compute_boundary_loss_backward(x_bc)

        # Compute log likelihood loss
        bc_loss = self._compute_loglik_loss_backward(x_bc)

        # Total loss
        total_loss = (
            self.loss_weights["pde"] * pde_loss + self.loss_weights["bc"] * bc_loss
        )

        return total_loss
