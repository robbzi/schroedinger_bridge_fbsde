import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from typing import Callable, Optional


from utils import get_derivatives, time_positional_embedding

class FCBlock(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        n_hidden_layers: int,
        hidden_size: int,
        activation: Callable[[], nn.Module],
    ):
        super().__init__()

        layers = []
        if n_hidden_layers == 0:
            layers.append(nn.Linear(input_dimension, output_dimension))
        else:
            layers.append(nn.Linear(input_dimension, hidden_size))
            layers.append(activation())
            for _ in range(n_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(activation())
            layers.append(nn.Linear(hidden_size, output_dimension))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class ResBlock(FCBlock):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        n_hidden_layers: int,
        hidden_size: int,
        activation: Callable[[], nn.Module],
    ):
        assert input_dimension == output_dimension, "For ResBlock, input and output dimensions must be the same."
        super().__init__(
            input_dimension,
            output_dimension,
            n_hidden_layers,
            hidden_size,
            activation,
        )
        self.norm_factor = torch.sqrt(torch.tensor(2.0))

    def forward(self, x):
        return (x + self.model(x))/self.norm_factor

class NNAnsatz(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        time_embed_dim,
        n_hidden_layers,
        hidden_dim,
        activation,
        learning_rate,  # 0.01
        domain_extrema: torch.Tensor,
        time_grid_tensor: torch.Tensor,
        backward: bool,
        differentiate_through_drift: bool,
        batch_normalization: bool,
        idx: Optional[int] = None,
        name: str = "NNAnsatz",
        # optimizer,
        # lr_scheduler,
        # decayRate,
    ):
        super().__init__()

        # TODO: remove this as this is always true
        self.prepend_time_grid = True
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.time_embed_dim = time_embed_dim
        self.hidden_dim = hidden_dim
        
        assert len(n_hidden_layers) == 3, "n_hidden_layers must be a list of three integers for t, x, and out models."
        self.n_hidden_layers_t = n_hidden_layers[0]
        self.n_hidden_layers_x = n_hidden_layers[1]
        self.n_hidden_layers_out = n_hidden_layers[2]
        
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.domain_extrema = domain_extrema
        self.backward = backward
        self.differentiate_through_drift = differentiate_through_drift
        self.learning_rate = learning_rate
        self.idx = idx
        self.name = name
        # self.time_continuous = time_continuous

        self.n_res_blocks = 3
        
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



        # # Define layers
        # layers = []
        # if n_hidden_layers == 0:
        #     layers.append(nn.Linear(input_dimension, output_dimension))
        # else:
        #     layers.append(nn.Linear(input_dimension, hidden_size))
        #     if self.batch_normalization:
        #         layers.append(nn.BatchNorm1d(hidden_size))
        #     layers.append(activation)
        #     for _ in range(n_hidden_layers - 1):
        #         layers.append(nn.Linear(hidden_size, hidden_size))
        #         if self.batch_normalization:
        #             layers.append(nn.BatchNorm1d(hidden_size))
        #         layers.append(activation)
        #     layers.append(nn.Linear(hidden_size, output_dimension))

        # self.model = nn.Sequential(*layers)

        self.model = self._build_model()

        # Xavier initialization
        for model in (self.t_model, self.x_model, self.output_model):
            for layer in model:
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

    def _build_model(self):
        t_layers = []
        t_layers.append(FCBlock(
            input_dimension= self.time_embed_dim,
            output_dimension= self.hidden_dim,
            n_hidden_layers= self.n_hidden_layers_t,
            hidden_size= self.hidden_dim,
            activation= self.activation,
        ))
        self.t_model = nn.Sequential(*t_layers)

        x_layers = []
        x_layers.append(nn.Linear(self.input_dimension, self.hidden_dim))        
        x_layers.extend([ResBlock(
            input_dimension= self.hidden_dim,
            output_dimension= self.hidden_dim,
            n_hidden_layers= self.n_hidden_layers_x,
            hidden_size= self.hidden_dim,
            activation= self.activation,
        ) for _ in range(self.n_res_blocks)])

        self.x_model = nn.Sequential(*x_layers)

        output_layers = []
        # output_layers.append(ResBlock(
        #     input_dimension= self.hidden_dim,
        #     output_dimension= self.hidden_dim,
        #     n_hidden_layers= self.n_hidden_layers_out,
        #     hidden_size= self.hidden_dim,
        #     activation= self.activation,
        # ))
        output_layers.append(FCBlock(
            input_dimension= self.hidden_dim,
            output_dimension= self.output_dimension,
            n_hidden_layers= self.n_hidden_layers_out,
            hidden_size= self.hidden_dim,
            activation= self.activation,
        ))

        self.output_model = nn.Sequential(*output_layers)




    def forward(self, x):
        # assert x does not contain nan
        #assert not torch.isnan(x).any(), "Input contains NaN values."
        t = x[..., 0]
        x_ = x[..., 1:]
        t_embed = time_positional_embedding(t, self.time_embed_dim)
        t_out = self.t_model(t_embed)
        x_out = self.x_model(x_)
        out =  self.output_model(t_out + x_out)
        #assert not torch.isnan(out).any(), "Output contains NaN values."
        return out

    def grad_forward(self, x, create_graph: bool = False, spacial_only: bool = False):
        """
        Gradient of the forward pass.
        """
        # x.requires_grad = True
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
        
   
    def forward_and_divergence(self, x, create_graph: bool = False):
        """
        Forward pass and divergence of the forward pass.
        """
        # x.requires_grad = True
        y = self.forward(x)
        e = torch.randn_like(y)
        grad = torch.autograd.grad(
            y,
            x,  # only spatial dimensions
            e,
            create_graph=create_graph,
        )[0][:, :, 1:]

        div = grad*e
        return y, div

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
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Drift SDE solution function. Assumes particles are transported by the vector field
        given by grad_log_forward.
        """
        # self.model.eval()
        # self.model.to("cuda")
        # TODO: ensure init_vals are generated on cuda, and copy constructed on cuda
        # init_vals = init_vals.to("cuda")
        assert init_vals.device == torch.device("cuda", index=0)
        assert noise.device == torch.device("cuda", index=0)
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

        # TODO: do we need to move the model to the device? Maybe to it externally
        ## initialize X with zeros
        # X = torch.zeros(
        #     (init_vals.shape[0], self.time_grid_tensor.shape[0], init_vals.shape[1]),
        #     device=init_vals.device,
        # )
        # # set initial value
        # X[:, 0, :] = init_vals
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
            drift = self.forward(
                X_for_forward,
            )
            # Euler–Maruyama step
            x_next = (
                x_prev
                + drift * current_dt
                + torch.sqrt(current_dt) * noise[:, t - 1, :]
            )
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


class LitSchroedingerBridgeFBSDE(pl.LightningModule):
    def __init__(
        self,
        # optimizer,
        # lr_scheduler,
        # decayRate,
        input_dimension: int,
        output_dimension: int,
        time_embed_dim: int,
        n_hidden_layers: list[int], # for t, x, and out model
        hidden_size: int,
        activation: Callable[[], nn.Module],
        learning_rate: float,
        domain_extrema: torch.Tensor,
        time_grid_tensor: torch.Tensor,
        train_steps: int,
        batches_per_block: int,
        artifacts_dir: str,
        differentiate_through_drift: bool,
        prior_distribution: Optional[torch.distributions.Distribution] = None,
        batch_normalization: bool = False,
        train_jointly: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["activation"])

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.time_embed_dim = time_embed_dim
        self.n_hidden_layers = n_hidden_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.learning_rate = learning_rate
        self.domain_extrema = domain_extrema
        self.time_grid_tensor = time_grid_tensor
        self.dt = time_grid_tensor[1] - time_grid_tensor[0]
        self.train_steps = train_steps
        self.batches_per_block = batches_per_block
        self.artifacts_dir = artifacts_dir
        self.differentiate_through_drift = differentiate_through_drift
        self.batch_normalization = batch_normalization
        self.train_jointly = train_jointly

        self.prior_distribution = prior_distribution

        self.forward_net = self._create_nn_ansatz(
            name="forward model", idx=0, backward=False
        )
        self.backward_net = self._create_nn_ansatz(
            name="backward model", idx=1, backward=True
        )

        self.training_step_count = 0
        self.block_count = 0
        self.epoch_count = 0

        self.decay_rate = 0.995

        # TODO: this should be defined externally
        self.g = 1.0  # diffusion coefficient

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def configure_optimizers(self):
        opimizer_forward = torch.optim.AdamW(
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

        optimizer_backward = torch.optim.AdamW(
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
        # create paths
        # x_forward_paths = self._create_paths(batch, backward=False)
        # x_backward_paths = self._create_paths(batch, backward=True)
        # x_forward_bc = x_forward_paths[:, -1, :]
        # x_backward_bc = x_backward_paths[:, 0, :]

        # x_init_vals = x_forward_paths[:, 0, :].detach()
        # x_final_vals = x_backward_paths[:, -1, :].detach()



        if self.train_jointly:
            # ===== Joint Optimization =====

            x_forward_paths = self._create_paths(batch, backward=False)

            joint_loss = self._compute_joint_loss(x_forward_paths)

            opt_fwd.zero_grad()
            opt_bwd.zero_grad()
            self.manual_backward(joint_loss)
            torch.nn.utils.clip_grad_norm_(self.forward_net.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.backward_net.parameters(), 1.0)
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
                    "loss_joint": joint_loss.item(),
                },
                prog_bar=True,
            )
        else:
            # Determine which network to train based on current block
            block = (self.training_step_count // self.batches_per_block) % 2
            self.training_step_count += 1
            if self.training_step_count % self.batches_per_block == 0:
                self.block_count += 1
                if self.block_count % 2 == 0:
                    self.epoch_count += 1
            # ===== Blockwise Alternating Optimization =====
            if block == 0:

                # loglik loss
                loss_loglik_forward = self.loss_weights["loglik"] *self._compute_loglik_loss_forward(x_final_vals)
                assert False, "fix this first"
                loss_fwd, loss_fwd_pde, loss_fwd_bc = self._compute_alt_loss_backward(x_backward_paths, x_backward_bc)#self._compute_loss_forward(x_forward_paths, x_forward_bc)

                opt_fwd.zero_grad()
                self.manual_backward(loss_fwd)
                opt_fwd.step()

                # step scheduler
                schedulers = self.lr_schedulers()
                if isinstance(schedulers, list):    
                    schedulers[0].step()

                self.log_dict(
                    {
                        "lr": opt_fwd.param_groups[0]["lr"],
                        #"loss_fwd_total": loss_fwd + loss_loglik_forward,
                        "loss_loglik_forward": loss_loglik_forward,
                        #"loss_forward": loss_fwd,
                        "loss_forward_pde": loss_fwd_pde,
                        #"loss_forward_bc": loss_fwd_bc,
                    },
                    prog_bar=True,
                )
            else:
                # loglik loss
                loss_loglik_backward = self.loss_weights["loglik"] * self._compute_loglik_loss_backward(x_init_vals)

                loss_bwd, loss_bwd_pde, loss_bwd_bc = self._compute_alt_loss_forward(x_forward_paths, x_forward_bc)#self._compute_loss_backward(x_backward_paths, x_backward_bc)

                opt_bwd.zero_grad()
                self.manual_backward(loss_bwd)
                opt_bwd.step()


                # step scheduler
                schedulers = self.lr_schedulers()
                if isinstance(schedulers, list):    
                    schedulers[1].step()

                self.log_dict(
                    {
                        "lr": opt_bwd.param_groups[0]["lr"],
                        #"loss_bwd_total": loss_bwd + loss_loglik_backward,
                        "loss_loglik_backward": loss_loglik_backward,
                        #"loss_backward": loss_bwd,
                        "loss_backward_pde": loss_bwd_pde,
                        #"loss_backward_bc": loss_bwd_bc,
                    },
                    prog_bar=True,
                )

    def _create_paths(self, batch, backward: bool):
        if backward:
            ansatz = self.backward_net
            # we want an increasing time grid for the final path
            time_grid_tensor = self.backward_net.time_grid_tensor.flip(dims=[0])
            init_val_batch = batch["final_vals"]
            noise = batch["backward_noise"]
        else:
            ansatz = self.forward_net
            time_grid_tensor = self.forward_net.time_grid_tensor
            init_val_batch = batch["init_vals"]
            noise = batch["forward_noise"]
        paths = ansatz.drift_sde_solution_func(init_val_batch, noise)
        if backward:
            paths = paths.flip(dims=[1])

        if ansatz.prepend_time_grid:
            time_grid = (
                time_grid_tensor[None, :, None]
                .expand(paths.shape[0], -1, -1)
                .to(paths.device)
            )
            return torch.cat((time_grid, paths), dim=-1)
        else:
            return paths

    def generate_paths_only(self, batch):
        """Generate paths only without computing the loss."""
        self.to(batch["init_vals"].device)
        forward_paths = self._create_paths(batch, backward=False)
        backward_paths = self._create_paths(batch, backward=True)

        return forward_paths, backward_paths

    def _create_nn_ansatz(self, name: str, idx: int, backward: bool):
        return NNAnsatz(
            input_dimension=self.input_dimension,
            output_dimension=self.output_dimension,
            n_hidden_layers=self.n_hidden_layers,
            hidden_dim=self.hidden_size,
            activation=self.activation,
            time_embed_dim=self.time_embed_dim,
            learning_rate=self.learning_rate,
            domain_extrema=self.domain_extrema,
            time_grid_tensor=self.time_grid_tensor,
            backward=backward,
            differentiate_through_drift=self.differentiate_through_drift,
            batch_normalization=self.batch_normalization,
            idx=idx,
        )

    def _compute_joint_loss(self, x_interior):
        """
        Compute the joint loss for the Schrödinger Bridge.
        """
        assert self.prior_distribution is not None, "Prior distribution must be provided for joint loss computation."

        batch_size = x_interior.shape[0]
        dt = self.forward_net.time_grid_tensor[1:] - self.forward_net.time_grid_tensor[:-1]

        # divergence of backward net
        z_hat, div_z_hat = self.backward_net.forward_and_divergence(x_interior, create_graph=self.differentiate_through_drift)
        z = self.forward_net(x_interior)
        loss = 0.5*(z+z_hat).pow(2) + self.g*div_z_hat
        loss = (self.dt*loss).sum() / batch_size
        loss = loss - self.prior_distribution.log_prob(x_interior[:, -1, 1:]).mean()

        return loss

    def _compute_alt_loss_forward(self, x_interior):
        """
        Compute the loss for the forward model.
        """

        batch_size = x_interior.shape[0]
        dt = self.forward_net.time_grid_tensor[1:] - self.forward_net.time_grid_tensor[:-1]

        # divergence of backward net
        z_hat, div_z_hat = self.backward_net.forward_and_divergence(x_interior, create_graph=self.differentiate_through_drift)
        z = self.forward_net(x_interior)
        loss = 0.5*(z_hat).pow(2) + self.g*div_z_hat + z*z_hat
        loss = (self.dt*loss).sum() / batch_size
        loss = loss - self.prior_distribution.log_prob(x_interior[:, -1, 1:]).mean()
        assert not torch.isnan(loss), "Loss is NaN."
        return loss 
    
    def _compute_alt_loss_backward(self, x_interior, x_bc):
        """
        Compute the loss for the backward model.
        """
        # Compute PDE residual
        # pde_loss = self._compute_pde_loss_backward(x_interior.detach())
        # dummy pde loss
        pde_loss = torch.tensor(0.0, device=x_interior.device)

        # Compute boundary condition loss
        # bc_loss = self._compute_boundary_loss_backward(x_bc)

        # Compute log naive likelihood loss
        # bc_loss = self._compute_naive_loglik_loss_backward(x_bc)

        # Compute log likelihood loss (FBSDE loss)
        # bc_loss = self._compute_loglik_fbsde_loss_backward(x_interior)

        # dummy log likelihood loss
        bc_loss = torch.tensor(0.0, device=x_interior.device)

        # Total loss
        total_loss = (
            self.loss_weights["pde"] * pde_loss + self.loss_weights["bc"] * bc_loss
        )

        return total_loss, pde_loss, bc_loss
    
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

    def _compute_naive_loglik_loss_forward(self, x_bc):
        """
        Compute the log likelihood loss for the forward model. Assumes that x_bc is generated
        by the forward model gradient vector field at time T.
        """

        bc_loss = -torch.mean(self.log_p_T(x_bc[:, 1:]))

        return bc_loss

    def _compute_loglik_fbsde_loss_forward(self, x_forward_paths):
        """
        Compute the log likelihood loss for the forward model based on the loss of Theorem 4.
        Assumes knowledge of final distribution p_T.
        """
        # x_forward_paths is of shape (batch_size, nbr_time_steps, d+1)
        # where the first dimension is time
        # TODO: remove: x_forward_paths.requires_grad_(True)
        u = self.forward_net(x_forward_paths)
        u_hat = self.backward_net(x_forward_paths)

        grad_u_t, grad_u_x, laplace_u = get_derivatives(u, x_forward_paths)

        # x_forward_paths.requires_grad_(True)
        grad_u_hat_t, grad_u_hat_x, laplace_u_hat = get_derivatives(u_hat, x_forward_paths)

        dt = self.forward_net.time_grid_tensor[1:] - self.forward_net.time_grid_tensor[:-1]
        g = self.g
        # assert dt is only positive
        assert torch.all(dt > 0) 
        
        integrand = 0.5*(grad_u_x+grad_u_hat_x).pow(2)+g**2 * laplace_u_hat
        # integral = torch.sum(
        #     # left endpoint Riemann sum
        #     #integrand[:, :-1, :] * dt[None, :, None], dim=1
        #     # right endpoint Riemann sum
        #     # integrand[:, 1:, :] * dt[None, :, None], dim=1
        # )
        # trapezoidal rule
        integral = torch.sum(
            0.5 * (integrand[:, :-1, :] + integrand[:, 1:, :]) * dt[None, :, None], dim=1
        )

        loglik = self.log_p_T(x_forward_paths[:, -1, 1:])
        # dummy log likelihood loss
        # loglik = torch.zeros((x_forward_paths.shape[0], 1), device=x_forward_paths.device)
        bc_loss = -loglik + integral
        return bc_loss.mean()

    def _compute_loglik_fbsde_loss_backward(self, x_backward_paths):
        """
        Compute the log likelihood loss for the backward model based on the loss of Theorem 4.
        Assumes knowledge of initial distribution p_0.
        """
        # x_backward_paths is of shape (batch_size, nbr_time_steps, d+1)
        # where the first dimension is time
        u_hat = self.backward_net(x_backward_paths)
        u = self.forward_net(x_backward_paths)

        grad_u_hat_t, grad_u_hat_x, laplace_u_hat = get_derivatives(u_hat, x_backward_paths)
        # x_backward_paths.requires_grad_(True)
        grad_u_t, grad_u_x, laplace_u = get_derivatives(u, x_backward_paths)
        dt = self.backward_net.time_grid_tensor[1:] - self.backward_net.time_grid_tensor[:-1]
        g = self.g
        # assert dt is only negative
        assert torch.all(dt < 0)
        dt = torch.abs(dt)
        integrand = 0.5 * (grad_u_x + grad_u_hat_x).pow(2) + g**2 * laplace_u
        # integral = torch.sum(
        #     # left endpoint Riemann sum (since last step is final distribution)
        #     integrand[:, :-1, :] * dt[None, :, None], dim=
        #     # right endpoint Riemann sum
        #     #integrand[:, 1:, :] * dt[None, :, None], dim=1
        # )
        # trapezoidal rule
        integral = torch.sum(
            0.5 * (integrand[:, :-1, :] + integrand[:, 1:, :]) * dt[None, :, None], dim=1
        )
        loglik = self.log_p_0(x_backward_paths[:, 0, 1:])
        # dummy log likelihood loss
        # loglik = torch.zeros((x_backward_paths.shape[0], 1), device=x_backward_paths.device)
        bc_loss = -loglik + integral
        return bc_loss.mean()

    def _compute_pde_loss_forward(self, x_interior):
        # x_interior is of shape (batch_size, d+1) where the first dimension is time
        assert x_interior.is_leaf, "x_interior must be a leaf tensor."
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


    def _compute_pde_loss_backward(self, x_interior):
        assert x_interior.is_leaf, "x_interior must be a leaf tensor."
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

    def _compute_naive_loglik_loss_backward(self, x_bc):
        """
        Compute the log likelihood loss for the backward model. Assumes that x_bc is generated
        by the backward model gradient vector field at time 0.
        """

        bc_loss = -torch.mean(self.log_p_0(x_bc[:, 1:]))

        return bc_loss

    def _compute_loglik_loss_jointly(self, x_boundary_vals, constrained=True):
        """
        Compute the log likelihood loss.
        """
        # x_bc is of shape (batch_size, d+1) where the first dimension is time
        # we assume that x_bc is generated by the backward model gradient vector field at time 0
        # and we want to compute the log likelihood of the initial distribution p_0
        loglik = self.forward_net(x_boundary_vals) + self.backward_net(x_boundary_vals) 
        t = x_boundary_vals[0, 0]
        if constrained:
            # make sure the sum is a density by integration 
            grid = torch.linspace(
                self.domain_extrema[0], self.domain_extrema[1], steps=100
            ).to(x_boundary_vals.device)
            # concatenate time of boundary values with grid
            grid = torch.cat(
                (
                    t.unsqueeze(0).repeat(grid.shape[0], 1),
                    grid.reshape(-1, 1),
                ),
                dim=-1,
            )
            
            # intergrate the likelihood over the grid
            integral = torch.trapezoid(
                torch.exp(self.forward_net(grid)+self.backward_net(grid)).squeeze(1), x = grid[:, 1], dim=-1
            )
            residue = (torch.sum(integral)-1.0).pow(2)
            #residue = torch.sum(integral) - 1.0
        else:
            residue = torch.tensor(0.0, device=x_boundary_vals.device)

        loglik = loglik.mean()

        return -loglik + residue
    
    def _compute_loglik_loss_forward(self, x_final_vals, constrained=True):
        """
        Compute the log likelihood loss for the forward model.
        """
        # x_final_vals is of shape (batch_size, d+1) where the first dimension is time = 1

        loglik = self.forward_net(x_final_vals) #+ self.backward_net(x_final_vals)
        t = x_final_vals[0, 0]
        if constrained:
            # make sure the sum is a density by integration 
            grid = torch.linspace(
                self.domain_extrema[0], self.domain_extrema[1], steps=100
            ).to(x_final_vals.device)
            # concatenate time of boundary values with grid
            grid = torch.cat(
                (
                    t.unsqueeze(0).repeat(grid.shape[0], 1),
                    grid.reshape(-1, 1),
                ),
                dim=-1,
            )
            
            # intergrate the likelihood over the grid
            integral = torch.trapezoid(
                torch.exp(self.forward_net(grid)+self.backward_net(grid)).squeeze(1), x = grid[:, 1], dim=-1
            )
            residue = (torch.sum(integral)-1.0).pow(2)
            #residue = torch.sum(integral) - 1.0
        else:
            residue = torch.tensor(0.0, device=x_final_vals.device)

        loglik = loglik.mean()

        return -loglik + residue

    def _compute_loglik_loss_backward(self, x_init_vals, constrained=True):
        """
        Compute the log likelihood loss for the backward model.
        """
        # x_init_vals is of shape (batch_size, d+1) where the first dimension is time = 0

        loglik = self.backward_net(x_init_vals)
        t = x_init_vals[0, 0]
        if constrained:
            # make sure the sum is a density by integration 
            grid = torch.linspace(
                self.domain_extrema[0], self.domain_extrema[1], steps=100
            ).to(x_init_vals.device)
            # concatenate time of boundary values with grid
            grid = torch.cat(
                (
                    t.unsqueeze(0).repeat(grid.shape[0], 1),
                    grid.reshape(-1, 1),
                ),
                dim=-1,
            )
            
            # intergrate the likelihood over the grid
            integral = torch.trapezoid(
                torch.exp(self.forward_net(grid)+self.backward_net(grid)).squeeze(1), x = grid[:, 1], dim=-1
            )
            residue = (torch.sum(integral)-1.0).pow(2)
            #residue = torch.sum(integral) - 1.0
        else:
            residue = torch.tensor(0.0, device=x_init_vals.device)

        loglik = loglik.mean()
        return -loglik + residue

   