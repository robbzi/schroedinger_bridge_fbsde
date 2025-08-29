import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal


class InitValDataset(IterableDataset):
    def __init__(
        self, init_val_distribution: torch.distributions.Distribution, batch_size: int
    ):
        super().__init__()
        self.init_val_distribution = init_val_distribution
        self.batch_size = batch_size

    def generate(self):
        while True:
            # TODO: this should be sampled on GPU for speed
            yield self.init_val_distribution.sample((self.batch_size,))

    def __iter__(self):
        return iter(self.generate())


class GaussianDataset(InitValDataset):
    def __init__(
        self, mean: torch.Tensor, cov: torch.Tensor, dim: int, batch_size: int
    ):
        assert mean.shape == (dim,)
        assert cov.shape == (dim, dim)
        self.mean = mean
        self.cov = cov
        self.dim = dim
        self.batch_size = batch_size
        super().__init__(
            init_val_distribution=MultivariateNormal(
                loc=self.mean.to("cuda"), covariance_matrix=self.cov.to("cuda")
            ),
            batch_size=batch_size,
        )

# class GaussianMixtureDataset(InitValDataset):
#     def __init__(
#         self, means_list: list[torch.Tensor], covs_list: list[torch.Tensor], weights: list[float], batch_size: int
#     ):
#         means = torch.stack(means_list)
#         covs = torch.block_diag(*covs_list)
#         self.means = means
#         self.covs = covs
#         self.weights = torch.tensor(weights)
#         self.batch_size = batch_size
#         super().__init__(
#             init_val_distribution=torch.distributions.MixtureSameFamily(
#                 mixture_distribution=torch.distributions.Categorical(self.weights),
#                 component_distribution=torch.distributions.MultivariateNormal(
#                     loc=means.to("cuda"), covariance_matrix=covs.to("cuda")
#                 ),
#             ),
#             batch_size=batch_size,
#         )

class GaussianMixtureDataset(InitValDataset):
    def __init__(
        self,
        means_list: list[torch.Tensor],
        covs_list: list[torch.Tensor],
        weights: list[float],
        batch_size: int,
    ):
        # normalize shapes
        normalized_covs = []
        for c in covs_list:
            if c.dim() == 1:  # e.g. tensor([1.0]) -> make (1,1)
                c = torch.diag(c)
            assert c.dim() == 2 and c.shape[0] == c.shape[1], "Each cov must be (d,d)"
            normalized_covs.append(c)
        means = torch.stack([m.view(-1) for m in means_list])              # (K, D)
        covs = torch.stack(normalized_covs)                                # (K, D, D)
        K, D = means.shape
        assert covs.shape == (K, D, D), "Cov stack must be (K,D,D)"
        self.means = means
        self.covs = covs
        self.weights = torch.as_tensor(weights, dtype=torch.float32)
        assert torch.isclose(self.weights.sum(), torch.tensor(1.0)), "Weights must sum to 1"
        self.batch_size = batch_size
        component_dist = torch.distributions.MultivariateNormal(
            loc=means.to("cuda"), covariance_matrix=covs.to("cuda")
        )
        mixture_dist = torch.distributions.Categorical(self.weights.to("cuda"))
        super().__init__(
            init_val_distribution=torch.distributions.MixtureSameFamily(
                mixture_distribution=mixture_dist,
                component_distribution=component_dist,
            ),
            batch_size=batch_size,
        )

class NoiseDataset(IterableDataset):
    """
    Like path dataset but doesn't solve SDEs.
    Only generates iid d-dim standard Gaussian noise.
    """

    def __init__(
        self,
        n_time_steps: int,
        batch_size: int,
        dim: int,
    ):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.batch_size = batch_size
        self.dim = dim

    def generate(self):
        while True:
            yield torch.randn(
                (self.batch_size, self.n_time_steps, self.dim),
                dtype=torch.float32,
                device=torch.device("cuda"),
            )

    def __iter__(self):
        return iter(self.generate())


class ForwardBackwardNoiseDataset(IterableDataset):
    """
    Generates iid d-dim standard Gaussian noise for both forward and backward paths.
    """

    def __init__(
        self,
        init_val_dataset: InitValDataset,
        final_val_dataset: InitValDataset,
        n_time_steps: int,
        batch_size: int,
        dim: int,
    ):
        super().__init__()
        self.init_val_dataset = init_val_dataset
        self.final_val_dataset = final_val_dataset
        self.n_time_steps = n_time_steps
        self.batch_size = batch_size
        self.dim = dim
        self.forward_noise_dataset = NoiseDataset(n_time_steps, batch_size, dim)
        self.backward_noise_dataset = NoiseDataset(n_time_steps, batch_size, dim)

    def generate(self):
        for forward_noise, backward_noise, init_vals, final_vals in zip(
            self.forward_noise_dataset,
            self.backward_noise_dataset,
            self.init_val_dataset,
            self.final_val_dataset,
        ):
            yield {
                "forward_noise": forward_noise,
                "backward_noise": backward_noise,
                "init_vals": init_vals,
                "final_vals": final_vals,
            }

    def __iter__(self):
        return iter(self.generate())


# class PathDataset(IterableDataset):
#     def __init__(
#         self,
#         init_val_dataset: InitValDataset,
#         sde_solution_func: Callable[[torch.Tensor], torch.Tensor],
#         T: float,
#         time_steps: int,
#         prepend_time_grid: bool = True,
#         backward_paths: bool = False,
#     ):
#         super().__init__()
#         self.sde_solution_func = sde_solution_func
#         self.init_val_dataset = init_val_dataset
#         self.time_grid_tensor = torch.linspace(0, T, time_steps + 1)
#         self.batch_size = init_val_dataset.batch_size
#         self.prepend_time_grid = prepend_time_grid
#         self.backward_paths = backward_paths

#     def generate(self):
#         torch.autograd.set_detect_anomaly(True)
#         for batch in self.init_val_dataset:
#             paths = self.sde_solution_func(batch)
#             if self.backward_paths:
#                 paths = paths.flip(dims=[1])
#             if self.prepend_time_grid:
#                 time_grid = (
#                     self.time_grid_tensor[None, :, None]
#                     .expand(paths.shape[0], -1, -1)
#                     .to(paths.device)
#                 )
#                 yield torch.cat((time_grid, paths), dim=-1)
#             else:
#                 yield paths

#     def __iter__(self):
#         return iter(self.generate())

#     def update_sde_solution_func(
#         self,
#         new_sde_solution_func: Callable[[torch.Tensor], torch.Tensor],
#     ):
#         self.sde_solution_func = new_sde_solution_func


# class BMPathDataset(PathDataset):
#     def __init__(
#         self,
#         init_val_dataset: InitValDataset,
#         T: float,
#         time_steps: int,
#         prepend_time_grid: bool = True,
#         backward_paths: bool = False,
#     ):
#         super().__init__(
#             init_val_dataset=init_val_dataset,
#             sde_solution_func=partial(
#                 brownian_motion_solution,
#                 time_grid_tensor=torch.linspace(0, T, time_steps + 1),
#             ),
#             T=T,
#             time_steps=time_steps,
#             prepend_time_grid=prepend_time_grid,
#             backward_paths=backward_paths,
#         )


# class DriftPathDataset(PathDataset):
#     def __init__(
#         self,
#         init_val_dataset: InitValDataset,
#         drift_func: Callable[[torch.Tensor], torch.Tensor],
#         T: float,
#         time_steps: int,
#         prepend_time_grid: bool = True,
#         backward_paths: bool = False,
#     ):
#         super().__init__(
#             init_val_dataset=init_val_dataset,
#             sde_solution_func=drift_func,
#             T=T,
#             time_steps=time_steps,
#             prepend_time_grid=prepend_time_grid,
#             backward_paths=backward_paths,
#         )


# class ForwardBackwardDataset(IterableDataset):
#     def __init__(self, forward_dataset: PathDataset, backward_dataset: PathDataset):
#         super().__init__()
#         self.forward_dataset = forward_dataset
#         assert backward_dataset.backward_paths, (
#             "Backward paths must be set to True for backward dataset."
#         )
#         self.backward_dataset = backward_dataset

#     def generate(self):
#         for forward_batch, backward_batch in zip(
#             self.forward_dataset, self.backward_dataset
#         ):
#             # x_forward_interior, x_forward_bc = batch['forward_interior'], batch['forward_bc']
#             # x_backward_interior, x_backward_bc = batch['backward_interior'], batch['backward_bc']

#             yield {
#                 "forward_interior": forward_batch,
#                 "backward_interior": backward_batch,
#                 "forward_bc": forward_batch[:, -1, :],
#                 "backward_bc": backward_batch[:, 0, :],
#             }

#     def __iter__(self):
#         return iter(self.generate())


# class StreamDataloader(DataLoader):
#     def __init__(self, dataset: PathDataset, **kwargs):
#         self.path_dataset: PathDataset = dataset
#         # self.device: torch.device = self.dataset.device

#         super().__init__(self.path_dataset, batch_size=None, **kwargs)

#     def update_sde_solution_func(
#         self,
#         new_sde_solution_func: Callable[[torch.Tensor], torch.Tensor],
#     ):
#         self.path_dataset.sde_solution_func = new_sde_solution_func

#     def update_init_val_dataset(self, new_init_val_dataset: InitValDataset):
#         self.path_dataset.init_val_dataset = new_init_val_dataset

#     def train_dataloader(self):
#         return self

#     def val_dataloader(self):
#         return self


class ForwardBackwardDataloader(DataLoader):
    def __init__(self, dataset: ForwardBackwardNoiseDataset, **kwargs):
        self.forward_backward_dataset: ForwardBackwardNoiseDataset = dataset
        # self.device: torch.device = self.dataset.device

        super().__init__(self.forward_backward_dataset, batch_size=None, **kwargs)

    def update_init_val_dataset(self, new_init_val_dataset: InitValDataset):
        # TODO
        pass

    def update_final_val_dataset(self, new_final_val_dataset: InitValDataset):
        # TODO
        pass

    def train_dataloader(self):
        return self

    def val_dataloader(self):
        return self


# if __name__ == "__main__":
#     print(torch.tensor([[1.0, 1.0], [0, 1.0]]).shape)

#     # dataset = GaussianDataset(mean=torch.Tensor([1., -1.]), cov=torch.tensor([[1., 0.5], [0.5, 1.]]), dim=2, batch_size = 5)
#     dataset = BMPathDataset(
#         init_val_dataset=GaussianDataset(
#             mean=torch.Tensor([1.0, -1.0]),
#             cov=torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
#             dim=2,
#             batch_size=1000,
#         ),
#         T=100.0,
#         time_steps=100,
#     )

#     from plot import plot_path_batch

#     batch = next(iter(dataset))
#     plot_path_batch(
#         batch,
#         dataset.time_grid_tensor,
#         title="Brownian Motion Paths",
#         save_path="brownian_motion_paths.png",
#     )
