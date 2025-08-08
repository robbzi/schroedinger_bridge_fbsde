from collections.abc import Callable
import torch


def brownian_motion_solution(
    init_vals: torch.Tensor, time_grid_tensor: torch.Tensor
) -> torch.Tensor:
    # dt can be unevenly spaced
    dt = time_grid_tensor[1:] - time_grid_tensor[:-1]
    # d dimensional noise increments
    noise = torch.randn(
        (init_vals.shape[0], time_grid_tensor.shape[0] - 1, init_vals.shape[1]),
        device=init_vals.device,
    )
    # add zero for the initial value
    noise = torch.cat(
        (
            torch.zeros(
                (init_vals.shape[0], 1, init_vals.shape[1]), device=init_vals.device
            ),
            noise,
        ),
        dim=1,
    )
    # cumulative sum to get the Brownian motion
    noise = torch.cumsum(noise, dim=1)
    # scale the noise by the square root of dt
    noise[:, 1:, :] = noise[:, 1:, :] * torch.sqrt(dt[None, :, None])
    # add the initial value to the noise
    noise = noise + init_vals[:, None, :]
    # return the Brownian motion paths
    return noise


class SDESolver:
    def __init__(self, time_grid_tensor: torch.Tensor):
        """
        Args:
            time_grid_tensor (torch.Tensor): Time grid for the simulation.
        """
        self.time_grid_tensor = time_grid_tensor
        self.dt = time_grid_tensor[1:] - time_grid_tensor[:-1]

    def solve(
        self, init_vals: torch.Tensor, drift: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Solve the SDE using the provided drift function and initial values.

        Args:
            init_vals (torch.Tensor): Initial values for the SDE, shape (batch_size, d).
            drift (Callable[[torch.Tensor], torch.Tensor]): Drift function.

        Returns:
            torch.Tensor: Simulated paths of the SDE.
        """

        # d dimensional noise increments
        noise = torch.randn(
            (
                init_vals.shape[0],
                self.time_grid_tensor.shape[0] - 1,
                init_vals.shape[1],
            ),
            device=init_vals.device,
        )
        # Initialize the path with the initial value
        path = torch.zeros(
            (
                init_vals.shape[0],
                self.time_grid_tensor.shape[0],
                self.init_vals.shape[1],
            ),
            device=self.init_vals.device,
        )
        path[:, 0, :] = init_vals[:, None, :]

        # Iterate over the time steps
        for t in range(1, self.time_grid_tensor.shape[0]):
            # Calculate the drift at the previous time step
            drift_value = drift(path[:, t - 1, :])
            # Update the path using the drift and noise
            path[:, t, :] = (
                path[:, t - 1, :]
                + drift_value * self.dt[t - 1]
                + noise[:, t - 1, :] * torch.sqrt(self.dt[t - 1])
            )
        return path
