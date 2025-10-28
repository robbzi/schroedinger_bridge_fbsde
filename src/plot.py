import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import ForwardBackwardNoiseDataset
from models import LitSchroedingerBridgeFBSDE


def plot_path_batch(paths, time_grid_tensor, title="SDE Paths", save_path=None):
    """
    Plot multiple SDE paths.

    Args:
        paths (torch.Tensor): Simulated paths of the SDE, shape (batch_size, time_steps, d).
        time_grid_tensor (torch.Tensor): Time grid for the simulation.
        title (str): Title of the plot.
        save_path (str): Path to save the plot. If None, the plot will not be saved.
    """

    plt.figure(figsize=(10, 6))
    for i in range(paths.shape[0]):
        plt.plot(time_grid_tensor.numpy(), paths[i].numpy(), label=f"Path {i + 1}")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# def plot_paths_from_prior(
#     simulation_dataset: PathDataset, constants, max_nbr_paths=1000
# ):
#     """
#     Plot one batch of paths from the prior distribution.
#     """
#     d = constants["d"]
#     if d != 1:
#         raise ValueError("This function is only for 1D models.")

#     fig, ax = plt.subplots()
#     T = constants["T"]

#     # get batch
#     batch = next(iter(simulation_dataset))[:max_nbr_paths, :, :]
#     # plot
#     time_grid = np.linspace(0, T, batch.shape[1])
#     for i in range(batch.shape[0]):
#         # make the line width small
#         ax.plot(time_grid, batch[i, :, 1].cpu().numpy(), alpha=0.3, linewidth=0.5)

#     ax.set_xlim(0, T)

#     return fig, ax


def plot_paths_1d_from_prior_and_final(
    schroedinger_bridge_module: LitSchroedingerBridgeFBSDE,
    dataset: ForwardBackwardNoiseDataset,
    constants,
    max_nbr_paths=1000,
):
    """
    Plot one batch of paths from the prior distribution and the final distribution.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    d = constants["d"]
    if d != 1:
        raise ValueError("This function is only for 1D models.")
    T = constants["T"]
    # get batch
    batch = next(iter(dataset))
    forward_paths, backward_paths = schroedinger_bridge_module.generate_paths_only(
        batch
    )
    forward_paths = forward_paths[:max_nbr_paths, :, :]
    backward_paths = backward_paths[:max_nbr_paths, :, :]
    # plot forward paths
    time_grid = np.linspace(0, T, forward_paths.shape[1])
    for i in range(forward_paths.shape[0]):
        # make the line width small
        ax[0].plot(
            time_grid, forward_paths[i, :, 1].detach().cpu().numpy(), alpha=0.3, linewidth=0.5
        )
    ax[0].set_title("Paths from Prior Distribution")
    ax[0].set_xlim(0, T)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Value")

    # plot backward paths
    for i in range(backward_paths.shape[0]):
        # make the line width small
        ax[1].plot(
            time_grid, backward_paths[i, :, 1].detach().cpu().numpy(), alpha=0.3, linewidth=0.5
        )
    ax[1].set_title("Paths from Final Distribution")
    ax[1].set_xlim(0, T)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Value")

    plt.tight_layout()
    return fig, ax

def plot_paths_2d_from_prior_and_final(
    schroedinger_bridge_module: LitSchroedingerBridgeFBSDE,
    dataset: ForwardBackwardNoiseDataset,
    constants,
    max_nbr_paths=1000,
    max_timesteps=10,
    horizontal=True,
):
    """
    Plot one batch of paths from the prior distribution and the final distribution.
    For each time step, there is a scatter plot of the positions of the paths in 2D.
    """
    d = constants["d"]
    if d != 2:
        raise ValueError("This function is only for 2D models.")
    T = constants["T"]
    nbr_time_steps = constants["nbr_time_steps"]
    time_grid = np.linspace(0, T, nbr_time_steps + 1)

    # Adjust subplot layout based on orientation
    if horizontal:
        fig, axs = plt.subplots(2, max_timesteps + 1, figsize=(5 * max_timesteps, 10), sharex=True, sharey=True)
    else:
        fig, axs = plt.subplots(max_timesteps + 1, 2, figsize=(10, 5 * max_timesteps), sharex=True, sharey=True)

    # Set aspect ratio to equal for all subplots
    for ax_row in axs:
        for ax in ax_row:
            ax.set_aspect('equal', 'box')

    # get batch
    batch = next(iter(dataset))
    forward_paths, backward_paths = schroedinger_bridge_module.generate_paths_only(
        batch
    )
    forward_paths = forward_paths[:max_nbr_paths, :, :]
    backward_paths = backward_paths[:max_nbr_paths, :, :]

    time_indices = np.linspace(0, nbr_time_steps, max_timesteps+1, dtype=int)

    # plot forward paths at different time steps
    for i, t_index in enumerate(time_indices):
        if horizontal:
            ax = axs[0, i]
        else:
            ax = axs[i, 0]

        plot_2d_paths_at_t(
            forward_paths,
            t_index,
            ax=ax,
            title=f"Forward Paths at t={time_grid[t_index]:.2f}",
        )

    # plot backward paths at different time steps
    for i, t_index in enumerate(time_indices):
        if horizontal:
            ax = axs[1, i]
        else:
            ax = axs[i, 1]

        plot_2d_paths_at_t(
            backward_paths,
            t_index,
            ax=ax,
            title=f"Backward Paths at t={time_grid[t_index]:.2f}",
            color='orange'
        )

    plt.tight_layout()
    return fig, axs


def plot_models(forward_model, backward_model, nbr_grid_points, constants, max_timesteps=10):
    d = constants["d"]
    nbr_time_steps = constants["nbr_time_steps"]
    time_grid = np.linspace(0, constants["T"], nbr_time_steps + 1)
    time_indices = np.linspace(0, nbr_time_steps, max_timesteps+1, dtype=int)

    n = len(time_indices)
    fig, ax = plt.subplots(n, 4, figsize=(5 * 3, 5 * n))

    for i, model in enumerate([forward_model, backward_model]):
        for j, t_ind in enumerate(time_indices):
            if d == 1:
                plot_1d_sb_model_at_t(
                    model,
                    time_grid[t_ind],
                    nbr_grid_points=nbr_grid_points,
                    constants=constants,
                    ax=ax[j, i],
                    title=f"{'forward' if i == 0 else 'backward'} model output at time {time_grid[t_ind]:.2f}",
                )
            elif d == 2:
                plot_2d_sb_model_at_t(
                    model,
                    time_grid[t_ind],
                    nbr_grid_points=nbr_grid_points,
                    constants=constants,
                    ax=ax[j, i],
                    title=f"{'forward' if i == 0 else 'backward'} model output at time {time_grid[t_ind]:.2f}",
                )
            else:
                raise ValueError("This function is only for 1D or 2D models.")

    # print sum of models and exp of sum of models
    # sum_of_models = lambda x: forward_model(x) + backward_model(x)
    # exp_of_sum = lambda x: torch.exp(sum_of_models(x))
    # for j, t in enumerate(time_grid):
    #     if d == 1:
    #         plot_1d_sb_model_at_t(
    #             sum_of_models,
    #             t,
    #             nbr_grid_points=nbr_grid_points,
    #             constants=constants,
    #             ax=ax[j, 2],
    #             title=f"Sum of models at time {t:.2f}",
    #         )
    #         plot_1d_sb_model_at_t(
    #             exp_of_sum,
    #             t,
    #             nbr_grid_points=nbr_grid_points,
    #             constants=constants,
    #             ax=ax[j, 3],
    #             title=f"Exp of sum of models at time {t:.2f}",
    #         )
        # # elif d == 2:
        # # TODO
        # else:
        #     raise ValueError("This function is only for 1D or 2D models.")

    plt.tight_layout()
    return fig, ax


def plot_1d_sb_model_at_t(model, t: float, nbr_grid_points, constants, ax, title):
    """
    Plot the output of a 1D Schroedinger Bridge model at a specific time t.
    """
    x = np.linspace(
        constants["domain_extrema"][0].numpy(),
        constants["domain_extrema"][1].numpy(),
        nbr_grid_points,
    )
    # concatenate constant t to x
    t_const = np.array([t] * nbr_grid_points)
    tx = np.concatenate(
        (t_const.reshape(-1, 1), x.reshape(-1, 1)), axis=1
    )  # shape (nbr_grid_points, 2)
    # Get model output
    output = (
        model(torch.tensor(tx, dtype=torch.float32).unsqueeze(0))
        .squeeze()
        .detach()
        .numpy()
    )

    ax.plot(x, output, label=f"t={t:.2f}")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Model Output")
    ax.legend()

def plot_1d_paths_at_t(paths, t_index: int, ax, title):
    """
    Plot the values of multiple 1D paths at a specific time index.
    """
    x = paths[:, t_index, 1].detach().cpu().numpy()  # shape (batch_size,)

    ax.hist(x, bins=30, density=True, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()

def plot_2d_sb_model_at_t(model, t, nbr_grid_points, constants, ax, title):
    """
    Plot the output of a 2D Schroedinger Bridge model as a vector field.
    """
    x = np.linspace(
        constants["domain_extrema"][0].numpy(),
        constants["domain_extrema"][1].numpy(),
        nbr_grid_points,
    )
    y = np.linspace(
        constants["domain_extrema"][0].numpy(),
        constants["domain_extrema"][1].numpy(),
        nbr_grid_points,
    )
    X, Y = np.meshgrid(x, y)
    # Create input tensor for model
    t_const = np.array([t] * nbr_grid_points * nbr_grid_points)  # use t parameter
    tx = np.concatenate(
        (t_const.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1
    )  # shape (nbr_grid_points^2, 3)
    # Get model output
    output = model(torch.tensor(tx, dtype=torch.float32))
    U = output[:, 0].detach().cpu().numpy().reshape(nbr_grid_points, nbr_grid_points)
    V = output[:, 1].detach().cpu().numpy().reshape(nbr_grid_points, nbr_grid_points)

    ax.quiver(X, Y, U, V)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
def plot_2d_paths_at_t(paths, t_index: int, ax, title, color='blue'):
    """
    Plot the values of multiple 2D paths at a specific time index as a scatter plot.
    """
    x = paths[:, t_index, 1].detach().cpu().numpy()  # shape (batch_size,)
    y = paths[:, t_index, 2].detach().cpu().numpy()  # shape (batch_size,)

    ax.scatter(x, y, alpha=0.4, color=color)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
