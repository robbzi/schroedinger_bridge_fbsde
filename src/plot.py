import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import ForwardBackwardNoiseDataset
from models import LitSchroedingerBridgePINN


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


def plot_paths_from_prior_and_final(
    schroedinger_bridge_module: LitSchroedingerBridgePINN,
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


def plot_models(forward_model, backward_model, nbr_grid_points, constants):
    d = constants["d"]
    nbr_time_steps = constants["nbr_time_steps"]
    time_grid = np.linspace(0, constants["T"], nbr_time_steps + 1)
    n = len(time_grid)
    fig, ax = plt.subplots(n, 4, figsize=(5 * 3, 5 * n))

    for i, model in enumerate([forward_model, backward_model]):
        for j, t in enumerate(time_grid):
            if d == 1:
                plot_1d_sb_model_at_t(
                    model,
                    t,
                    nbr_grid_points=nbr_grid_points,
                    constants=constants,
                    ax=ax[j, i],
                    title=f"{'forward' if i == 0 else 'backward'} model output at time {t:.2f}",
                )
            # elif d == 2:
            #     TODO:
            #     plot_2d_sb_model(model, nbr_grid_points=nbr_grid_points, constants=constants, ax=ax[i, 0])
            else:
                raise ValueError("This function is only for 1D or 2D models.")

    # print sum of models and exp of sum of models
    sum_of_models = lambda x: forward_model(x) + backward_model(x)
    exp_of_sum = lambda x: torch.exp(sum_of_models(x))
    for j, t in enumerate(time_grid):
        if d == 1:
            plot_1d_sb_model_at_t(
                sum_of_models,
                t,
                nbr_grid_points=nbr_grid_points,
                constants=constants,
                ax=ax[j, 2],
                title=f"Sum of models at time {t:.2f}",
            )
            plot_1d_sb_model_at_t(
                exp_of_sum,
                t,
                nbr_grid_points=nbr_grid_points,
                constants=constants,
                ax=ax[j, 3],
                title=f"Exp of sum of models at time {t:.2f}",
            )
        # elif d == 2:
        # TODO
        else:
            raise ValueError("This function is only for 1D or 2D models.")

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
