import torch
import numpy as np
import datetime
import os


def create_artifacts_dir(project, run_id):
    artifacts_dir = os.path.join("artifacts", project, run_id)
    os.makedirs(artifacts_dir, exist_ok=True)
    # create folders for plots, model checkpoints, losses, errors
    for folder in ["plots", "checkpoints", "losses", "errors", "hyperparameters"]:
        os.makedirs(os.path.join(artifacts_dir, folder), exist_ok=True)

    return artifacts_dir


def get_load_path(project_to_load_models_from):
    # matches the checkpoint file starting with "model-final" in the checkpoints folder
    for filedir in os.listdir(
        f"artifacts/schroedinger_bridge/{project_to_load_models_from}/checkpoints"
    ):
        if filedir.startswith("model-final") and filedir.endswith(".ckpt"):
            return f"artifacts/schroedinger_bridge/{project_to_load_models_from}/checkpoints/{filedir}"
    # if not found, error
    raise FileNotFoundError(
        f"No final model found in project {project_to_load_models_from} checkpoints."
    )


def get_run_id():
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d_%H-%M-%S")
    return now


def d_dim_mesh(dim, lower, upper, n_points):
    mesh_coord = [torch.linspace(lower, upper, n_points) for _ in range(dim)]
    mesh = torch.meshgrid(mesh_coord, indexing="ij")
    XY = np.array([c.flatten() for c in mesh]).T
    # corresponding flattened list of d-dim indices
    idx = np.array(
        [
            c.flatten()
            for c in torch.meshgrid(
                [torch.arange(n_points) for _ in range(dim)], indexing="ij"
            )
        ]
    ).T
    return XY, idx


def calculate_Lp_errors(nn_ansatz, u_mc: np.array, constants, l):
    d = constants["d"]
    domain_extrema = constants["domain_extrema"]
    grid, inds = d_dim_mesh(d, domain_extrema[0], domain_extrema[1], l)

    u_deep_splitting = (
        nn_ansatz(torch.tensor(grid, dtype=torch.float32)).detach().cpu().numpy()
    )
    # recast in cube shape
    u_deep_splitting = u_deep_splitting.reshape(np.ones(d, dtype=int) * l)

    # calculate L_p errors and print in table

    L_p_errors = {}
    mc_p_norm = {}
    L_p_errors_rel = {}

    p_norm = lambda x, p: (np.mean(np.abs(x) ** p) * (l / (l - 1)) ** d * volume) ** (
        1 / p
    )
    inf_norm = lambda x: np.max(np.abs(x))

    for p in [1, 2]:
        volume = (domain_extrema[1] - domain_extrema[0]) ** d
        L_p_error = p_norm(u_mc - u_deep_splitting, p)
        mc_norm = p_norm(u_mc, p)
        L_p_error_rel = L_p_error / mc_norm
        print(f"L_{p} error: {L_p_error}")
        print(f"MC L_{p} norm: {mc_norm}")
        print(f"Relative L_{p} error: {L_p_error_rel}")
        L_p_errors[p] = L_p_error
        mc_p_norm[p] = mc_norm
        L_p_errors_rel[p] = L_p_error_rel

    # L_infty error
    L_infty_error = np.max(np.abs(u_mc - u_deep_splitting))
    mc_norm = inf_norm(u_mc)
    L_infty_error_rel = L_infty_error / mc_norm
    L_p_errors["infty"] = L_infty_error
    mc_p_norm["infty"] = mc_norm
    L_p_errors_rel["infty"] = L_infty_error_rel

    print(f"L_infty error: {L_infty_error}")
    print(f"MC L_infty norm: {mc_norm}")
    print(f"Relative L_infty error: {L_infty_error_rel}")

    return L_p_errors, mc_p_norm, L_p_errors_rel


def dict_to_latex_table(d):
    table = "\\begin{tabular}{|c|c|}\n"
    table += "\\hline\n"
    for key, value in d.items():
        table += f"{key} & {value} \\\\\n"
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    return table


def save_dicts_to_file(dicts, filenames, dir):
    for d, filename in zip(dicts, filenames):
        with open(os.path.join(dir, filename), "w") as f:
            for key, value in d.items():
                f.write(f"{key}: {value}\n")

            f.write(f"{dict_to_latex_table(d)}")


def calculate_and_save_Lp_errors(nn_ansatz, u_mc, constants, l, artifacts_dir):
    L_p_errors, mc_p_norm, L_p_errors_rel = calculate_Lp_errors(
        nn_ansatz, u_mc, constants, l
    )
    save_dicts_to_file(
        [L_p_errors, mc_p_norm, L_p_errors_rel],
        ["Lp_errors.txt", "mc_p_norm.txt", "Lp_errors_rel.txt"],
        artifacts_dir + "/errors",
    )

    return L_p_errors, mc_p_norm, L_p_errors_rel


def save_hyperparams_dicts_to_file(hyperparams, filenames, dir):
    save_dicts_to_file(hyperparams, filenames, dir + "/hyperparameters")


def get_derivatives(u, x_interior):
    grad_u = torch.autograd.grad(outputs=u.sum(), inputs=x_interior, create_graph=True)[
        0
    ]

    grad_u_t = grad_u[:, :, 0].unsqueeze(-1)  # Keep time dimension

    grad_u_x = grad_u[:, :, 1:]

    # TODO : check this
    # TODO: calculate only spacial Laplacian
    laplace_u = torch.autograd.grad(
        outputs=grad_u_x.sum(), inputs=x_interior, create_graph=True
    )[0][:, :, 1:]
    return grad_u_t, grad_u_x, laplace_u
