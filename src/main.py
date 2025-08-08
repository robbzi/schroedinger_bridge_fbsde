import os
import torch

import pytorch_lightning as pl
from torch import nn

torch.set_float32_matmul_precision("medium")
from pytorch_lightning.callbacks import ModelCheckpoint

from datasets import (
    DriftPathDataset,
    ForwardBackwardDataloader,
    ForwardBackwardDataset,
    GaussianDataset,
)

from utils import (
    get_load_path,
    get_run_id,
    create_artifacts_dir,
    save_hyperparams_dicts_to_file,
)
from plot import *

from models import LitSchroedingerBridgePINN

assert torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

debug_fast = False
load_final_models = False
project_to_load_models_from = "2025-08-07_18-29-38"  # good transport from mu=3 to 0
# "2025-08-07_18-22-49" # bad transport from mu=5 to mu=0
# "2025-08-07_17-32-09" # good transport from mu=1 to mu=0
# "2025-08-07_12-59-10"
checkpointing = False
ckpt_interval = 1000  # steps

if __name__ == "__main__":
    # multiprocessing for data loading
    torch.multiprocessing.set_start_method("spawn")
    torch.autograd.set_detect_anomaly(True)

    num_workers = 0

    # rng seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    pl.seed_everything(42)

    # run id for saving
    run_id = get_run_id()
    artifacts_dir = create_artifacts_dir("schroedinger_bridge", run_id)
    ckpt_dir = f"{artifacts_dir}/checkpoints"

    # wether to use the drift for optimization in boundary conditions
    differentiate_through_drift = True

    # specify initial and final distributions for Gaussian dataset
    mean_init = torch.Tensor([3.0])
    cov_init = torch.tensor([[1.0]])
    mean_final = torch.Tensor([0.0])
    cov_final = torch.tensor([[1.0]])

    # parameters
    d = 1
    T = 1
    nbr_time_steps = 10
    domain_extrema = torch.tensor([-10.0, 10.0])
    time_grid_tensor = torch.linspace(0, T, nbr_time_steps + 1)
    constants = {
        "d": d,
        "T": T,
        "nbr_time_steps": nbr_time_steps,
        "domain_extrema": domain_extrema,
    }

    # network parameters
    input_dimension = d
    output_dimension = 1
    n_hidden_layers = 5
    hidden_size = d + 100
    activation = nn.Sigmoid()  # nn.Softplus() #nn.GELU() #ReLU2() #  # nn.Tanh
    network_params = {
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "n_hidden_layers": n_hidden_layers,
        "hidden_size": hidden_size,
        "activation": str(activation.__class__.__name__),
    }

    # train parameters
    learning_rate = 0.01
    batch_size = 1000  # 60000
    batches_per_block = 10
    train_steps = 10000  # 10**4
    block_steps = 10
    train_jointly = True
    train_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batches_per_block": batches_per_block,
        "train_steps": train_steps,
        "block_steps": block_steps,
        "train_jointly": train_jointly,
    }

    save_hyperparams_dicts_to_file(
        [constants, network_params, train_params],
        ["constants", "network_params", "train_params"],
        artifacts_dir,
    )

    if debug_fast:
        batch_size = 1000
        train_steps = 10**1

    init_val_dataset = GaussianDataset(
        mean=mean_init,
        cov=cov_init,
        dim=d,
        batch_size=batch_size,
    )
    final_val_dataset = GaussianDataset(
        mean=mean_final,
        cov=cov_final,
        dim=d,
        batch_size=batch_size,
    )
    # forward_path_dataset = BMPathDataset(
    #     init_val_dataset=init_val_dataset,
    #     T=T,
    #     time_steps=nbr_time_steps,
    # )
    # backward_path_dataset = BMPathDataset(
    #     init_val_dataset=final_val_dataset,
    #     T=T,
    #     time_steps=nbr_time_steps,
    #     backward_paths=True,
    # )

    if not load_final_models:
        # Initialize the model
        model_sb_pinn = LitSchroedingerBridgePINN(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            n_hidden_layers=n_hidden_layers,
            hidden_size=hidden_size,
            activation=activation,
            learning_rate=learning_rate,
            domain_extrema=domain_extrema,
            time_grid_tensor=time_grid_tensor,
            train_steps=train_steps,
            batches_per_block=batches_per_block,
            artifacts_dir=artifacts_dir,
            differentiate_through_drift=differentiate_through_drift,
            train_jointly=train_jointly,
        )

        forward_path_dataset = DriftPathDataset(
            init_val_dataset=init_val_dataset,
            T=T,
            time_steps=nbr_time_steps,
            drift_func=model_sb_pinn.forward_net.drift_sde_solution_func,
        )

        backward_path_dataset = DriftPathDataset(
            init_val_dataset=final_val_dataset,
            T=T,
            time_steps=nbr_time_steps,
            drift_func=model_sb_pinn.backward_net.drift_sde_solution_func,
            backward_paths=True,
        )

        dataset = ForwardBackwardDataset(
            forward_dataset=forward_path_dataset,
            backward_dataset=backward_path_dataset,
        )

        # create dataloader
        dataloader = ForwardBackwardDataloader(dataset)

        # Define callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="model-{step:02d}-{loss_total:.2f}",
            save_top_k=1,
            monitor="loss_total",
            mode="min",
            every_n_train_steps=ckpt_interval,
        )

        # Initialize the trainer
        trainer = pl.Trainer(
            max_epochs=1,  # not implemented yet
            max_steps=train_steps,
            accelerator="gpu",  # Use GPU if available
            devices=1,  # Number of GPUs to use
            callbacks=[checkpoint_callback] if checkpointing else None,
            default_root_dir=artifacts_dir,
        )

        # Train the model
        trainer.fit(model_sb_pinn, dataloader)
        # save best model
        if checkpointing:
            best_model_path = checkpoint_callback.best_model_path
            print(f"Best model saved at: {best_model_path}")

        # save final model
        final_model_path = f"{ckpt_dir}/model-final-step={trainer.global_step:06d}-loss={trainer.callback_metrics.get('loss_total'):.4f}.ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.save_checkpoint(
            f"{ckpt_dir}/model-final-step={trainer.global_step:06d}-loss={trainer.callback_metrics.get('loss_total'):.4f}.ckpt"
        )
        print(f"Final model saved at: {final_model_path}")

    else:  # load final forward models from a previous run
        print(f"Loading final models from project {project_to_load_models_from}...")
        final_model_path = get_load_path(project_to_load_models_from)
        model_sb_pinn = LitSchroedingerBridgePINN.load_from_checkpoint(
            final_model_path, activation=activation
        )
        print("Loaded final forward models.")

    # plot
    if d <= 2:
        fig, ax = plot_models(
            model_sb_pinn.forward_net,
            model_sb_pinn.backward_net,
            nbr_grid_points=100,
            constants=constants,
        )
        print(f"Saving plot to {artifacts_dir}/plots/round_final/forward_final.png")
        os.makedirs(f"{artifacts_dir}/plots/round_final", exist_ok=True)
        fig.savefig(f"{artifacts_dir}/plots/round_final/forward_final.png")
    else:
        print("Skipping plotting for d > 2, as it is not implemented.")

    # simulating particles from prior
    print("Simulating particles from prior...")
    # normal distribution in d dimensions
    # get brownian motion up to time T

    simulation_data_forward = DriftPathDataset(
        init_val_dataset=init_val_dataset,
        T=T,
        time_steps=nbr_time_steps,
        drift_func=model_sb_pinn.forward_net.drift_sde_solution_func,
    )

    simulation_data_backward = DriftPathDataset(
        init_val_dataset=final_val_dataset,
        T=T,
        time_steps=nbr_time_steps,
        drift_func=model_sb_pinn.backward_net.drift_sde_solution_func,
        backward_paths=True,
    )

    # plot 100 paths
    print("Plotting 100 paths from prior...")
    fig, ax = plot_paths_from_prior_and_final(
        simulation_data_forward, simulation_data_backward, constants
    )
    print(f"Saving plot to {artifacts_dir}/plots/paths.png")
    fig.savefig(f"{artifacts_dir}/plots/paths.png")
    print("Done.")
