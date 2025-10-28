import os
import torch

import pytorch_lightning as pl
from torch import nn

torch.set_float32_matmul_precision("medium")
from pytorch_lightning.callbacks import ModelCheckpoint

torch.autograd.set_detect_anomaly(True)

from datasets import (
    ForwardBackwardDataloader,
    ForwardBackwardNoiseDataset,
    GaussianDataset,
    GaussianPriorDataset,
    Dataset2D,
    GaussianMixtureDataset,
    CircleDistribution2D,
    CrossDistribution2D,
)

from utils import (
    get_load_path,
    get_run_id,
    create_artifacts_dir,
    save_hyperparams_dicts_to_file,
)
from plot import *

from models import LitSchroedingerBridgeFBSDE

assert torch.cuda.is_available()
device = "cuda" if torch.cuda.is_available() else "cpu"

debug_fast = False
load_final_models = False
project_to_load_models_from = (
    "2025-10-28_18-41-09" #<-- good circle to cross
)



checkpointing = False
ckpt_interval = 1000  # steps

if __name__ == "__main__":
    # multiprocessing for data loading
    torch.multiprocessing.set_start_method("spawn")

    num_workers = 0

    # rng seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    pl.seed_everything(42)

    # run id for saving
    run_id = get_run_id()
    artifacts_dir = create_artifacts_dir("schroedinger_bridge_alt", run_id)
    ckpt_dir = f"{artifacts_dir}/checkpoints"

    # wether to use the drift for optimization in boundary conditions
    differentiate_through_drift = False


    # mean_final = torch.Tensor([0.0])
    # cov_final = torch.tensor([[1.0]])

    # parameters
    d = 2
    T = 1
    nbr_time_steps = 100#30
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
    output_dimension = input_dimension
    time_embed_dim = 128
    n_hidden_layers = [1, 3, 1]  # for t, x, and out model
    hidden_size = 256
    activation = nn.SiLU #nn.Sigmoid() #nn.GELU()   # nn.Softplus()  #ReLU2() #  # nn.Tanh
    network_params = {
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "n_hidden_layers": n_hidden_layers,
        "hidden_size": hidden_size,
        "activation": str(activation.__class__.__name__),
    }

    # train parameters
    learning_rate = 5e-4
    batch_size = 1000  # 60000
    batches_per_block = 100
    alt_refresh_rate = 10
    train_steps = 1000  # 10**4
    train_jointly = False
    train_params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batches_per_block": batches_per_block,
        "train_steps": train_steps,
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

    init_val_dataset = Dataset2D(
        distribution=CircleDistribution2D(radius=5.0),
        #distribution=CrossDistribution2D(),
        batch_size=batch_size,
    )

    final_val_dataset = Dataset2D(
        #distribution=CircleDistribution2D(radius=5.0),
        distribution=CrossDistribution2D(),
        batch_size=batch_size,
    )

    dataset = ForwardBackwardNoiseDataset(
        init_val_dataset=init_val_dataset,
        final_val_dataset=final_val_dataset,
        n_time_steps=nbr_time_steps,
        batch_size=batch_size,
        dim=d,
    )

    # create dataloader
    dataloader = ForwardBackwardDataloader(dataset)

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
        model_sb = LitSchroedingerBridgeFBSDE(
            input_dimension=input_dimension,
            output_dimension=output_dimension,
            time_embed_dim=time_embed_dim,
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
            prior_distribution=None,
            train_jointly=train_jointly,
            alt_refresh_rate=alt_refresh_rate,
        )

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
        trainer.fit(model_sb, dataloader)
        # save best model
        if checkpointing:
            best_model_path = checkpoint_callback.best_model_path
            print(f"Best model saved at: {best_model_path}")

        # save final model
        total_loss = trainer.callback_metrics.get("loss_total", None)
        if total_loss is None:
            total_loss = trainer.callback_metrics.get("loss_forward", 0) + trainer.callback_metrics.get("loss_backward", 0)
        final_model_path = f"{ckpt_dir}/model-final-step={trainer.global_step:06d}-loss={total_loss:.4f}.ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        trainer.save_checkpoint(
            f"{ckpt_dir}/model-final-step={trainer.global_step:06d}-loss={total_loss:.4f}.ckpt"
        )
        print(f"Final model saved at: {final_model_path}")

    else:  # load final forward models from a previous run
        print(f"Loading final models from project {project_to_load_models_from}...")
        final_model_path = get_load_path(project_to_load_models_from)
        model_sb = LitSchroedingerBridgeFBSDE.load_from_checkpoint(
            final_model_path, activation=activation
        )
        print("Loaded final forward models.")

    # plot
    if d <= 2:
        fig, ax = plot_models(
            model_sb.forward_net,
            model_sb.backward_net,
            nbr_grid_points=100 if d == 1 else 20,
            constants=constants,
        )
        print(f"Saving plot to {artifacts_dir}/plots/models_final.png")
        fig.savefig(f"{artifacts_dir}/plots/models_final.png")
    else:
        print("Skipping plotting for d > 2, as it is not implemented.")

    # simulating particles from prior
    print("Simulating particles from prior...")
    # normal distribution in d dimensions
    # get brownian motion up to time T

    # plot 100 paths
    paths_to_plot = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for paths in paths_to_plot:
        if d <= 2:
            print("Plotting {paths} paths from prior...")
            if d == 1:
                fig, ax = plot_paths_1d_from_prior_and_final(model_sb, dataset, constants)
            elif d == 2:
                fig, ax = plot_paths_2d_from_prior_and_final(
                    model_sb, dataset, constants, max_nbr_paths=paths, max_timesteps=4, horizontal=False
                )
            print(f"Saving plot to {artifacts_dir}/plots/paths_{paths}.png")
            fig.savefig(f"{artifacts_dir}/plots/paths_{paths}.png")
            #save as svg and pdf
            fig.savefig(f"{artifacts_dir}/plots/paths_{paths}.svg")
            fig.savefig(f"{artifacts_dir}/plots/paths_{paths}.pdf")
            print("Done.")
