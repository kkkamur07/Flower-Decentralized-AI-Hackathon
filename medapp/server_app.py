"""medapp: A Flower / pytorch_msg_api app."""

import torch
import wandb
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

# Import the module, not the variable
import medapp.task as task
from medapp.task import Net, load_centralized_dataset, maybe_init_wandb, test, get_dataset_config

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    num_classes: int = context.run_config["num-classes"]
    lr: float = context.run_config["lr"]
    dataset_name = context.run_config["dataset"]
    data_path = context.node_config[dataset_name]

    # Decide channels and set global transform on the module
    in_channels, tfm = get_dataset_config(dataset_name)
    task.pytorch_transforms = tfm

    # Initialize Weights & Biases if set
    use_wandb = context.run_config["use-wandb"]
    wandbtoken = context.run_config.get("wandb-token")
    maybe_init_wandb(use_wandb, wandbtoken)

    # Load global model
    global_model = Net(num_classes=num_classes, in_channels=in_channels)
    arrays = ArrayRecord(global_model.state_dict())

    # Strategy (FedProx; note mu=0 equals FedAvg; set via run-config if desired)
    strategy = FedProx(fraction_train=fraction_train)

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=get_global_evaluate_fn(
            num_classes=num_classes,
            in_channels=in_channels,
            use_wandb=use_wandb,
            data_path=data_path,
        ),
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    out_dir = context.node_config["output_dir"]
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, f"{out_dir}/final_model.pt")


def get_global_evaluate_fn(num_classes: int, in_channels: int, use_wandb: bool, data_path: str):
    """Return an evaluation function for server-side evaluation."""

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate model on central data."""

        # Load model and set device
        model = Net(num_classes=num_classes, in_channels=in_channels)
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load entire test set (uses task.pytorch_transforms set in main)
        test_dataloader = load_centralized_dataset(data_path)

        # Evaluate
        loss, accuracy = test(model, test_dataloader, device=device)
        metric = {"accuracy": accuracy, "loss": loss}

        if use_wandb:
            wandb.log(metric, step=server_round)

        return MetricRecord(metric)

    return global_evaluate
