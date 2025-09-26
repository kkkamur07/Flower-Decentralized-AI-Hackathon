"""medapp: A Flower / pytorch_msg_api app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Import the module, not the variable
import medapp.task as task
from medapp.task import Net, load_data, get_dataset_config
from medapp.task import test as test_fn
from medapp.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    dataset_name = context.run_config["dataset"]
    in_channels, tfm = get_dataset_config(dataset_name)

    # IMPORTANT: set the module-level variable used by apply_transforms
    task.pytorch_transforms = tfm

    # Load the model and initialize it with the received weights
    model = Net(num_classes=context.run_config["num-classes"], in_channels=in_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    data_path = context.node_config[dataset_name]
    trainloader, _ = load_data(data_path)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    dataset_name = context.run_config["dataset"]
    in_channels, tfm = get_dataset_config(dataset_name)

    # IMPORTANT: set the module-level variable used by apply_transforms
    task.pytorch_transforms = tfm

    # Load the model and initialize it with the received weights
    model = Net(num_classes=context.run_config["num-classes"], in_channels=in_channels)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    data_path = context.node_config[dataset_name]
    _, valloader = load_data(data_path)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(model, valloader, device)

    # Construct and return reply Message
    metrics = {"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(valloader.dataset)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
