import re
import wandb
import os

def sanitize_model_name(model_name: str) -> str:
    """
    Given the model name, returns a sanitized version of it.
    """
    return re.sub(r"[\"<>:/\|\\?\*\[\]]+", "__", model_name)

def wandb_init(args):
    wandb_name = f"{args.model}/lr{args.learning_rate}/seed{args.seed}"
    checkpoint_path = f"{args.checkpoint_path}/{wandb_name}"
    if not args.wandb_off and args.global_rank == 0:
        wandb.login()
        if not os.path.exists(f"{checkpoint_path}/wandb") :
            os.makedirs(f"{checkpoint_path}/wandb")
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=wandb_name,
            config=args,
            save_code=True,
            dir=checkpoint_path
        )