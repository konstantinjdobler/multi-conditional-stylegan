import os
import subprocess
import wandb


def get_wandb_api_key():
    try:
        api_key = os.environ.get("WANDB_API_KEY")
    except Exception as e:
        print(e)
        api_key = None
    return api_key


def try_wandb_login():
    WAND_API_KEY = get_wandb_api_key()
    if WAND_API_KEY:
        try:
            subprocess.run(["wandb", "login", WAND_API_KEY], check=True)
            return True
        except Exception as e:
            print(e)
            return False
    else:
        print("WARNING: No wandb API key found, this run will NOT be logged to wandb.")
        input("Press any key to continue...")
        return False


WANDB_PROJECT_NAME = (
    "multi-conditional-stylegan"  # NOTE: mock project, replace with your own
)
WANDB_TEAM_NAME = (
    "hasso-plattner-institute-research"  # NOTE: mock team, replace with your own
)


def start_wandb_logging(args, project=WANDB_PROJECT_NAME):
    if try_wandb_login():
        run_name = args.run_name or args.run_dir.split("/")[-1]
        wandb.init(
            project=project,
            entity=WANDB_TEAM_NAME,
            dir=args.run_dir,
            name=run_name,
            sync_tensorboard=True,
        )

        wandb.config.update(args)
        # wandb.watch(model)


def push_file_to_wandb(filepath):
    wandb.save(filepath)
