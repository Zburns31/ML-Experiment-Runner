from pathlib import Path


def get_directory(child_dir: str) -> Path:
    path = Path.cwd() / child_dir
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")

    return path
