"""
Output directory manager - organize results by git commit.

the output structure will be:
outputs/
  {commit_hash}/
    {timestamp}/
"""

import subprocess
from datetime import datetime
from pathlib import Path


def get_git_commit_hash(short: bool = True) -> str:
    """Get current git commit hash."""
    try:
        cmd = ["git", "rev-parse", "--short" if short else "--long", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_output_dir(base_dir: str = "outputs") -> Path:
    """Get output directory organized by git commit.
    
    Structure: outputs/{commit_hash}/{timestamp}/
    
    Returns:
        Path to the output directory for current run
    """
    commit_hash = get_git_commit_hash()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(base_dir) / commit_hash / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a symlink to "latest" for convenience
    latest_link = Path(base_dir) / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(output_dir.resolve(), target_is_directory=True)
    
    # Also create a symlink to "latest_commit"
    latest_commit_link = Path(base_dir) / "latest_commit"
    if latest_commit_link.is_symlink() or latest_commit_link.exists():
        latest_commit_link.unlink()
    latest_commit_link.symlink_to(
        output_dir.parent.resolve(), target_is_directory=True
    )
    
    return output_dir


def save_run_info(output_dir: Path, extra_info: dict | None = None):
    """Save run metadata to output directory."""
    import json
    
    info = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(short=False),
        "git_branch": get_git_branch(),
        "output_dir": str(output_dir),
    }
    
    if extra_info:
        info.update(extra_info)
    
    info_file = output_dir / "run_info.json"
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    return info_file


def list_commit_results(base_dir: str = "outputs") -> dict:
    """List all results organized by commit.
    
    Returns:
        dict: {commit_hash: [timestamp_dirs]}
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return {}
    
    results = {}
    for commit_dir in sorted(base_path.iterdir()):
        if commit_dir.is_dir() and not commit_dir.name.startswith("."):
            if commit_dir.name in ("latest", "latest_commit"):
                continue
            timestamps = sorted(
                [d.name for d in commit_dir.iterdir() if d.is_dir()]
            )
            results[commit_dir.name] = timestamps
    
    return results


def print_results_summary(base_dir: str = "outputs"):
    """Print summary of all results."""
    results = list_commit_results(base_dir)
    
    if not results:
        print(f"No results found in {base_dir}/")
        return
    
    print(f"\n{'='*60}")
    print(f"Results Summary in {base_dir}/")
    print(f"{'='*60}\n")
    
    for commit, timestamps in sorted(results.items()):
        print(f"Commit: {commit}")
        for ts in timestamps:
            result_dir = Path(base_dir) / commit / ts
            # Check for results
            has_json = any(result_dir.glob("*.json"))
            status = "✓" if has_json else "○"
            print(f"  {status} {ts}")
        print()
    
    print("Symlinks:")
    latest = Path(base_dir) / "latest"
    latest_commit = Path(base_dir) / "latest_commit"
    if latest.exists():
        print(f"  latest -> {latest.resolve().relative_to(Path(base_dir).resolve())}")
    if latest_commit.exists():
        print(f"  latest_commit -> {latest_commit.resolve().relative_to(Path(base_dir).resolve())}")
