"""Preserve the Python sources and effective settings of a training invocation."""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4


def backup_run_sources(args, source_root=None):
    """Return a new snapshot path; propagate errors so training cannot run unbacked.

    Resolve sources relative to this module, independent of the working directory.
    ``source_root`` supports testing with a small isolated source tree.
    """
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parent
    if not args.game or Path(args.game).name != args.game or args.game in ('.', '..'):
        raise ValueError(f'Expected a game directory name, got {args.game!r}')
    game_root = root / args.game
    if not game_root.is_dir():
        raise FileNotFoundError(f'Missing game source directory: {game_root}')

    checkpoint = Path(args.checkpoint).resolve()
    backups = checkpoint / 'source_backups'
    sources = sorted(root.glob('*.py'))
    if not sources:
        raise FileNotFoundError(f'No shared Python sources in {root}')

    def raise_walk_error(error):
        raise error

    game_sources = []
    for directory, subdirs, filenames in os.walk(game_root, onerror=raise_walk_error):
        # Avoid caches and an output directory placed inside the game package.
        subdirs[:] = sorted(name for name in subdirs if name not in ('__pycache__', '.git')
                            and (Path(directory) / name).resolve() not in (checkpoint, backups))
        game_sources.extend(Path(directory) / name for name in sorted(filenames) if name.endswith('.py'))
    if not game_sources:
        raise FileNotFoundError(f'No Python sources in {game_root}')
    sources.extend(game_sources)

    backups.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + uuid4().hex
    snapshot = backups / run_id
    settings = str(args) + '\n'
    # Publish only after every required file has been copied successfully.
    with tempfile.TemporaryDirectory(prefix='.pending-', dir=backups) as staging:
        staging = Path(staging)
        for source in sources:
            target = staging / source.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        (staging / 'settings.txt').write_text(settings, encoding='utf-8')
        staging.rename(snapshot)

    # Retain legacy settings too, including those from before snapshots existed.
    current_settings = checkpoint / 'settings.txt'
    if current_settings.exists():
        shutil.copy2(current_settings, checkpoint / f'settings.{run_id}.txt')
    with tempfile.TemporaryDirectory(prefix='.settings-', dir=checkpoint) as staging:
        replacement = Path(staging) / 'settings.txt'
        replacement.write_text(settings, encoding='utf-8')
        replacement.replace(current_settings)
    return snapshot
