import argparse
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from source_backup import backup_run_sources


class SourceBackupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'sources'
        for name in ('main.py', 'Coach.py', 'intransitive/NNet.py',
                     'intransitive/nested/logic.py', 'santorini/NNet.py',
                     'santorini/nested/logic.py', 'santorini/pretrained.pt'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.checkpoint = Path(self.temp.name) / 'run with spaces $(touch injected) `echo hi`'
        self.args = argparse.Namespace(game='intransitive', checkpoint=str(self.checkpoint),
                                       epochs=2, parallel_inferences=1, no_compression=True)

    def test_selected_game_layout_and_literal_settings(self):
        for game in ('intransitive', 'santorini'):
            with self.subTest(game=game):
                self.args.game = game
                snapshot = backup_run_sources(self.args, self.root)
                names = {str(path.relative_to(snapshot)) for path in snapshot.rglob('*') if path.is_file()}
                self.assertEqual(names, {'main.py', 'Coach.py', f'{game}/NNet.py',
                                         f'{game}/nested/logic.py', 'settings.txt'})
                for name in names - {'settings.txt'}:
                    self.assertEqual((snapshot / name).read_bytes(), (self.root / name).read_bytes())
                self.assertEqual((snapshot / 'settings.txt').read_text(), str(self.args) + '\n')
                self.assertEqual((self.checkpoint / 'settings.txt').read_text(), str(self.args) + '\n')

    def test_repeated_runs_retain_legacy_settings_and_source_versions(self):
        self.checkpoint.mkdir()
        legacy = 'Namespace(game="santorini", epochs=1)\n'
        (self.checkpoint / 'settings.txt').write_text(legacy)
        first = backup_run_sources(self.args, self.root)
        first_settings = (first / 'settings.txt').read_text()
        (self.root / 'intransitive/NNet.py').write_text('changed network')
        self.args.epochs = 3
        second = backup_run_sources(self.args, self.root)
        third = backup_run_sources(self.args, self.root)
        self.assertEqual(len({first, second, third}), 3)
        self.assertEqual((first / 'intransitive/NNet.py').read_text(), 'intransitive/NNet.py')
        self.assertEqual((second / 'intransitive/NNet.py').read_text(), 'changed network')
        self.assertEqual((first / 'settings.txt').read_text(), first_settings)
        retained = [path.read_text() for path in self.checkpoint.glob('settings.*.txt')]
        self.assertEqual(len(retained), 3)
        self.assertIn(legacy, retained)
        self.assertIn(first_settings, retained)

    def test_copy_failure_is_visible_and_does_not_publish_partial_backup(self):
        self.checkpoint.mkdir()
        (self.checkpoint / 'settings.txt').write_text('original settings')
        with patch('source_backup.shutil.copy2', side_effect=PermissionError('cannot read source')):
            with self.assertRaisesRegex(PermissionError, 'cannot read source'):
                backup_run_sources(self.args, self.root)
        self.assertEqual(list((self.checkpoint / 'source_backups').iterdir()), [])
        self.assertEqual((self.checkpoint / 'settings.txt').read_text(), 'original settings')

    def test_missing_sources_fail_clearly(self):
        for game in ('missing', 'empty'):
            if game == 'empty':
                (self.root / game).mkdir()
            self.args.game = game
            with self.subTest(game=game), self.assertRaises(FileNotFoundError):
                backup_run_sources(self.args, self.root)
        self.assertFalse(self.checkpoint.exists())

    def test_default_source_root_is_independent_of_working_directory(self):
        self.args.game = 'santorini'
        # Patch cwd lookup without changing the process-wide working directory.
        with patch('os.getcwd', return_value=self.temp.name):
            snapshot = backup_run_sources(self.args)
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((snapshot / 'santorini/NNet.py').read_bytes(),
                         (root / 'santorini/NNet.py').read_bytes())
        self.assertTrue((snapshot / 'source_backup.py').is_file())
        self.assertFalse((snapshot / 'intransitive').exists())

    def test_checkpoint_inside_game_is_not_recursively_backed_up(self):
        for directory in ('runs', '.'):
            with self.subTest(directory=directory):
                self.args.checkpoint = str(self.root / 'intransitive' / directory)
                backup_run_sources(self.args, self.root)
                second = backup_run_sources(self.args, self.root)
                self.assertFalse((second / 'intransitive/source_backups').exists())
                if directory == 'runs':
                    self.assertFalse((second / 'intransitive/runs').exists())


class RunBackupIntegrationTests(unittest.TestCase):
    def test_backup_precedes_learning_failure_stops_learning_and_ray_skips_backup(self):
        # Exercise main.run without requiring Torch/ONNX or starting training.
        coach = Mock()
        modules = {'Coach': types.SimpleNamespace(Coach=coach),
                   'coloredlogs': types.SimpleNamespace(install=Mock()),
                   'GameSwitcher': types.SimpleNamespace(import_game=Mock(
                       return_value=(Mock(__name__='Game'), Mock(__name__='NNet'), None, 2)))}
        spec = importlib.util.spec_from_file_location('backup_test_main', Path(__file__).resolve().parents[1] / 'main.py')
        main = importlib.util.module_from_spec(spec)
        args = argparse.Namespace(game='intransitive', learn_rate=0.001, dropout=0, epochs=2,
                                  batch_size=32, nn_version=1, no_compression=True, q_weight=0.5,
                                  load_model=False, useray=False)
        with patch.dict('sys.modules', modules):
            spec.loader.exec_module(main)
            with patch.object(main, 'backup_run_sources') as backup:
                events = Mock()
                events.attach_mock(backup, 'backup')
                events.attach_mock(coach.return_value.learn, 'learn')
                main.run(args)
                self.assertEqual([call[0] for call in events.mock_calls], ['backup', 'learn'])
                backup.assert_called_once_with(args)
                coach.return_value.learn.reset_mock()
                backup.side_effect = OSError('backup failed')
                with self.assertRaisesRegex(OSError, 'backup failed'):
                    main.run(args)
                coach.return_value.learn.assert_not_called()
                backup.reset_mock()
                args.useray = True
                main.run(args)
                backup.assert_not_called()
                coach.return_value.learn.assert_called_once_with()


if __name__ == '__main__':
    unittest.main()
