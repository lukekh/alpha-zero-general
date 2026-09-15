import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import pickle
import sqlite3
import zlib
from unittest.mock import patch

import numpy as np
import torch

from intransitive import supervised_minimax as sm


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = []
        for index, stage in enumerate(sm.STAGES):
            state, provenance = sm.generate_position(2026091600+index, stage)
            legal = sm.IntransitiveGame().getValidMoves(state, 0)
            action = int(np.flatnonzero(legal)[0])
            orbit = sm.orbit_key(state, action, legal)
            cls.rows.append(dict(state=state, legal=legal, action=action, orbit=orbit,
                family=orbit, split='train', teacher_depth=5, teacher_reason='maximum_depth',
                provenance=provenance))

    def test_supplement_preserves_splits_and_is_idempotent(self):
        rows = [dict(r, split=s) for r, s in zip(self.rows, ('train', 'validation', 'test'))]
        accepted, report = sm.select_supplement([], rows)
        self.assertEqual(report['by_split'], dict(train=1, validation=1, test=1))
        self.assertEqual(len(accepted), 3)
        self.assertEqual(sm.select_supplement(rows, rows)[0], [])

    def test_overlap_excludes_whole_related_family(self):
        first, second = self.rows[:2]
        second = dict(second, family=first['family'])
        accepted, report = sm.select_supplement([first], [first, second])
        self.assertEqual(accepted, [])
        self.assertEqual(report['skipped_families'], [first['family']])

    def test_incomplete_teacher_label_is_rejected(self):
        with self.assertRaises(ValueError):
            sm.select_supplement([], [dict(self.rows[0], teacher_depth=4)])

    def test_manifest_checksum_and_record_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path/'dataset').mkdir()
            shard = path/'dataset/shard-00001.pkl.gz'
            sm.save_shard(shard, self.rows)
            manifest = dict(unique_positions=3, shards=[dict(file='dataset/shard-00001.pkl.gz',
                positions=3, sha256=hashlib.sha256(shard.read_bytes()).hexdigest())])
            sm.write_json(path/'dataset-manifest.json', manifest)
            rows, _ = sm.read_corpus(path)
            self.assertEqual(len(rows), 3)
            manifest['shards'][0]['sha256'] = 'wrong'
            sm.write_json(path/'dataset-manifest.json', manifest)
            with self.assertRaises(ValueError):
                sm.read_corpus(path)

    def test_optimizer_resume_matches_uninterrupted_update_and_tracks_samples(self):
        settings = dict(sm.SETTINGS, no_compression=True, batch_size=8)
        original = sm.make_net(settings)
        original.args['batches_per_epoch'] = 1
        for p in original.nnet.value_head.parameters():
            p.requires_grad_(False)
        examples = sm.SymmetryExamples([dict(self.rows[0], provenance=dict(
            self.rows[0]['provenance'], source='tactical_defence'))])
        sm.seed_all(9)
        original.train(examples)
        with tempfile.TemporaryDirectory() as temporary:
            original.save_checkpoint(temporary, 'latest.pt')
            resumed = sm.make_net(settings)
            resumed.load_checkpoint(temporary, 'latest.pt')
            resumed.args['batches_per_epoch'] = 1
            for p in resumed.nnet.value_head.parameters():
                p.requires_grad_(False)
            self.assertEqual(resumed.optimizer_updates, 1)
            self.assertIsNotNone(resumed._pending_optimizer_state)
            for a, b in zip(original.nnet.parameters(), resumed.nnet.parameters()):
                self.assertTrue(torch.equal(a, b))
            sm.seed_all(10)
            original.train(examples)
            examples.sampled_sources.clear()
            sm.seed_all(10)
            resumed.train(examples)
            self.assertEqual(resumed.optimizer_updates, 2)
            self.assertEqual(examples.sampled_sources['tactical_defence'], 8)
            for a, b in zip(original.nnet.parameters(), resumed.nnet.parameters()):
                self.assertTrue(torch.equal(a, b))

    def test_catalog_stream_skips_primary_echoes_and_preserves_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'catalog.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE positions (id INTEGER PRIMARY KEY, orbit, family, split, origin, record)')
                for i, row in enumerate(self.rows):
                    row = dict(row, split='test' if i == 2 else 'train')
                    db.execute('INSERT INTO positions VALUES (?,?,?,?,?,?)',
                        (i+1,row['orbit'],row['family'],row['split'],'primary' if i == 0 else 'extra',
                         zlib.compress(pickle.dumps(row))))
            accepted, report = sm.read_generator_positions(path, [])
            self.assertEqual(len(accepted), 2)
            self.assertEqual(report['by_split'], dict(train=1,test=1))
            self.assertTrue(all(r['ingested_from'] == 'generator_catalog' for r in accepted))
            self.assertFalse(sm.read_generator_positions(path, accepted)[0])
            # IDs are reusable after catalog deletions; no high-water cursor may
            # hide a subsequently inserted position with an older row ID.
            with sqlite3.connect(path) as db:
                db.execute('UPDATE positions SET origin="extra" WHERE id=1')
            fresh, _ = sm.read_generator_positions(path, accepted)
            self.assertEqual([r['orbit'] for r in fresh], [self.rows[0]['orbit']])

    def test_catalog_stream_rejects_conflicting_family(self):
        first, second = self.rows[:2]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'catalog.sqlite3'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE positions (id INTEGER PRIMARY KEY, orbit, family, split, origin, record)')
                for i, row in enumerate((first, second)):
                    row = dict(row, family=first['family'], split='validation')
                    db.execute('INSERT INTO positions VALUES (?,?,?,?,?,?)',
                        (i+1,row['orbit'],row['family'],row['split'],'extra',zlib.compress(pickle.dumps(row))))
            self.assertFalse(sm.read_generator_positions(path, [first])[0])


if __name__ == '__main__':
    unittest.main()
