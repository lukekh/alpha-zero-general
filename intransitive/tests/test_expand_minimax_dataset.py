import gzip
import hashlib
import json
import pickle
from pathlib import Path
import tempfile
import unittest
import zlib

from intransitive.expand_minimax_dataset import Catalog, run
from intransitive.greedy_process import validate_worker_count


def record(orbit, family=None, split='train', stage='opening', depth=5):
    return dict(orbit=orbit, family=family or orbit, split=split,
                provenance=dict(stage=stage), teacher_depth=depth,
                teacher_reason='maximum_depth', legal=[True, False], action=0)


class CatalogTests(unittest.TestCase):
    def test_explicit_worker_counts(self):
        for count in (1,2,3,4,6,8,64):
            self.assertEqual(validate_worker_count(count),count)
        for count in (0,-1,65,True,4.0,'6'):
            with self.assertRaises(ValueError):
                run('/unused', '/unused', 100, generation_workers=count)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.catalog = Catalog(self.directory / 'test.sqlite3', 300)

    def tearDown(self):
        self.catalog.db.commit()
        self.catalog.close()
        self.temp.cleanup()

    def add(self, rows, origin='extra'):
        with self.catalog.db:
            return self.catalog.insert(rows, origin)

    def test_deduplication_and_primary_precedence(self):
        self.assertEqual(self.add([record('a')]), 1)
        self.assertEqual(self.add([record('a')]), 0)
        self.assertEqual(self.add([record('a')], 'primary'), 1)
        self.assertEqual(self.catalog.counts()['total'], 1)
        self.assertEqual(self.catalog.counts()['origin'], {'primary': 1})

    def test_late_primary_conflict_removes_entire_extra_family(self):
        self.add([record('a', 'family'), record('b', 'family')])
        self.add([record('a', 'primary', split='test')], 'primary')
        self.assertEqual(self.catalog.counts()['total'], 1)
        self.assertEqual(self.add([record('c', 'family')]), 0)
        self.assertEqual(self.catalog.stage_counts['opening'], 1)

    def test_extra_conflict_preserves_primary_and_blocks_family(self):
        self.add([record('a', split='validation')], 'primary')
        self.add([record('b', 'family')])
        self.assertEqual(self.add([record('a', 'family'), record('c', 'family')]), 0)
        self.assertEqual(self.catalog.counts()['total'], 1)
        self.assertEqual(self.catalog.counts()['split'], {'validation': 1})

    def test_quota_and_replacement_with_changed_stage(self):
        self.catalog.quotas['opening'] = 1
        self.add([record('a')])
        self.assertEqual(self.add([record('b')]), 0)
        self.add([record('a', stage='midgame')], 'primary')
        self.assertEqual(self.catalog.stage_counts['opening'], 0)
        self.assertEqual(self.catalog.stage_counts['midgame'], 1)
        self.assertEqual(self.add([record('b')]), 1)

    def test_reject_incomplete_search_and_illegal_target(self):
        with self.assertRaises(ValueError):
            self.add([record('a', depth=4)])
        with self.assertRaises(ValueError):
            self.add([dict(record('b'), action=1)])
        self.assertEqual(self.add([dict(record('c', depth=1), teacher_reason='proven_result')]), 1)

    def test_resume_preserves_payload_and_seed(self):
        row = record('a')
        self.add([row])
        with self.catalog.db:
            self.catalog.set('next_seed', 1234)
        self.catalog.close()
        self.catalog = Catalog(self.directory / 'test.sqlite3', 300)
        self.assertEqual(self.catalog.get('next_seed'), 1234)
        blob = self.catalog.db.execute('SELECT record FROM positions').fetchone()[0]
        self.assertEqual(pickle.loads(zlib.decompress(blob)), row)
        self.assertEqual(self.catalog.stage_counts['opening'], 1)

    def test_import_is_checksummed_and_idempotent(self):
        (self.directory / 'dataset').mkdir()
        raw = gzip.compress(pickle.dumps([record('a')]))
        path = self.directory / 'dataset' / 'shard.pkl.gz'
        path.write_bytes(raw)
        shard = dict(file='dataset/shard.pkl.gz', positions=1, sha256=hashlib.sha256(raw).hexdigest())
        manifest = self.directory / 'dataset-manifest.json'
        manifest.write_text(json.dumps(dict(shards=[shard])))
        self.catalog.import_primary(self.directory)
        self.catalog.import_primary(self.directory)
        self.assertEqual(self.catalog.counts()['total'], 1)
        manifest.write_text(json.dumps(dict(shards=[dict(shard, sha256='changed')])))
        with self.assertRaises(ValueError):
            self.catalog.import_primary(self.directory)


if __name__ == '__main__':
    unittest.main()
