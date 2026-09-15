"""Mixed opponent quotas, dedicated workers and unchanged selection semantics."""
import os
from pathlib import Path
import tempfile
import unittest

from intransitive.greedy_process import SETTINGS, GameProcesses, game_tasks, make_net, make_opponent
from intransitive.greedy_training import checkpoint

OPPONENTS = [dict(opponent='AlphaBetaPlayer', opponent_search=dict(max_depth=3)),
             dict(opponent='AlphaBetaPlayer', opponent_search=dict(max_depth=5)),
             dict(opponent='GreedyPlayer'), dict(opponent='ReferenceGreedyPlayer')]


def describe_task(task):
    index, snapshot, settings, seed, side, training = task
    return [], dict(index=index, pid=os.getpid(), model_side=side,
                    opponent=settings['opponent'], config=settings.get('opponent_search'),
                    worker_seconds=0., cpu_seconds=0.)


class MixedProcesses(unittest.TestCase):
    def test_tasks_balance_each_opponent_and_leave_selection_unchanged(self):
        settings = dict(SETTINGS, worker_opponents=OPPONENTS)
        tasks = game_tasks('fixture.pt', list(range(8)), True, settings)
        self.assertEqual([t[4] for t in tasks], [0]*4+[1]*4)
        self.assertEqual([t[2]['opponent'] for t in tasks], [p['opponent'] for p in OPPONENTS]*2)
        self.assertEqual(tasks[1][2]['opponent_search']['max_depth'], 5)
        evaluated = game_tasks('fixture.pt', list(range(20)), False, settings)
        self.assertTrue(all(t[2]['opponent']=='ReferenceGreedyPlayer' for t in evaluated))
        self.assertEqual([t[4] for t in evaluated], [0,1]*10)
        with self.assertRaises(ValueError):
            game_tasks('fixture.pt', list(range(6)), True, settings)

    def test_opponent_factory(self):
        from intransitive.IntransitiveGame import IntransitiveGame
        game = IntransitiveGame()
        for config in OPPONENTS:
            player = make_opponent(game, 123, config)
            self.assertEqual(type(player).__name__, config['opponent'])
            if config['opponent']=='AlphaBetaPlayer':
                self.assertEqual(player.config.max_depth, config['opponent_search']['max_depth'])
        with self.assertRaises(ValueError):
            make_opponent(game, 0, dict(opponent='unknown'))

    def test_four_dedicated_workers_receive_both_colours(self):
        settings = dict(SETTINGS, numMCTSSims=2, worker_opponents=OPPONENTS)
        with tempfile.TemporaryDirectory() as folder:
            path = checkpoint(make_net(settings), folder)
            with GameProcesses(4, snapshot=path, settings=settings, _task=describe_task) as pool:
                rows, _ = pool.collect(path, range(8), True)
                for worker in range(4):
                    pair = [rows[worker][1], rows[worker+4][1]]
                    self.assertEqual([r['pid'] for r in pair], [pool.pids[worker]]*2)
                    self.assertEqual([r['model_side'] for r in pair], [0,1])
                    self.assertEqual(pair[0]['opponent'], OPPONENTS[worker]['opponent'])
            self.assertTrue(all(not p.is_alive() for p in pool.initial_workers))


if __name__ == '__main__':
    unittest.main()
