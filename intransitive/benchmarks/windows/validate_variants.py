"""Run the unchanged tactical assertions with each window variant enabled.

The five historical defensive failures remain ordinary failures and this command
returns failure if any assertion fails. No expected-failure exemptions.
"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intransitive.heuristics import AlphaBetaPlayer, SearchConfig


def main():
    original = AlphaBetaPlayer.__init__
    success = True
    for pvs, aspiration in ((True, False), (False, True), (True, True)):
        print(f'\nPVS={pvs}, aspiration={aspiration}', file=sys.stderr, flush=True)
        def initialize(self, game=None, config=None, use_table=True):
            original(self, game, replace(config or SearchConfig(),
                     pvs_enabled=pvs, aspiration_enabled=aspiration), use_table)
        suite = unittest.defaultTestLoader.loadTestsFromNames([
            'intransitive.tests.test_tactics', 'intransitive.tests.test_game_blunders'])
        with patch.object(AlphaBetaPlayer, '__init__', initialize):
            result = unittest.TextTestRunner(verbosity=2).run(suite)
        success &= result.wasSuccessful()
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
