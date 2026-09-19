"""Browser-session checks against actual engine transitions and history."""

import unittest
import json
import socket
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from intransitive.IntransitiveConstants import parse_coordinate
from intransitive.play import GameSession, PlayServer
from intransitive.IntransitiveLogicNumba import Board


class FirstLegalOpponent:
    label = 'Test model'

    def __init__(self):
        self.reloads = 0
        self.fail = False

    def reload(self):
        if self.fail:
            raise RuntimeError('Unavailable checkpoint')
        self.reloads += 1

    def choose(self, state, player):
        if self.fail:
            raise RuntimeError('Inference failed')
        board = Board()
        board.copy_state(state, True)
        return next(i for i, valid in enumerate(board.valid_moves(player)) if valid)


class PlaySessionTests(unittest.TestCase):
    def setUp(self):
        self.game = GameSession()

    def move(self, source, target):
        source, target = list(parse_coordinate(source)), list(parse_coordinate(target))
        move = next(m for m in self.game.snapshot()["legal"]
                    if m["source"] == source and m["target"] == target)
        return self.game.update("move", dict(action=move["action"], revision=self.game.revision))

    def test_capture_and_undo_restore_exact_history(self):
        for source, target in [("C5", "D5"), ("H5", "G4"), ("D5", "E5"), ("G4", "F5")]:
            self.move(source, target)
        before = self.game.board.get_state().tobytes()
        state = self.move("E5", "F5")
        self.assertEqual(state["board"][4][5], 2)
        self.assertEqual(state["noncapture"], 0)
        self.assertEqual(state["moves"][-1], "E5 × F5")
        self.game.update("undo", dict(revision=self.game.revision))
        self.assertEqual(self.game.board.get_state().tobytes(), before)

    def test_repetition_continues_and_undo(self):
        for _ in range(2):
            for source, target in [("B5", "B6"), ("H5", "H4"), ("B6", "B5"), ("H4", "H5")]:
                state = self.move(source, target)
        self.assertEqual(state["reason"], "ongoing")
        self.assertIsNone(state["winner"])
        self.assertEqual(state["repetition"], 3)
        self.assertTrue(state["legal"])
        self.move('B5', 'B6')
        state = self.game.update("undo", dict(revision=self.game.revision))
        self.assertEqual(state["reason"], "ongoing")
        self.assertTrue(state["legal"])

    def test_long_play_counters_undo_and_ai_after_cutoff(self):
        for _ in range(10):
            for source, target in [('B5', 'B6'), ('H5', 'H4'), ('B6', 'B5'), ('H4', 'H5')]:
                state = self.move(source, target)
        self.assertEqual((state['ply'], state['noncapture'], state['repetition']), (40, 40, 11))
        self.assertEqual(state['reason'], 'ongoing')
        self.game.opponent = FirstLegalOpponent()
        self.game.human_player = 1
        # The test opponent uses a modelling Board: it must receive a usable
        # search observation even though the physical history exceeds its limit.
        state = self.game.update('ai', dict(revision=self.game.revision))
        self.assertEqual(state['ply'], 41)
        self.game.opponent = None
        state = self.game.update('undo', dict(revision=self.game.revision))
        self.assertEqual((state['noncapture'], state['repetition']), (40, 11))

    def test_invalid_and_stale_moves_are_atomic(self):
        self.move("B5", "B6")
        before = self.game.snapshot()
        for data in [dict(action=-1, revision=1), dict(action=True, revision=1),
                     dict(action=before["legal"][0]["action"], revision=0)]:
            with self.assertRaises(ValueError):
                self.game.update("move", data)
            self.assertEqual(self.game.snapshot(), before)

    def test_restart_clears_history_and_counters(self):
        initial = self.game.board.get_state().tobytes()
        self.move("B5", "B6")
        state = self.game.update("restart", dict(revision=1))
        self.assertEqual(self.game.board.get_state().tobytes(), initial)
        self.assertEqual(state["moves"], [])
        self.assertEqual(state["ply"], 0)
        with self.assertRaises(ValueError):
            self.game.update("undo", dict(revision=2))


class AIPlaySessionTests(unittest.TestCase):
    def setUp(self):
        self.opponent = FirstLegalOpponent()
        self.game = GameSession(self.opponent)

    def update(self, command, **data):
        return self.game.update(command, dict(revision=self.game.revision, **data))

    def human_move(self):
        return self.update('move', action=self.game.snapshot()['legal'][0]['action'])

    def test_reply_and_undo_restore_complete_human_turn(self):
        initial = self.game.board.get_state().tobytes()
        self.assertTrue(self.human_move()['ai_turn'])
        reply = self.update('ai')
        self.assertFalse(reply['ai_turn'])
        self.assertEqual(reply['ply'], 2)
        self.assertEqual(reply['player'], 0)
        self.update('undo')
        self.assertEqual(self.game.board.get_state().tobytes(), initial)

    def test_red_opening_and_undo_keep_ai_opening(self):
        state = self.update('restart', human_player=1)
        self.assertTrue(state['ai_turn'])
        self.assertEqual(self.opponent.reloads, 1)
        opening = self.update('ai')
        self.assertFalse(opening['can_undo'])
        before = self.game.board.get_state().tobytes()
        with self.assertRaises(ValueError):
            self.update('undo')
        self.human_move()
        self.update('ai')
        self.update('undo')
        self.assertEqual(self.game.board.get_state().tobytes(), before)

    def test_turn_ownership_and_stale_ai_requests_are_atomic(self):
        with self.assertRaises(ValueError):
            self.update('ai')
        self.human_move()
        before = self.game.snapshot()
        with self.assertRaises(ValueError):
            self.human_move()
        with self.assertRaises(ValueError):
            self.game.update('ai', {'revision': 0})
        self.assertEqual(self.game.snapshot(), before)
        self.update('ai')
        with self.assertRaises(ValueError):
            self.update('ai')

    def test_inference_and_reload_failure_preserve_game(self):
        self.human_move()
        before = self.game.snapshot()
        self.opponent.fail = True
        for command in ('ai', 'restart'):
            with self.assertRaises(RuntimeError):
                self.update(command)
            self.assertEqual(self.game.snapshot(), before)
        self.opponent.fail = False
        self.update('ai')

    def test_human_can_undo_before_ai_reply(self):
        self.human_move()
        state = self.update('undo')
        self.assertEqual(state['ply'], 0)
        self.assertFalse(state['ai_turn'])

    def test_terminal_human_move_does_not_request_ai(self):
        from intransitive.IntransitiveConstants import NE, encode_action
        from intransitive.tests.test_draws import load_history, sparse_position
        pieces = sparse_position()
        pieces[7, 7] = 1
        pieces[1, 1] = -1
        load_history(self.game.board, [pieces])
        state = self.update('move', action=encode_action(7, 7, NE))
        self.assertEqual(state['reason'], 'corner')
        self.assertFalse(state['ai_turn'])
        with self.assertRaises(ValueError):
            self.update('ai')


class SelectableOpponentTests(unittest.TestCase):
    @patch('intransitive.play.ModelOpponent.reload', autospec=True)
    def test_model_simulations_round_trip_and_watch_bots(self, reload):
        from intransitive.play import OpponentFactory
        reload.side_effect=lambda bot: setattr(bot, 'label', f'Test model · {bot.simulations} simulations')
        factory=OpponentFactory(checkpoint='unused.pt',simulations=128)
        session=GameSession(opponent_factory=factory)
        self.assertEqual(session.snapshot()['model_simulations'],128)
        for count in (32,256,512):
            state=session.update('restart',dict(revision=session.revision,opponent='model',model_simulations=count))
            self.assertEqual(session.opponent.simulations,count)
            self.assertEqual(state['model_simulations'],count)
            self.assertIn(str(count),state['model'])
        state=session.update('restart',dict(revision=session.revision,mode='watch',
            blue_opponent='model',red_opponent='model',model_simulations=256))
        self.assertEqual([bot.simulations for bot in session.bots],[256,256])
        self.assertIsNot(session.bots[0],session.bots[1])
        self.assertEqual(state['model_simulations'],256)
        self.assertEqual(factory.simulations,128)
        session.update('restart',dict(revision=session.revision,mode='play',opponent='local'))
        state=session.update('restart',dict(revision=session.revision,opponent='model'))
        self.assertEqual(state['model_simulations'],256)
        self.assertEqual(session.opponent.simulations,256)

    @patch('intransitive.play.ModelOpponent.reload', autospec=True)
    def test_invalid_model_budget_and_reload_failure_are_atomic(self, reload):
        from intransitive.play import OpponentFactory, ModelOpponent
        factory=OpponentFactory(checkpoint='unused.pt')
        session=GameSession(opponent_factory=factory)
        session.update('move',dict(revision=0,action=session.snapshot()['legal'][0]['action']))
        before=session.snapshot()
        for count in (None,True,False,0,1,-1,2.5,'512',float('inf')):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    session.update('restart',dict(revision=session.revision,opponent='model',model_simulations=count))
                with self.assertRaises(ValueError):
                    ModelOpponent('unused.pt',count)
                self.assertEqual(session.snapshot(),before)
        reload.assert_not_called()
        reload.side_effect=RuntimeError('Unavailable checkpoint')
        with self.assertRaises(RuntimeError):
            session.update('restart',dict(revision=session.revision,opponent='model',model_simulations=512))
        self.assertEqual(session.snapshot(),before)

    def test_switch_all_opponents_and_independent_options(self):
        from itertools import product
        from intransitive.play import OpponentFactory
        from intransitive.heuristics import SearchConfig
        factory = OpponentFactory(config=SearchConfig(max_depth=1, time_limit=5, node_limit=500000))
        session = GameSession(opponent_factory=factory)
        for flags in product((False, True), repeat=3):
            options = dict(zip(('attack_enabled', 'defence_enabled', 'overload_enabled'), flags))
            snapshot = session.update('restart', dict(revision=session.revision, opponent='alphabeta',
                                                      human_player=1, ab_options=options))
            self.assertEqual({key: snapshot['ab_options'][key] for key in options}, options)
            self.assertEqual(snapshot['ab_options']['max_depth'], factory.config.max_depth)
            self.assertTrue(snapshot['ai_turn'])
            result = session.update('ai', dict(revision=session.revision))
            self.assertEqual(result['ply'], 1)
            self.assertIsNotNone(result['analysis'])
        for opponent in ('random', 'greedy', 'local'):
            snapshot = session.update('restart', dict(revision=session.revision, opponent=opponent))
            self.assertEqual(snapshot['opponent'], opponent)
            self.assertEqual(snapshot['mode'], 'local' if opponent == 'local' else 'ai')

    def test_invalid_selection_or_options_preserve_game(self):
        from intransitive.play import OpponentFactory
        session = GameSession(opponent_factory=OpponentFactory())
        before = session.snapshot()
        for options in ({'opponent': 'model'}, {'opponent': 'unknown'},
                        {'opponent': 'alphabeta', 'ab_options': {'attack_enabled': 1}},
                        {'opponent': 'alphabeta', 'ab_options': {'unknown': 999}},
                        {'opponent': 'alphabeta', 'ab_options': ['attack']}):
            with self.assertRaises(ValueError):
                session.update('restart', dict(revision=session.revision, **options))
            self.assertEqual(session.snapshot(), before)

    def test_search_limits_round_trip_and_reach_the_player(self):
        from intransitive.play import OpponentFactory
        from intransitive.heuristics import SearchConfig
        from intransitive.heuristics.config import TIME_FIRST_LIMITS
        factory = OpponentFactory(config=SearchConfig(max_depth=6, time_limit=2.5, node_limit=700000))
        session = GameSession(opponent_factory=factory)
        # A page opened in local mode must still show the configured defaults.
        self.assertEqual(session.snapshot()['ab_options']['max_depth'], 6)
        self.assertEqual(session.snapshot()['ab_options']['time_limit'], 2.5)
        self.assertEqual(session.snapshot()['ab_presets']['time_first'], TIME_FIRST_LIMITS)
        state = session.update('restart', dict(revision=0, opponent='alphabeta', human_player=0,
                                               ab_options=TIME_FIRST_LIMITS))
        for key, value in TIME_FIRST_LIMITS.items():
            self.assertEqual(state['ab_options'][key], value)
            self.assertEqual(getattr(session.opponent.config, key), value)
        # Applying another New game can still select an explicit lower cap.
        options = dict(max_depth=1, time_limit=5., node_limit=500000, attack_enabled=True)
        state = session.update('restart', dict(revision=state['revision'], opponent='alphabeta',
                                               human_player=1, ab_options=options))
        for key, value in options.items():
            self.assertEqual(state['ab_options'][key], value)
            self.assertEqual(getattr(session.opponent.config, key), value)
        reply = session.update('ai', dict(revision=state['revision']))
        self.assertEqual(reply['ply'], 1)
        for key, value in options.items():
            self.assertEqual(reply['analysis']['config'][key], value)
        self.assertLessEqual(session.opponent.last_result.completed_depth, 1)
        self.assertEqual(reply['analysis']['completed_depth'],
                         session.opponent.last_result.completed_depth)
        self.assertEqual(reply['analysis']['selected_depth'],
                         session.opponent.last_result.selected_depth)
        self.assertEqual(reply['analysis']['partial_depth'],
                         session.opponent.last_result.partial_depth)
        self.assertEqual(reply['analysis']['stop_reason'],
                         session.opponent.last_result.stop_reason)
        # Per-game settings do not mutate the factory's CLI defaults.
        self.assertEqual(factory.config.max_depth, 6)

    def test_invalid_search_limits_leave_existing_game_unchanged(self):
        from intransitive.play import OpponentFactory
        session = GameSession(opponent_factory=OpponentFactory())
        session.update('move', dict(revision=0, action=session.snapshot()['legal'][0]['action']))
        before = session.snapshot()
        for key, values in {
            'max_depth': [-1, 65, 2.5, True, '4', None],
            'time_limit': [-1, float('inf'), float('nan'), True, '5', None],
            'node_limit': [-1, 2.5, True, '1000', None],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        session.update('restart', dict(revision=session.revision,
                            opponent='alphabeta', ab_options={key: value}))
                    self.assertEqual(session.snapshot(), before)
        for options in ([], False, ''):
            with self.assertRaises(ValueError):
                session.update('restart', dict(revision=session.revision,
                    opponent='alphabeta', ab_options=options))
            self.assertEqual(session.snapshot(), before)


class PlayServerTests(unittest.TestCase):
    def setUp(self):
        self.game = GameSession()
        self.initial = self.game.snapshot()  # Warm compilation outside request deadlines.
        self.server = PlayServer(('127.0.0.1', 0), self.game)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)

    def test_idle_browser_connection_does_not_block_state_or_page(self):
        with socket.create_connection(self.server.server_address, timeout=2):
            with urlopen(self.url + '/api/state', timeout=2) as response:
                state = json.load(response)
            self.assertEqual(state['pgn'], self.initial['pgn'])
            with urlopen(self.url, timeout=2) as response:
                page = response.read()
                self.assertIn(b'id="copy-position"', page)
                self.assertIn(b'id="time-first"', page)

    def test_concurrent_moves_with_same_revision_apply_only_once(self):
        barrier = Barrier(2)
        body = json.dumps(dict(revision=0, action=self.initial['legal'][0]['action'])).encode()

        def move():
            request = Request(self.url + '/api/move', data=body,
                              headers={'Content-Type': 'application/json'})
            barrier.wait(timeout=3)
            try:
                with urlopen(request, timeout=3) as response:
                    return response.status
            except HTTPError as exc:
                exc.close()
                return exc.code

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: move(), range(2)))
        self.assertEqual(sorted(results), [200, 400])
        self.assertEqual(self.game.revision, 1)
        self.assertEqual(len(self.game.moves), 1)

    def test_history_and_playback_asset_are_available(self):
        self.game.update('move',dict(revision=0,action=self.initial['legal'][0]['action']))
        with urlopen(self.url + '/api/history',timeout=3) as response:
            states=json.load(response)['states']
        self.assertEqual([s['ply'] for s in states],[0,1])
        self.assertEqual(states[0]['board'],self.initial['board'])
        self.assertEqual(states[1],self.game.snapshot())
        with urlopen(self.url + '/playback.js',timeout=3) as response:
            self.assertIn(b'class MovePlayback',response.read())


class WatchBotTests(unittest.TestCase):
    def setUp(self):
        from intransitive.heuristics import SearchConfig
        class Factory:
            config=SearchConfig()
            choices=['local','random','greedy']
            fail=False
            def create(self,kind,options=None):
                if self.fail and kind=='greedy':
                    raise RuntimeError('Second bot unavailable')
                if kind=='local':
                    return None
                bot=FirstLegalOpponent()
                bot.kind=kind
                bot.label=kind.title()
                return bot
        self.factory=Factory()
        self.game=GameSession(opponent_factory=self.factory)

    def start(self,**options):
        return self.game.update('restart',dict(revision=self.game.revision,mode='watch',
            blue_opponent='random',red_opponent='greedy',**options))

    def test_both_bots_play_and_history_is_read_only_replayable_and_correctly_named(self):
        from intransitive.record import load_record
        states=[self.start()]
        self.assertEqual(states[0]['mode'],'watch')
        self.assertEqual(states[0]['watch_opponents'],['random','greedy'])
        self.assertIsNot(self.game.bots[0],self.game.bots[1])
        self.assertTrue(states[0]['ai_turn'])
        for _ in range(8):
            states.append(self.game.update('ai',dict(revision=self.game.revision)))
        before=self.game.board.get_state().tobytes()
        revision=self.game.revision
        views=self.game.snapshots()
        self.assertEqual(len(views),9)
        for original,view in zip(states,views):
            self.assertEqual(original['board'],view['board'])
            self.assertEqual(original['moves'],view['moves'])
            self.assertEqual(original['ply'],view['ply'])
            record=load_record(view['pgn'])
            self.assertEqual(record.tags['Blue'],'Random')
            self.assertEqual(record.tags['Red'],'Greedy')
            self.assertEqual(record.states[-1][:,:,0].tolist(),view['board'])
        self.assertEqual(self.game.board.get_state().tobytes(),before)
        self.assertEqual(self.game.revision,revision)
        for cmd in ('undo','move'):
            with self.assertRaises(ValueError):
                self.game.update(cmd,dict(revision=revision,action=states[-1]['legal'][0]['action']))
        self.assertFalse(self.game.snapshot()['can_undo'])

    def test_failed_restart_is_atomic_and_mode_can_switch_back_to_play(self):
        self.start()
        before=self.game.snapshot()
        for options in ({'blue_opponent':'local','red_opponent':'greedy'},
                        {'blue_opponent':'random','red_opponent':'unknown'}):
            with self.assertRaises(ValueError):
                self.game.update('restart',dict(revision=self.game.revision,mode='watch',**options))
            self.assertEqual(self.game.snapshot(),before)
        self.factory.fail=True
        with self.assertRaises(RuntimeError):
            self.start()
        self.assertEqual(self.game.snapshot(),before)
        state=self.game.update('restart',dict(revision=self.game.revision,mode='play',opponent='local'))
        self.assertEqual(state['mode'],'local')
        self.assertIsNone(state['watch_opponents'])
        self.assertNotEqual(state['game_id'],before['game_id'])


if __name__ == "__main__":
    unittest.main()
