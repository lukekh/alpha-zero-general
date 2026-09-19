"""Local browser play using the compiled rules engine: python -m intransitive.play."""

import argparse
from dataclasses import asdict, replace
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

import numpy as np

from .IntransitiveConstants import action_destination, decode_action, format_coordinate
from .IntransitiveLogicNumba import Board, search_observation


AB_OPTION_FIELDS = ('nmp_enabled', 'futility_enabled', 'nmp_min_depth', 'nmp_reduction',
                    'futility_max_depth', 'futility_margin', 'attack_enabled', 'defence_enabled', 'overload_enabled',
                    'max_depth', 'time_limit', 'node_limit')

# Analysis runs on demand from the browser and must not become a way to hang
# the shared game lock. These ceilings apply on top of the requested limits.
ANALYSIS_MAX_SECONDS = 30.
ANALYSIS_MAX_WORK = 50_000_000
ANALYSIS_MAX_DEPTH = 8


class BaselineOpponent:
    def __init__(self, kind):
        from .IntransitiveGame import IntransitiveGame
        from .IntransitivePlayers import GreedyPlayer, RandomPlayer, ReferenceGreedyPlayer
        self.kind = kind
        self.label = kind.replace('-', ' ').title()
        self.game = IntransitiveGame()
        self.player = {'greedy': GreedyPlayer, 'random': RandomPlayer,
                       'reference-greedy': ReferenceGreedyPlayer}[kind](self.game)

    def reload(self):
        pass

    def choose(self, state, player):
        return self.player.play(self.game.getCanonicalForm(state, player))


class OpponentFactory:
    def __init__(self, checkpoint=None, simulations=32, config=None, flybrain_bank=None):
        from .heuristics import SearchConfig
        self.checkpoint, self.simulations = checkpoint, simulations
        self.config = config or SearchConfig()
        self.flybrain_bank = flybrain_bank

    @property
    def choices(self):
        from .flybrain import DEFAULT_BANK
        flybrain = ['flybrain'] if Path(self.flybrain_bank or DEFAULT_BANK).is_file() else []
        return ['local', 'alphabeta', 'greedy', 'reference-greedy', 'random'] + flybrain + (['model'] if self.checkpoint else [])

    def create(self, kind, options=None):
        from .heuristics import AlphaBetaPlayer
        if kind not in self.choices:
            raise ValueError('Choose an available opponent')
        if kind == 'local':
            return None
        if kind == 'model':
            return ModelOpponent(self.checkpoint, self.simulations)
        if kind == 'flybrain':
            from .flybrain import FlybrainPlayer
            from .IntransitiveGame import IntransitiveGame
            return FlybrainPlayer(IntransitiveGame(), self.flybrain_bank)
        if kind == 'alphabeta':
            options = {} if options is None else options
            if not isinstance(options, dict) or set(options) - set(AB_OPTION_FIELDS):
                raise ValueError('Invalid alpha-beta options')
            return AlphaBetaPlayer(config=replace(self.config, **options))
        return BaselineOpponent(kind)


class ModelOpponent:
    """A fixed model per game; reload the checkpoint when starting a new game."""

    kind = 'model'

    def __init__(self, checkpoint, simulations=32):
        if simulations < 2:
            raise ValueError("Use at least two search simulations.")
        self.checkpoint = Path(checkpoint).resolve()
        self.simulations = simulations
        self.reload()

    def reload(self):
        os.environ['ORT_DISABLE_TELEMETRY'] = '1'
        import onnxruntime as ort
        ort.disable_telemetry_events()
        from MCTS import MCTS
        from .IntransitiveGame import IntransitiveGame
        from .NNet import NNetWrapper

        game = IntransitiveGame()
        net = NNetWrapper(game, dict(nn_version=-1, dropout=0.))
        metadata = net.load_checkpoint(str(self.checkpoint.parent), self.checkpoint.name)
        args = argparse.Namespace(numMCTSSims=self.simulations, prob_fullMCTS=1.,
            ratio_fullMCTS=1, universes=0, cpuct=metadata.get('cpuct', 1.25),
            fpu=metadata.get('fpu', 0.), forced_playouts=False, no_mem_optim=False)
        # Prepare inference before replacing the previous working opponent.
        board = game.getInitBoard()
        net.predict(board, game.getValidMoves(board, 0))
        self.game, self.search = game, MCTS(game, net, args)
        iteration = metadata.get('run_iteration', metadata.get('candidate_iteration'))
        self.label = f"Saved model · iteration {iteration}" if iteration is not None else "Saved model"

    def choose(self, board, player):
        canonical = self.game.getCanonicalForm(board, player)
        # Start fresh after undo as well as after a normal turn.
        self.search.nodes_data.clear()
        policy, _, _ = self.search.getActionProb(canonical, temp=0, force_full_search=True)
        return int(np.argmax(policy))


class GameSession:
    def __init__(self, opponent=None, human_player=0, opponent_factory=None):
        self.opponent_factory = opponent_factory
        self.board = Board(modelling_draws=False)
        self.history = []
        self.moves = []
        self.ai_decisions = {}
        self.revision = 0
        self.opponent = opponent
        self.human_player = human_player

    def ai_turn(self):
        return (self.opponent is not None
                and self.board.get_next_player() != self.human_player
                and self.board.get_terminal_reason() == 'ongoing')

    def can_undo(self):
        # Red cannot undo the AI's opening before making a move of their own.
        return len(self.history) > (1 if self.opponent and self.human_player == 1 else 0)

    def snapshot(self):
        player = self.board.get_next_player()
        legal = []
        for action in np.flatnonzero(self.board.valid_moves(player)):
            x, y, _ = decode_action(int(action))
            dx, dy = action_destination(int(action))
            legal.append(dict(action=int(action), source=[x, y], target=[dx, dy]))
        result = self.board.check_end_game(player)
        config = (self.opponent.config if getattr(self.opponent, 'kind', None) == 'alphabeta'
                  else self.opponent_factory.config if self.opponent_factory else None)
        from .heuristics.config import SearchConfig, TIME_FIRST_LIMITS
        from .record import export_record
        last_ai = self.ai_decisions[max(self.ai_decisions)] if self.ai_decisions else None
        analysis_result = getattr(self.opponent, 'last_result', None)
        analysis = None
        if analysis_result is not None:
            analysis = dict(analysis_result.explanation)
            for name in ('completed_depth', 'selected_depth', 'partial_depth',
                         'root_moves_completed', 'root_moves_total', 'selection_source',
                         'score_bound', 'stop_reason', 'diagnostics_status',
                         'effective_limits', 'selective'):
                analysis[name] = getattr(analysis_result, name)
        noncapture = 0
        for move in reversed(self.moves):
            if ' × ' in move:
                break
            noncapture += 1
        current = self.board.get_board()
        repetition = 1 + sum(
            int(s[:, :, 82:84].flat[1]) == player and np.array_equal(s[:, :, 0], current)
            for s in (self.history[-noncapture:] if noncapture else []))
        return dict(
            board=self.board.get_board().tolist(), player=player, legal=legal,
            reason=self.board.get_terminal_reason(),
            winner=next((p for p in range(2) if result[p] == 1), None),
            noncapture=noncapture, repetition=repetition,
            ply=self.board.get_total_ply(), moves=self.moves.copy(),
            revision=self.revision,
            mode='ai' if self.opponent else 'local', human_player=self.human_player,
            ai_turn=self.ai_turn(), can_undo=self.can_undo(),
            model=self.opponent.label if self.opponent else None,
            opponent=getattr(self.opponent, 'kind', 'model') if self.opponent else 'local',
            opponents=self.opponent_factory.choices if self.opponent_factory else [],
            ab_options={name: getattr(config, name) for name in AB_OPTION_FIELDS} if config else {},
            ab_presets={'time_first': TIME_FIRST_LIMITS.copy()},
            pgn=export_record(self.moves, self.board, config or SearchConfig(),
                              opponent=getattr(self.opponent, 'kind', 'model') if self.opponent else 'local',
                              human_player=self.human_player, last_ai=last_ai),
            analysis=analysis,
        )

    def analyse(self, data):
        """Per-square heuristic attribution for the position now on the board.

        Read-only: no move is made and the revision is unchanged. A fresh
        budget keeps diagnostics from competing with the opponent's own search,
        and the caller's limits are clamped so a browser cannot stall the game.
        """
        from .heuristics.attribution import DEFAULT_MODULES, attribute
        from .heuristics.budget import Budget, BudgetExpired
        from .heuristics.evaluation import MODULES, Evaluator
        from .heuristics.search import AlphaBetaPlayer
        from .IntransitiveDisplay import move_to_str
        from .IntransitiveGame import IntransitiveGame
        if type(data.get('revision')) is not int or data['revision'] != self.revision:
            raise ValueError('The board changed. Refresh and try again.')
        modules = data.get('modules', list(DEFAULT_MODULES))
        if (type(modules) is not list or not all(type(name) is str for name in modules)
                or set(modules) - set(MODULES)):
            raise ValueError('Choose analysis modules from the evaluator module list.')
        state = search_observation(self.board.get_state())
        side = self.board.get_next_player()
        perspective = data.get('perspective', side)
        if type(perspective) is not int or perspective not in (0, 1):
            raise ValueError('Choose Blue or Red as the analysis perspective.')
        config = replace(self.analysis_config(), **self.analysis_limits(data))
        game = IntransitiveGame()
        budget = Budget(config.node_limit, config.time_limit)
        try:
            explanation = Evaluator(game, config).explain(state, perspective, budget, diagnostics=False)
            report = attribute(game, state, perspective, budget, config,
                               modules=modules, explanation=explanation)
            report['status'] = 'completed'
        except BudgetExpired as expired:
            return dict(status='budget exhausted', reason=expired.reason, revision=self.revision,
                        config=config.to_dict(), requested_modules=modules)
        report.update(revision=self.revision, side_to_move=side, config=config.to_dict(),
                      requested_modules=modules,
                      work=budget.work, elapsed=budget.clock() - budget.start)
        report['search'] = None
        if data.get('search') and self.board.get_terminal_reason() == 'ongoing':
            exact_root = bool(data.get('exact_root'))
            result = AlphaBetaPlayer(game, config).analyze(state, exact_root=exact_root)
            report['search'] = dict(
                action=result.action, move=move_to_str(result.action), score=result.score,
                completed_depth=result.completed_depth, selected_depth=result.selected_depth,
                stop_reason=result.stop_reason, score_bound=result.score_bound,
                selection_source=result.selection_source, exact_root=result.exact_root,
                nodes=result.nodes, work=result.work, elapsed=result.elapsed,
                root_moves=result.root_moves, side=side,
                line=[move_to_str(step) for step in result.pv])
        return report

    def analysis_config(self):
        from .heuristics.config import SearchConfig
        if getattr(self.opponent, 'kind', None) == 'alphabeta':
            return self.opponent.config
        return self.opponent_factory.config if self.opponent_factory else SearchConfig()

    @staticmethod
    def analysis_limits(data):
        """Validate and clamp the browser's requested analysis limits."""
        limits = {}
        for key, field, ceiling, kind in (('depth', 'max_depth', ANALYSIS_MAX_DEPTH, int),
                                          ('time', 'time_limit', ANALYSIS_MAX_SECONDS, float),
                                          ('work', 'node_limit', ANALYSIS_MAX_WORK, int)):
            value = data.get(key)
            if value is None:
                continue
            if kind is int and type(value) is not int:
                raise ValueError(f'Analysis {key} must be a whole number.')
            if kind is float and (type(value) not in (int, float) or value != value):
                raise ValueError(f'Analysis {key} must be a number.')
            if value < 0:
                raise ValueError(f'Analysis {key} must not be negative.')
            limits[field] = min(kind(value), ceiling)
        return limits

    def make_move(self, action):
        before = self.board.get_state()
        self.board.make_move(action, self.board.get_next_player())
        x, y, _ = decode_action(action)
        dx, dy = action_destination(action)
        capture = before[dy, dx, 0] != 0
        self.history.append(before)
        self.moves.append(format_coordinate(x, y) + (" × " if capture else " → ")
                          + format_coordinate(dx, dy))

    def update(self, command, data):
        if type(data.get("revision")) is not int or data["revision"] != self.revision:
            raise ValueError("The board changed. Refresh and try again.")
        if command == "move":
            if self.ai_turn():
                raise ValueError("It is the AI's turn.")
            action = data.get("action")
            if type(action) is not int:
                raise ValueError("Choose a legal move.")
            self.make_move(action)
        elif command == "ai":
            if not self.ai_turn():
                raise ValueError("It is not the AI's turn.")
            from .record import state_hash
            before = self.board.get_state()
            ply = len(self.moves)
            action = int(self.opponent.choose(search_observation(before), self.board.get_next_player()))
            self.make_move(action)
            result = getattr(self.opponent, 'last_result', None)
            search = asdict(result) if result else None
            if search:
                # Keep the original scores/features and search statistics in the
                # copied message; verbose route traces can be recomputed later.
                for name in ('races', 'config', 'pv', 'search_score'):
                    search['explanation'].pop(name, None)
                # Per-reply rows are an analysis view, not part of the decision
                # record, and they would dominate the copied position.
                search.pop('root_moves', None)
            self.ai_decisions[ply] = dict(ply=ply, action=action, state_sha256=state_hash(before),
                                          search=search)
        elif command == "undo":
            if not self.can_undo():
                raise ValueError("There are no moves to undo.")
            while self.history:
                self.board.copy_state(self.history.pop(), True)
                self.moves.pop()
                if not self.opponent or self.board.get_next_player() == self.human_player:
                    break
            self.ai_decisions = {ply: record for ply, record in self.ai_decisions.items()
                                 if ply < len(self.moves)}
        elif command == "restart":
            human = data.get('human_player', self.human_player)
            if type(human) is not int or human not in (0, 1):
                raise ValueError("Choose Blue or Red.")
            opponent = self.opponent
            if 'opponent' in data:
                if self.opponent_factory is None:
                    raise ValueError('Opponent selection is unavailable')
                opponent = self.opponent_factory.create(data['opponent'], data.get('ab_options'))
            elif opponent:
                opponent.reload()
            self.opponent = opponent
            self.human_player = human
            self.board.init_game()
            self.history.clear()
            self.moves.clear()
            self.ai_decisions.clear()
        else:
            raise ValueError("Unknown command.")
        self.revision += 1
        return self.snapshot()


class PlayServer(ThreadingHTTPServer):
    """Idle browser preconnections must not block other requests.

    The shared game and opponent remain serialized, including revision checks.
    """

    def __init__(self, address, game):
        self.game = game
        self.game_lock = Lock()
        super().__init__(address, PlayHandler)


class PlayHandler(BaseHTTPRequestHandler):
    timeout = 10

    def respond(self, status, body, content_type="application/json"):
        payload = json.dumps(body).encode() if content_type == "application/json" else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/":
            self.respond(200, Path(__file__).with_name("play.html").read_bytes(),
                         "text/html; charset=utf-8")
        elif self.path == "/api/state":
            with self.server.game_lock:
                state = self.server.game.snapshot()
            self.respond(200, state)
        else:
            self.respond(404, {"error": "Not found"})

    def do_POST(self):
        origin = self.headers.get("Origin")
        allowed = {f"http://127.0.0.1:{self.server.server_port}",
                   f"http://localhost:{self.server.server_port}"}
        if origin and origin not in allowed:
            self.respond(403, {"error": "Use the local game page."})
            return
        if self.headers.get("Content-Type") != "application/json":
            self.respond(415, {"error": "Expected JSON."})
            return
        if self.path not in ("/api/move", "/api/ai", "/api/undo", "/api/restart", "/api/analyze"):
            self.respond(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size.")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            with self.server.game_lock:
                state = (self.server.game.analyse(data) if self.path == "/api/analyze"
                         else self.server.game.update(self.path.removeprefix("/api/"), data))
        except (ValueError, UnicodeDecodeError) as exc:
            self.respond(400, {"error": str(exc)})
            return
        except Exception:
            logging.exception("Game request failed")
            self.respond(500, {"error": "The AI could not complete this request. Try again."})
            return
        self.respond(200, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, help="Model to play against; reloaded for each new game")
    parser.add_argument("--human-colour", choices=('blue', 'red'), default='blue')
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument('--opponent', choices=('local', 'alphabeta', 'greedy', 'reference-greedy', 'random', 'model', 'flybrain'))
    parser.add_argument('--flybrain-bank', type=Path)
    parser.add_argument('--ab-config', type=Path)
    args = parser.parse_args()
    if args.simulations < 2:
        parser.error('--simulations must be at least 2')
    print("Preparing Intransitive rules…", flush=True)
    warmup = GameSession()
    # Compile queries and transitions before accepting the first request.
    first = warmup.snapshot()["legal"][0]["action"]
    warmup.update("move", {"action": first, "revision": 0})
    warmup.update("undo", {"revision": 1})
    from .heuristics import SearchConfig
    config = SearchConfig.from_file(args.ab_config) if args.ab_config else SearchConfig()
    factory = OpponentFactory(args.checkpoint, args.simulations, config, args.flybrain_bank)
    opponent = factory.create(args.opponent or ('model' if args.checkpoint else 'local'))
    game = GameSession(opponent, human_player=0 if args.human_colour == 'blue' else 1,
                       opponent_factory=factory)
    with PlayServer(("127.0.0.1", args.port), game) as server:
        print(f"Play Intransitive at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nGame server stopped.")


if __name__ == "__main__":
    main()
