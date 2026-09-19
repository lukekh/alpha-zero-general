"""Local browser play using the compiled rules engine: python -m intransitive.play."""

import argparse
from dataclasses import asdict, replace
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np

from .IntransitiveConstants import action_destination, decode_action, format_coordinate
from .IntransitiveLogicNumba import Board, search_observation


AB_OPTION_FIELDS = ('attack_enabled', 'defence_enabled', 'overload_enabled',
                    'max_depth', 'time_limit', 'node_limit')
MINIMAX_KINDS = ('alphabeta', 'rust-minimax')


def validate_simulations(value):
    if type(value) is not int or value < 2:
        raise ValueError('MCTS simulations must be a whole number of at least 2.')
    return value


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
        return ['local', 'alphabeta', 'rust-minimax', 'greedy', 'reference-greedy', 'random'] + flybrain + (['model'] if self.checkpoint else [])

    def create(self, kind, options=None, *, simulations=None):
        from .heuristics import AlphaBetaPlayer
        if kind not in self.choices:
            raise ValueError('Choose an available opponent')
        if kind == 'local':
            return None
        if kind == 'model':
            return ModelOpponent(self.checkpoint, self.simulations if simulations is None else simulations)
        if kind == 'flybrain':
            from .flybrain import FlybrainPlayer
            from .IntransitiveGame import IntransitiveGame
            return FlybrainPlayer(IntransitiveGame(), self.flybrain_bank)
        if kind in MINIMAX_KINDS:
            options = {} if options is None else options
            if not isinstance(options, dict) or set(options) - set(AB_OPTION_FIELDS):
                raise ValueError('Invalid alpha-beta options')
            config = replace(self.config, **options)
            if kind == 'rust-minimax':
                from .rust_teacher.browser import RustMinimaxOpponent
                return RustMinimaxOpponent(config)
            player = AlphaBetaPlayer(config=config)
            player.label = 'Minimax · Python'
            return player
        return BaselineOpponent(kind)


class ModelOpponent:
    """A fixed model per game; reload the checkpoint when starting a new game."""

    kind = 'model'

    def __init__(self, checkpoint, simulations=32):
        validate_simulations(simulations)
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
        self.label += f" · {self.simulations} MCTS simulations"

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
        self.bots = None
        self.model_simulations = getattr(opponent, 'simulations', getattr(opponent_factory, 'simulations', 32))
        self.game_id = uuid4().hex

    def active_opponent(self):
        return self.bots[self.board.get_next_player()] if self.bots else self.opponent

    def ai_turn(self):
        return (self.active_opponent() is not None
                and (self.bots is not None or self.board.get_next_player() != self.human_player)
                and self.board.get_terminal_reason() == 'ongoing')

    def can_undo(self):
        if self.bots:
            return False  # Spectator history navigation never alters the game.
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
        opponent = self.bots[(len(self.moves)-1) % 2] if self.bots and self.moves else self.active_opponent()
        configured = next((bot for bot in (self.bots or [self.opponent])
                           if getattr(bot, 'kind', None) in MINIMAX_KINDS), None)
        config = (configured.config if configured
                  else self.opponent_factory.config if self.opponent_factory else None)
        from .heuristics.config import SearchConfig, TIME_FIRST_LIMITS
        from .record import export_record
        last_ai = self.ai_decisions[max(self.ai_decisions)] if self.ai_decisions else None
        search_result = last_ai.get('search') if last_ai else None
        analysis = None
        if search_result is not None:
            rust = last_ai.get('opponent') == 'rust-minimax'
            analysis = dict(search_result['explanation'])
            analysis.update(backend='Rust' if rust else 'Python',
                            elapsed_seconds=search_result['elapsed'], nodes=search_result['nodes'],
                            work=search_result['work'], score=search_result['score'],
                            work_unit='node_visits' if rust else 'charged_work')
            for name in ('completed_depth', 'selected_depth', 'partial_depth',
                         'root_moves_completed', 'root_moves_total', 'selection_source',
                         'score_bound', 'stop_reason', 'diagnostics_status',
                         'effective_limits'):
                analysis[name] = search_result[name]
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
            revision=self.revision, game_id=self.game_id,
            mode='watch' if self.bots else 'ai' if self.opponent else 'local', human_player=self.human_player,
            ai_turn=self.ai_turn(), can_undo=self.can_undo(),
            model=self.opponent.label if self.opponent else None,
            opponent=getattr(self.opponent, 'kind', 'model') if self.opponent else 'local',
            opponents=self.opponent_factory.choices if self.opponent_factory else [],
            watch_opponents=[getattr(bot,'kind','model') for bot in self.bots] if self.bots else None,
            bot_labels=[bot.label for bot in self.bots] if self.bots else None,
            model_simulations=self.model_simulations,
            ab_options={name: getattr(config, name) for name in AB_OPTION_FIELDS} if config else {},
            ab_presets={'time_first': TIME_FIRST_LIMITS.copy()},
            pgn=export_record(self.moves, self.board, config or SearchConfig(),
                              opponent=getattr(self.opponent, 'kind', 'model') if self.opponent else 'local',
                              human_player=self.human_player, last_ai=last_ai,
                              players=[bot.label for bot in self.bots] if self.bots else None),
            analysis=analysis,
        )

    def snapshots(self):
        """Read-only views of this line, including positions before a page reload."""
        views = []
        for ply, state in enumerate(self.history):
            past = GameSession(self.opponent, self.human_player, self.opponent_factory)
            past.board.copy_state(state, True)
            past.bots, past.game_id, past.revision = self.bots, self.game_id, self.revision
            past.history, past.moves = self.history[:ply], self.moves[:ply]
            past.ai_decisions = {p:r for p,r in self.ai_decisions.items() if p < ply}
            view = past.snapshot()
            # Diagnostics come from this prefix's recorded AI decision.
            views.append(view)
        return views + [self.snapshot()]

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
            if self.bots:
                raise ValueError('Bot games are read-only. Use the history controls to review moves.')
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
            opponent = self.active_opponent()
            action = int(opponent.choose(search_observation(before), self.board.get_next_player()))
            self.make_move(action)
            result = getattr(opponent, 'last_result', None)
            search = asdict(result) if result else None
            if search:
                # Keep the original scores/features and search statistics in the
                # copied message; verbose route traces can be recomputed later.
                for name in ('races', 'pv', 'search_score'):
                    search['explanation'].pop(name, None)
            self.ai_decisions[ply] = dict(ply=ply, action=action, state_sha256=state_hash(before),
                                          search=search, opponent=getattr(opponent, 'kind', 'model'),
                                          label=opponent.label)
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
            mode = data.get('mode', 'watch' if self.bots else 'play')
            if mode not in ('play','watch'):
                raise ValueError('Choose Play or Watch bots')
            simulations = validate_simulations(data.get('model_simulations', self.model_simulations))

            def create(kind):
                if self.opponent_factory is None:
                    raise ValueError('Opponent selection is unavailable')
                kwargs = {'simulations': simulations} if kind == 'model' else {}
                return self.opponent_factory.create(kind, data.get('ab_options'), **kwargs)

            opponent, bots = self.opponent, None
            if mode == 'watch':
                if self.opponent_factory is None:
                    raise ValueError('Opponent selection is unavailable')
                kinds = [data.get('blue_opponent'), data.get('red_opponent')]
                if any(kind not in self.opponent_factory.choices or kind == 'local' for kind in kinds):
                    raise ValueError('Choose a bot for both Blue and Red')
                # Construct both before changing the live board or either player.
                bots = tuple(create(kind) for kind in kinds)
                opponent = None
            elif 'opponent' in data:
                if self.opponent_factory is None:
                    raise ValueError('Opponent selection is unavailable')
                opponent = create(data['opponent'])
            elif opponent:
                if getattr(opponent, 'kind', None) == 'model' and simulations != opponent.simulations:
                    opponent = create('model')
                else:
                    opponent.reload()
            self.opponent = opponent
            self.bots = bots
            self.model_simulations = simulations
            self.human_player = human
            self.board.init_game()
            self.history.clear()
            self.moves.clear()
            self.ai_decisions.clear()
            self.game_id = uuid4().hex
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
        elif self.path == '/playback.js':
            self.respond(200, Path(__file__).with_name('playback.js').read_bytes(),
                         'text/javascript; charset=utf-8')
        elif self.path == '/api/history':
            with self.server.game_lock:
                states = self.server.game.snapshots()
            self.respond(200, {'states': states})
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
        if self.path not in ("/api/move", "/api/ai", "/api/undo", "/api/restart"):
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
                state = self.server.game.update(self.path.removeprefix("/api/"), data)
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
    parser.add_argument('--opponent', choices=('local', 'alphabeta', 'rust-minimax', 'greedy', 'reference-greedy', 'random', 'model', 'flybrain'))
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
