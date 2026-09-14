"""Local browser play using the compiled rules engine: python -m intransitive.play."""

import argparse
from dataclasses import replace
import json
import logging
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from .IntransitiveConstants import action_destination, decode_action, format_coordinate
from .IntransitiveLogicNumba import Board


class BaselineOpponent:
    def __init__(self, kind):
        from .IntransitiveGame import IntransitiveGame
        from .IntransitivePlayers import GreedyPlayer, RandomPlayer
        self.kind = kind
        self.label = kind.title()
        self.game = IntransitiveGame()
        self.player = (GreedyPlayer if kind == 'greedy' else RandomPlayer)(self.game)

    def reload(self):
        pass

    def choose(self, state, player):
        return self.player.play(self.game.getCanonicalForm(state, player))


class OpponentFactory:
    def __init__(self, checkpoint=None, simulations=32, config=None):
        from .heuristics import SearchConfig
        self.checkpoint, self.simulations = checkpoint, simulations
        self.config = config or SearchConfig()

    @property
    def choices(self):
        return ['local', 'alphabeta', 'greedy', 'random'] + (['model'] if self.checkpoint else [])

    def create(self, kind, options=None):
        from .heuristics import AlphaBetaPlayer
        if kind not in self.choices:
            raise ValueError('Choose an available opponent')
        if kind == 'local':
            return None
        if kind == 'model':
            return ModelOpponent(self.checkpoint, self.simulations)
        if kind == 'alphabeta':
            options = options or {}
            if not isinstance(options, dict) or set(options) - {'attack_enabled', 'defence_enabled', 'overload_enabled'}:
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
        self.board = Board()
        self.history = []
        self.moves = []
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
        return dict(
            board=self.board.get_board().tolist(), player=player, legal=legal,
            reason=self.board.get_terminal_reason(),
            winner=next((p for p in range(2) if result[p] == 1), None),
            noncapture=self.board.get_no_capture_count(),
            repetition=self.board.get_repetition_count(),
            ply=self.board.get_total_ply(), moves=self.moves.copy(),
            revision=self.revision,
            mode='ai' if self.opponent else 'local', human_player=self.human_player,
            ai_turn=self.ai_turn(), can_undo=self.can_undo(),
            model=self.opponent.label if self.opponent else None,
            opponent=getattr(self.opponent, 'kind', 'model') if self.opponent else 'local',
            opponents=self.opponent_factory.choices if self.opponent_factory else [],
            ab_options={name: getattr(self.opponent.config, name) for name in
                        ('attack_enabled', 'defence_enabled', 'overload_enabled')}
                       if getattr(self.opponent, 'kind', None) == 'alphabeta' else {},
            analysis=self.opponent.last_result.explanation
                     if getattr(self.opponent, 'last_result', None) else None,
        )

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
            action = self.opponent.choose(self.board.get_state(), self.board.get_next_player())
            self.make_move(action)
        elif command == "undo":
            if not self.can_undo():
                raise ValueError("There are no moves to undo.")
            while self.history:
                self.board.copy_state(self.history.pop(), True)
                self.moves.pop()
                if not self.opponent or self.board.get_next_player() == self.human_player:
                    break
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
        else:
            raise ValueError("Unknown command.")
        self.revision += 1
        return self.snapshot()


class PlayHandler(BaseHTTPRequestHandler):
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
            self.respond(200, self.server.game.snapshot())
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
    parser.add_argument('--opponent', choices=('local', 'alphabeta', 'greedy', 'random', 'model'))
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
    factory = OpponentFactory(args.checkpoint, args.simulations, config)
    opponent = factory.create(args.opponent or ('model' if args.checkpoint else 'local'))
    game = GameSession(opponent, human_player=0 if args.human_colour == 'blue' else 1,
                       opponent_factory=factory)
    with HTTPServer(("127.0.0.1", args.port), PlayHandler) as server:
        server.game = game
        print(f"Play Intransitive at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nGame server stopped.")


if __name__ == "__main__":
    main()
