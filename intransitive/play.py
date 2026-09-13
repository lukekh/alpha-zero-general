"""Local browser play using the compiled rules engine: python -m intransitive.play."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

from .IntransitiveConstants import action_destination, decode_action, format_coordinate
from .IntransitiveLogicNumba import Board


class GameSession:
    def __init__(self):
        self.board = Board()
        self.history = []
        self.moves = []
        self.revision = 0

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
        )

    def update(self, command, data):
        if type(data.get("revision")) is not int or data["revision"] != self.revision:
            raise ValueError("The board changed. Refresh and try again.")
        if command == "move":
            action = data.get("action")
            if type(action) is not int:
                raise ValueError("Choose a legal move.")
            before = self.board.get_state()
            self.board.make_move(action, self.board.get_next_player())
            x, y, _ = decode_action(action)
            dx, dy = action_destination(action)
            capture = before[dy, dx, 0] != 0
            self.history.append(before)
            self.moves.append(format_coordinate(x, y) + (" × " if capture else " → ")
                              + format_coordinate(dx, dy))
        elif command == "undo":
            if not self.history:
                raise ValueError("There are no moves to undo.")
            self.board.copy_state(self.history.pop(), True)
            self.moves.pop()
        elif command == "restart":
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
        if self.path not in ("/api/move", "/api/undo", "/api/restart"):
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
        self.respond(200, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Preparing Intransitive rules…", flush=True)
    game = GameSession()
    # Compile queries and transitions before accepting the first request.
    first = game.snapshot()["legal"][0]["action"]
    game.update("move", {"action": first, "revision": 0})
    game.update("undo", {"revision": 1})
    with HTTPServer(("127.0.0.1", args.port), PlayHandler) as server:
        server.game = game
        print(f"Play Intransitive at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nGame server stopped.")


if __name__ == "__main__":
    main()
