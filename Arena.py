import logging
log = logging.getLogger(__name__)

import bisect
from tqdm import trange
import zlib
import base64
import numpy as np
from os import environ

from MCTS import MCTS

class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None, *, modelling_draws=True):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.
            modelling_draws: keep training cutoffs by default; False uses
                     official play rules for games supporting for_play().

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        # Coach keeps bounded modelling games. Live matches opt into official
        # rules without changing the game objects owned by their searches.
        self.game = game.for_play() if not modelling_draws and hasattr(game, 'for_play') else game
        self.modelling_draws = modelling_draws
        self.display = display
        self.macos_terminal = (environ.get("TERM_PROGRAM", "") == "Apple_Terminal" and "ITERM_SESSION_ID" not in environ)

    @staticmethod
    def serialize_state(board, curPlayer, turn):
        """Encode the complete int8 state in the existing pit.py --state format.

        The trailer is one player byte and a two-byte big-endian turn count.
        History and game metadata belong in board and are never stripped.
        """
        if board.dtype != np.int8:
            raise ValueError("Arena state requires an int8 board")
        data = (board.tobytes(order='C') + int(curPlayer).to_bytes(1, 'big')
                + int(turn).to_bytes(2, 'big'))
        return base64.b64encode(zlib.compress(data, level=9, wbits=-15)).decode('ascii')

    def restore_state(self, initial_state):
        """Restore an owned, writable board, absolute next player, and turn."""
        data = zlib.decompress(base64.b64decode(initial_state, validate=True), wbits=-15)
        shape = self.game.getBoardSize()
        if len(data) != int(np.prod(shape)) + 3:
            raise ValueError("Arena state has the wrong byte count")
        player, turn = data[-3], int.from_bytes(data[-2:], 'big')
        if player >= self.game.getNumberOfPlayers():
            raise ValueError("Arena state has an invalid player")
        board = np.frombuffer(data[:-3], dtype=np.int8).reshape(shape).copy()
        return board, player, turn

    def playGame(self, initial_state="", verbose=False, other_way=False):
        """
        Executes one episode of a game.

        Returns:
            Absolute player 0's reward. playGames maps this to agent wins,
            accounting for other_way; a nonzero reward other than +/-1 is a draw.
        """
        # if NUMBER_PLAYERS == 2:
        #     players = [self.player2, self.player1]                             if other_way else [self.player1, self.player2]
        # elif NUMBER_PLAYERS == 3:
        #     players = [self.player2, self.player1, self.player1]               if other_way else [self.player1, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 4:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        # elif NUMBER_PLAYERS == 5:
        #     players = [self.player2, self.player1, self.player1, self.player1] if other_way else [self.player1, self.player2, self.player2, self.player2]
        if not other_way:
            players = [self.player1]+[self.player2]*(self.game.getNumberOfPlayers()-1)
        else:
            players = [self.player2]+[self.player1]*(self.game.getNumberOfPlayers()-1)
        curPlayer, it = 0, 0
        board = self.game.getInitBoard()

        # Load initial state
        if initial_state != "":
            board, curPlayer, it = self.restore_state(initial_state)

        while not self.game.getGameEnded(board, curPlayer).any():
            it += 1
            if verbose:
                if self.display:
                    self.display(board)
                print()
                print(f'Turn {it} Player {curPlayer}: ', end='')        
                
            canonical_board = self.game.getCanonicalForm(board, curPlayer)
            observation = (self.game.getSearchObservation(canonical_board)
                           if not self.modelling_draws and hasattr(self.game, 'getSearchObservation')
                           else canonical_board)
            action = players[curPlayer](observation, it)
            valids = self.game.getValidMoves(canonical_board, 0)

            if verbose:
                print(f'P{curPlayer} decided to {self.game.moveToString(action, curPlayer)}')

            if valids[action] == 0:
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action, random_seed=0)
            curPlayer = int(curPlayer)

            # if verbose:
            #     print(f'state = "{self.serialize_state(board, curPlayer, it)}"')
        if verbose:
            if self.display:
                self.display(board)
            print("Game over: Turn ", str(it), "Result ", self.game.getGameEnded(board, curPlayer))
        else:
            if initial_state != "":
                print(f"Game over: {self.game.getScore(board, 0)} - {self.game.getScore(board, 1)}")

        MCTS.reset_all_search_trees()
            
        return self.game.getGameEnded(board, curPlayer)[0]

    def playGames(self, num, initial_state="", verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        ratio_boundaries = [        1-0.60,        1-0.55,        0.55,        0.60         ]
        colors           = ['#d60000',     '#d66b00',     '#f9f900',   '#a0d600',  '#6b8e00'] #https://icolorpalette.com/ff3b3b_ff9d3b_ffce3b_ffff3b_ceff3b
        if self.macos_terminal:
            colors = ['RED', 'MAGENTA', 'YELLOW', 'CYAN', 'GREEN']

        oneWon, twoWon, draws = 0, 0, 0
        t = trange(num, desc="Arena.playGames", ncols=120, disable=None)
        for i in t:
            # Since trees may not be resetted, the first games (1vs2) can't be
            # considered as fair as the last games (2vs1). Switching between 
            # 1vs2 and 2vs1 like below seems more fair:
            # 1 2 2 1   1 2 2 1  ...
            one_vs_two = (i%4 == 0) or (i%4 == 3) or (initial_state != "")
            t.set_description('Arena ' + ('(1 vs 2)' if one_vs_two else '(2 vs 1)'), refresh=False)
            gameResult = self.playGame(verbose=verbose, initial_state=initial_state, other_way=not one_vs_two)
            if gameResult == (1. if one_vs_two else -1.):
                oneWon += 1
            elif gameResult == (-1. if one_vs_two else 1.):
                twoWon += 1
            else:
                draws += 1

            t.set_postfix(one_wins=oneWon, two_wins=twoWon, refresh=False)
            ratio = oneWon / (oneWon+twoWon) if oneWon+twoWon>0 else 0.5
            t.colour = colors[bisect.bisect_right(ratio_boundaries, ratio)]
        t.close()

        return oneWon, twoWon, draws
