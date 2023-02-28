import torch
import numpy as np

from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from bots.single_deck.board import BoardEmbedding
from bots.single_deck.q_net import QNet
from constants import CARD_NAMES, PRINCESS_HP, TILES_X, TILES_Y



class SingleDeckBot():
    """
    Can only see 8 different cards, that are all in his deck.
    """

    def __init__(self, unit_model_path, number_model_path, side_model_path, deck_names, board_embedding_size):
        self.unit_detector = UnitDetector(unit_model_path, side_model_path, deck_names)
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.board_emb = BoardEmbedding()
        self.Q_net = QNet()

        self.board_embedding_size = board_embedding_size

        self.label_to_deck_id = {CARD_NAMES.index(name): i for i, name in enumerate(deck_names)}
        self.princess_damaged = {"right": False, "left": False}

    def _get_context(self, state):
        context = torch.zeros((6 + 1 + 32 + 4 + 8), dtype=torch.float32)  # turret health, elixir, handcards, handcards ready, next handcard

        for i, team in enumerate(["ally", "enemy"]):
            context[i*3] = state["numbers"][f"{team}_king_hp"]["number"]
            level = state["numbers"][f"{team}_king_level"]["number"]
            for j, side in enumerate(["right", "left"]):
                hp = state["numbers"][f"{team}_{side}_princess_hp"]
                if team == "ally" and hp == -1 and not self.princess_damaged[side]:
                    hp = PRINCESS_HP[level]
                elif hp >= 0 and not self.princess_damaged[side]:
                    self.princess_damaged[side] = True

                context[i*3 + j + 1] = hp if hp >= 0 else 0.0

        context[6] = state["numbers"]["elixir"]["number"]

        handcards = sorted(state["cards"][1:], key=lambda x: x["deck_id"])
        handcards = filter(lambda x: x["deck_id"] >= 0, handcards)
        for i in range(max(len(handcards, 4))):
            idx = handcards[i]["deck_id"] + 7 + i*8
            context[idx] = 1.0
            context[39+i] = int(handcards[i]["ready"])

        idx = state["cards"][0]["deck_id"] + 43
        context[idx] = 1.0

        return context
    
    def _get_board_state(self, units):
        labels, tile_x, tile_y, team = units

        board = torch.zeros((16, TILES_Y, TILES_X), dtype=torch.float32)
        for i, label in enumerate(labels):
            if label not in self.label_to_deck_id:
                continue

            label = self.label_to_deck_id[label]
            channel_idx = team[i]*8 + label
            board[channel_idx, tile_y[i], tile_x[i]] = 1.0

        return board

    def set_state(self, image):
        state = {
            "units": self.unit_detector.run(image),  # label, tile, side
            "numbers": self.number_detector.run(image),
            "cards": self.card_detector.run(image),
        }

        board = self._get_board_state(state["units"])
        emb = self.board_emb(board.squeeze(0))[0]

        context = self._get_context(state)

        context = torch.cat((emb, context))
        self.state = context


if __name__ == "__main__":
    # debugging purposes

    bot = SingleDeckBot()
    # TODO: create screenshot -> get state -> get board and context -> display on screen

        
