import numpy as np

from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from state.board import BoardEmbedding
from constants import CARD_NAMES



class SingleDeckBot():
    """
    Can only see 8 different cards, that are all in his deck.
    """

    def __init__(self, unit_model_path, number_model_path, side_model_path, deck_names, board_embedding_size):
        self.unit_detector = UnitDetector(unit_model_path, side_model_path, deck_names)
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.board_emb = BoardEmbedding()

        self.board_embedding_size = board_embedding_size

        self.label_to_deck_id = {CARD_NAMES.index(name): i for i, name in enumerate(deck_names)}

    def set_state(self, image):
        state = {
            "units": self.unit_detector.run(image),  # label, tile, side
            "numbers": self.number_detector.run(image),
            "cards": self.card_detector.run(image),
        }

        context = np.zeros((6 + 1 + 32 + 4 + 8))  # turret health, elixir, handcards, handcards ready, next handcard

        context[:6] = [
            state["numbers"][x]["number"] for x in [
                "ally_king_hp","ally_right_princess_hp", "ally_left_princess_hp", "enemy_king_hp","enemy_right_princess_hp", "enemy_left_princess_hp", 
            ]
        ]

        context[6] = state["numbers"]["elixir"]["number"]

        handcards = sorted(state["cards"][1:], key=lambda x: x["deck_id"])
        handcards = filter(lambda x: x["deck_id"] >= 0, handcards)
        for i in range(max(len(handcards, 4))):
            idx = handcards[i]["deck_id"] + 7 + i*8
            context[idx] = 1.0
            context[39+i] = int(handcards[i]["ready"])

        idx = state["cards"][0]["deck_id"] + 43
        context[idx] = 1.0

