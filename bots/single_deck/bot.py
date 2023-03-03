import torch
import numpy as np

from bots.custom_bot import BotBase
from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from bots.single_deck.nn import BoardEmbedding, DenseNet
from constants import UNIT_NAMES, PRINCESS_HP, TILES_X, TILES_Y



class SingleDeckBot(BotBase):
    """
    Can only see 8 different cards, that are all in his deck.
    """

    def __init__(self, unit_model_path, number_model_path, side_model_path, deck_names):
        self.unit_detector = UnitDetector(unit_model_path, side_model_path, deck_names)
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.board_emb = BoardEmbedding()
        self.Q_net = DenseNet([512+51, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)

        self.label_to_deck_id = {UNIT_NAMES.index(name): i for i, name in enumerate(deck_names)}
        self.princess_damaged = {"right": False, "left": False}

    def _get_context(self, numbers, cards):
        context = torch.zeros((6 + 1 + 32 + 4 + 8), dtype=torch.float32)  # turret health, elixir, handcards, handcards ready, next handcard

        for i, team in enumerate(["ally", "enemy"]):
            context[i*3] = numbers[f"{team}_king_hp"]["number"]
            level = numbers[f"{team}_king_level"]["number"]
            for j, side in enumerate(["right", "left"]):
                hp = numbers[f"{side}_{team}_princess_hp"]["number"]
                if team == "ally" and hp == -1 and not self.princess_damaged[side]:
                    hp = PRINCESS_HP[level]
                elif hp >= 0 and not self.princess_damaged[side]:
                    self.princess_damaged[side] = True

                context[i*3 + j + 1] = hp if hp >= 0 else 0.0

        context[6] = numbers["elixir"]["number"]

        
        handcards = sorted(filter(lambda x: x["deck_id"] >= 0, cards[1:]), key=lambda x: x["deck_id"])
        self.sorted_handcards = handcards
        for i in range(min(len(handcards), 4)):
            idx = handcards[i]["deck_id"] + 7 + i*8
            context[idx] = 1.0
            context[39+i] = int(handcards[i]["ready"])

        idx = cards[0]["deck_id"] + 43
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

    @torch.no_grad()
    def get_state(self, image):
        state = {
            "units": self.unit_detector.run(image),  # label, tile, side
            "numbers": self.number_detector.run(image),
            "cards": self.card_detector.run(image),
        }

        self.handcards = map(lambda x: x["name"], state["cards"][1:])
        self.illegal_actions = torch.tensor(list(map(lambda x: (x["deck_id"]==-1) or (x["ready"]==0), state["cards"][1:])) + [False])

        board = self._get_board_state(state["units"])
        emb = self.board_emb(board)

        context = self._get_context(state["numbers"], state["cards"])

        return torch.cat((emb, context))
    
    @torch.no_grad()
    def get_actions(self, state, eps=0.01):
        if np.random.rand() < eps:
            action = (1 - self.illegal_actions.to(torch.float)).multinomial(1)
        else:
            q_vals = self.Q_net(state)
            q_vals[self.illegal_actions] = -torch.inf
            action = torch.argmax(q_vals)

        if action == 5:
            return -1
        
        slot_idx = self.handcards.index(self.sorted_handcards[action]["name"])
        return slot_idx


if __name__ == "__main__":
    # debugging purposes
    from PIL import ImageDraw, ImageFont

    from constants import TILE_WIDTH, TILE_HEIGHT, TOWER_HP_BOXES, ELIXIR_BOUNDING_BOX


    bot = SingleDeckBot()

    font = ImageFont.load_default()

    i = 0
    image = bot.screen.take_screenshot()
    while bot.is_game_end(image):
        state = bot.get_state(image)
        action = bot.get_actions(state)
        bot.play_actions(action)

        units = bot.unit_detector.run(image)

        board = bot._get_board_state(units)
        context = state[512:]

        ak, ar, al, ek, er, el = context[:6]
        hp_nums = [ek, ak, ar, al, er, el]

        draw = ImageDraw.Draw(image)
        for i in range(6):
            num = f"{hp_nums[i]}"
            _, (x1, y1, _, _) = TOWER_HP_BOXES[i]

            draw.text((x1, y1), num, fill="black", font=font)

        elixir = f"{context[6]}"
        draw.text(ELIXIR_BOUNDING_BOX[:2], elixir)

        for i in range(board.shape[1]):
            for j in range(board.shape[2]):
                for c in range(board.shape[0]):
                    if board[c, i, j] != 1.0:
                        continue

                    x, y = UnitDetector.tile_to_xy(j, i)
                    x1 = x - TILE_WIDTH/4
                    x2 = x + TILE_WIDTH/4
                    y1 = y - TILE_HEIGHT/4
                    y2 = y + TILE_HEIGHT/4

                    color = "red"
                    if c < 8:
                        color = "blue"
                    draw.ellipse([(x1, y1), (x2, y2)], outline=color, width=1.0)

        image.save(f"./output/debug/debug_img_{i}.png")
        
        image = bot.screen.take_screenshot()
        i += 1

    victory = bot.is_victory(image)
    # TODO draw victory or loss on image
    image.save(f"./output/debug/debug_img_{i}.png")
