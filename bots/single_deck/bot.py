import torch
import numpy as np

from bots.custom_bot import BotBase
from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from bots.single_deck.nn import BoardEmbedding, DenseNet
from constants import UNIT_NAMES, PRINCESS_HP, TILES_X, TILES_Y, CARD_TO_UNITS



EMB_SIZE = 512
HEALTH_START = 0
HEALTH_END = 6
ELIXIR = 6
CARDS_START = 7
CARDS_END = 39
READY_START = 39
READY_END = 43
NEXT_CARD_START = 43
NEXT_CARD_END = 51


class SingleDeckBot(BotBase):
    """
    Can only see 8 different cards, that are all in his deck.
    """

    def __init__(self, unit_model_path, number_model_path, side_model_path, deck_names):
        super().__init__()

        self.unit_detector = UnitDetector(unit_model_path, side_model_path, deck_names)
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.board_emb = BoardEmbedding()
        self.Q_net = DenseNet([512+51, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)

        label_to_deck_id = {}
        for i, name in enumerate(deck_names):
            if name in CARD_TO_UNITS:
                name = CARD_TO_UNITS[name]
            label_to_deck_id[UNIT_NAMES.index(name)] = i
        self.label_to_deck_id = label_to_deck_id

        self.princess_damaged = {"right": False, "left": False}

    def _get_context(self, numbers, cards):
        context = torch.zeros((NEXT_CARD_END), dtype=torch.float32)  # turret health, elixir, handcards, handcards ready, next handcard

        for i, team in enumerate(["ally", "enemy"]):
            context[i*3] = numbers[f"{team}_king_hp"]["number"]
            for j, side in enumerate(["right", "left"]):
                hp = numbers[f"{side}_{team}_princess_hp"]["number"]
                if team == "ally" and hp < 0 and not self.princess_damaged[side]:
                    hp = 1.0
                elif team == "ally" and hp >= 0 and not self.princess_damaged[side]:
                    self.princess_damaged[side] = True

                context[i*3 + j + 1] = hp if hp >= 0 else 0.0

        context[ELIXIR] = numbers["elixir"]["number"]

        
        handcards = sorted(filter(lambda x: x["deck_id"] >= 0, cards[1:]), key=lambda x: x["deck_id"])
        self.sorted_handcards = handcards
        self.illegal_actions = 4 - len(handcards)
        for i in range(len(handcards)):
            idx = handcards[i]["deck_id"] + CARDS_START + i*8
            context[idx] = 1.0
            context[READY_START+i] = int(handcards[i]["ready"])

        idx = cards[0]["deck_id"] + READY_END
        context[idx] = 1.0

        return context
    
    def _get_board_state(self, units):
        labels, bboxes, tile_y, team = units
        tile_x, tile_y = UnitDetector.box_to_tile(bboxes)

        labels = labels.astype(np.int32)
        tile_x = tile_x.astype(np.int32)
        tile_y = tile_y.astype(np.int32)
        team = team.astype(np.int32)

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

        self.raw_state = state

        self.handcards = [x["name"] for x in state["cards"][1:]]
        context = self._get_context(state["numbers"], state["cards"])

        board = self._get_board_state(state["units"])
        emb = self.board_emb(board)

        return torch.cat((emb, context))
    
    @torch.no_grad()
    def get_actions(self, state, eps=0.0):
        if np.random.rand() < eps:
            action = np.random.choice(np.arange(5-self.illegal_actions))
        else:
            q_vals = self.Q_net(state)
            q_vals[-self.illegal_actions:] = -torch.inf
            action = torch.argmax(q_vals)

        if action == 0:
            return -1
        
        action -= 1
        slot_idx = self.handcards.index(self.sorted_handcards[action]["name"])
        return slot_idx
    
    @staticmethod
    def is_game_end(image):
        # TODO
        return False
    

if __name__ == "__main__":
    # debugging purposes
    import os

    from PIL import ImageDraw, ImageFont, Image

    from constants import TILE_WIDTH, TILE_HEIGHT, TOWER_HP_BOXES, ELIXIR_BOUNDING_BOX, CARD_CONFIG

    OUTPUT = "./debug/single_deck_bot"
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)


    deck_names = ["minions", "giant", "arrows", "musketeer", "minipekka", "knight", "archers", "fireball"]
    bot = SingleDeckBot(
        "./models/units_cpu.onnx",
        "./models/number_cpu.onnx",
        "./models/side_cpu.onnx",
        deck_names,
    )

    font = ImageFont.load_default()

    count = 0
    image = bot.screen.take_screenshot()
    width, height = image.size
    while not bot.is_game_end(image):
        state = bot.get_state(image)
        action = bot.get_actions(state)

        units = bot.raw_state["units"]

        torch.save(bot.raw_state)

        # draw unit labels from unit detector
        image_ = image.copy()
        draw_ = ImageDraw.Draw(image_)
        label, bboxes, team = units
        for i in range(len(label)):
            name = UNIT_NAMES[int(label[i])]
            color = "red"
            if team[i] == 0:
                color = "blue"
            x1, y1, x2, y2 = bboxes[i]
            draw_.rectangle((x1, y1, x2, y2), outline=color, width=2)
            draw_.text((x1, y1), name, fill=color, font=font, anchor="lb")

        # final image will contain bot view and unit detector view
        conc_img = Image.new("RGB", (width*2, height))
        conc_img.paste(image_, (width, 0))

        del image_
        del draw_


        board = bot._get_board_state(units)
        context = state[EMB_SIZE:]

        draw = ImageDraw.Draw(image)

        # draw health
        ak, ar, al, ek, er, el = context[:HEALTH_END]
        hp_nums = [ek, ak, ar, al, er, el]
        for i in range(6):
            num = f"{hp_nums[i]:.2f}"
            _, (x1, y1, _, _) = TOWER_HP_BOXES[i]

            draw.text((x1, y1), num, fill="black", font=font, anchor="lb")

        # draw elixir
        elixir = f"{context[ELIXIR]}"
        draw.text(ELIXIR_BOUNDING_BOX[:2], elixir, anchor="lb", font=font, fill="black")

        # draw next card
        next_card_id = context[NEXT_CARD_START:NEXT_CARD_END].nonzero(as_tuple=False)
        assert len(next_card_id) <= 1, "Detected more than 1 next card"
        if len(next_card_id) == 1:
            name = deck_names[int(next_card_id[0].item())]
            draw.text(CARD_CONFIG[0][:2], name, anchor="lb", fill="black", font=font)

        # draw handcards
        handcards = context[CARDS_START:CARDS_END]
        deck_ids = handcards.nonzero(as_tuple=False)
        ready = context[READY_START:READY_END]
        assert len(deck_ids) <= 4, "More than 4 handcards detected"
        assert torch.sum(ready) <= len(deck_ids), "Detected more handcards ready as cards on hand"
        for i in range(len(deck_ids)):
            name = deck_names[int(deck_ids[i].item() - i*8)]
            slot_idx = bot.handcards.index(name)
            ready_ = ready[i]

            color = "black" if ready_ else "red"
            draw.text(CARD_CONFIG[slot_idx+1][:2], name, anchor="lb", fill=color, font=font)

        # draw unit tiles
        for idxs in board.nonzero(as_tuple=False):
            c, y, x = idxs

            x, y = UnitDetector.tile_to_xy(x, y)
            x1 = x - TILE_WIDTH/4
            x2 = x + TILE_WIDTH/4
            y1 = y - TILE_HEIGHT/4
            y2 = y + TILE_HEIGHT/4

            color = "red"
            if c < 8:
                color = "blue"
            draw.ellipse([(x1, y1), (x2, y2)], outline=color, width=1)
            
            # draw unit labels from state vector
            draw.text((x2, y1), deck_names[c%8], fill=color, font=font, anchor="lt")

        # draw choosen action
        if action == -1:
            bbox = CARD_CONFIG[1]
            draw.rectangle(bbox, outline="red", width=2)
        else:  # highlight choosen slot
            bbox = CARD_CONFIG[action+1]
            draw.rectangle(bbox, outline="green", width=2)

        conc_img.paste(image, (0, 0))
        conc_img.save(f"./output/debug/debug_img_{count}.png")

        image = bot.screen.take_screenshot()
        count += 1

    victory = bot.is_victory(image)
    # TODO draw victory or loss on image
    image.save(f"./output/debug/debug_img_{i}.png")
