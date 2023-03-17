import torch
import numpy as np

from bots.custom_bot import BotBase
from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from bots.single_deck.nn import BoardEmbedding, DenseNet
from timing import exec_time
from constants import TILES_X, TILES_Y



EMB_SIZE = 512
OVERTIME = 0
HEALTH_START = OVERTIME + 1
HEALTH_END = HEALTH_START + 4
ELIXIR = HEALTH_END
CARDS_START = ELIXIR + 1
CARDS_END = CARDS_START + 32
READY_START = CARDS_END
READY_END = READY_START + 4
NEXT_CARD_START = READY_END
NEXT_CARD_END = NEXT_CARD_START + 8


class SingleDeckBot(BotBase):
    """
    Can only see 8 different cards, that are all in his deck.
    """

    def __init__(self, side, unit_model_path, number_model_path, side_model_path, deck_names, hash_size=8, king_levels=None, port=5555):
        super().__init__(hash_size=hash_size, port=port)

        tile_y = 22
        if side == "right":
            tile_x = 14
        else:
            tile_x = 3

        self.place_pos = UnitDetector.tile_to_xy(tile_x, tile_y)

        self.unit_detector = UnitDetector(unit_model_path, side_model_path, [i for i in range(8)])
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.board_emb = BoardEmbedding()
        self.Q_net = DenseNet([512+NEXT_CARD_END, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)

        self.towers_destroyed = {k: False for k in ["enemy_king_hp", "ally_king_hp", f"{side}_ally_princess_hp", f"{side}_enemy_princess_hp"]}
        self.towers_unhit = {k: True for k in self.towers_destroyed}
        self.king_levels = king_levels if king_levels is not None else {"ally": 1, "enemy": 1}

    @exec_time
    def _get_context(self, numbers, cards, overtime):
        context = torch.zeros((NEXT_CARD_END), dtype=torch.float32)  # overtime, turret health, elixir, handcards, handcards ready, next handcard

        context[OVERTIME] = 1.0 if overtime else 0.0

        for i, name in enumerate(self.towers_destroyed):
            if self.towers_destroyed[name]:
                hp = 0.0
            elif self.towers_unhit[name]:
                hp = 1.0
            else:
                hp = numbers[name]["number"]

            context[HEALTH_START + i] = hp


        context[ELIXIR] = numbers["elixir"]["number"]


        handcards = sorted(filter(lambda x: x["deck_id"] >= 0, cards[1:]), key=lambda x: x["deck_id"])
        self.sorted_handcards = handcards
        for i in range(len(handcards)):
            idx = handcards[i]["deck_id"] + CARDS_START + i*8
            context[idx] = 1.0
            context[READY_START+i] = int(handcards[i]["ready"])

        # next card
        if cards[0]["deck_id"] >= 0:
            idx = cards[0]["deck_id"] + NEXT_CARD_START
            context[idx] = 1.0

        return context
    
    @exec_time
    def _get_board_state(self, units):
        labels, bboxes, team = units

        x1 = bboxes[:, 0]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        center_x = x1 + (x2 - x1)/2

        tile_x, tile_y = UnitDetector.xy_to_tile(center_x, y2)

        labels = labels.astype(np.int32)
        tile_x = tile_x.astype(np.int32)
        tile_y = tile_y.astype(np.int32)
        team = team.astype(np.int32)

        board = torch.zeros((16, TILES_Y, TILES_X), dtype=torch.float32)
        for i, label in enumerate(labels):
            channel_idx = team[i]*8 + label
            tx = tile_x[i]
            ty = tile_y[i]
            if board[channel_idx, ty, tx] == 1.0:
                for x in [-1, 1]:
                    for y in [-1, 1]:
                        if board[channel_idx, ty+y, tx+x] == 0.0:
                            tx += x
                            ty += y
                            break

            board[channel_idx, ty, tx] = 1.0

        return board

    @exec_time
    @torch.no_grad()
    def get_state(self, image):
        units = self.unit_detector.run(image)  # label, tile, side
        numbers = self.number_detector.run(image)
        cards = self.card_detector.run(image)

        elixir = numbers["elixir"]["number"]
        for i in range(4):
            cards[i+1]["ready"] = (cards[i+1]["cost"]<=elixir)

        if numbers["enemy_king_hp"]["number"] == 1 and self.towers_unhit["enemy_king_hp"]:
            numbers["enemy_king_hp"]["number"] = -1

        for name in self.towers_destroyed:
            num = numbers[name]["number"]
            if num >= 0.0 and self.towers_unhit[name]:
                self.towers_unhit[name] = False
            elif num < 0 and not self.towers_unhit[name]:
                self.towers_destroyed[name] = True

        NumberDetector.relative_tower_hp(numbers, king_level=self.king_levels)

        overtime = self.detect_game_screen(image, "overtime")

        self.handcards = [x["name"] for x in cards[1:]]
        context = self._get_context(numbers, cards, overtime)

        board = self._get_board_state(units)
        emb = self.board_emb(board)

        return torch.cat((emb, context))
    
    @exec_time
    @torch.no_grad()
    def get_actions(self, state, eps=0.0):
        N_cards = len(self.sorted_handcards)
        illegal_actions = [i+1 for i in range(4) if i>=N_cards or not self.sorted_handcards[i]["ready"]]

        if np.random.rand() < eps:
            legal_actions = [i for i in range(5) if i not in illegal_actions]
            action = np.random.choice(np.array(legal_actions))
        else:
            q_vals = self.Q_net(state)
            q_vals[illegal_actions] = -torch.inf
            action = torch.argmax(q_vals)

        if action == 0:
            return -1
        
        action -= 1
        slot_idx = self.handcards.index(self.sorted_handcards[action]["name"])
        return slot_idx
    
    def play_actions(self, actions):
        if actions == -1:
            return

        x, y = self.slot_to_xy(actions)

        self.screen.click(x, y)
        self.screen.click(*self.place_pos)
    


if __name__ == "__main__":
    # debugging purposes
    import os
    import time
    import logging

    import cv2
    from PIL import ImageDraw, ImageFont, Image

    import timing
    from constants import TILE_WIDTH, TILE_HEIGHT, TOWER_HP_BOXES, CARD_CONFIG, SCREEN_CONFIG

    UNIT_NAMES = [
        'archer',
        'arrows',
        'giant',
        'knight',
        'minion',
        'minipekka',
        'musketeer',
        'speargoblin',
    ]


    OUTPUT = "./output/debug/single_deck_bot_"
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
        os.makedirs(f"{OUTPUT}/raw")

    bot_logger = logging.getLogger("bot")
    bot_logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(os.path.join(OUTPUT, "bot.log"), mode="w+")
    bot_logger.addHandler(handler_file)
    bot_logger.info("Initialized bot logging")

    time_logger = logging.getLogger("time")
    time_logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(os.path.join(OUTPUT, "time.log"), mode="w+")
    time_logger.addHandler(handler_file)
    time_logger.info("Initialized time logging")

    timing.logger = time_logger


    deck_names = ["minions", "giant", "speargoblins", "musketeer", "minipekka", "knight", "archers", "arrows"]
    bot = SingleDeckBot(
        side="right",
        unit_model_path="./models/units_singledeck_cpu.onnx",
        number_model_path="./models/number_cpu.onnx",
        side_model_path="./models/side_cpu.onnx",
        deck_names=deck_names,
    )

    bot_logger.info(f"Ally units={bot.unit_detector.ally_units}")

    font = ImageFont.load_default()

    # class FakeScreen():
    #     def __init__(self, in_dir):
    #         self.i = 0
    #         self.in_dir = in_dir
    #         self.files = sorted(list(os.listdir(self.in_dir)), key=lambda x: int(x[:-4]))

    #     def take_screenshot(self):
    #         img = Image.open(os.path.join(self.in_dir, self.files[self.i]), "r")
    #         self.i += 1
    #         return img
        
    # bot.screen = FakeScreen(r"C:\Users\jurek\Desktop\projects\python\AI-Royale-dev\AIRoyale\data\game_screenshots\game_1")


    count = 0
    image = bot.screen.take_screenshot()
    width, height = image.size
    while bot.in_game(image):
        image.save(f"{OUTPUT}/raw/img_{count}.png")

        state = bot.get_state(image)
        action = bot.get_actions(state, eps=0.8)
        bot.play_actions(action)

        bot_logger.info(f"[{count}] action={action}, handcards={bot.handcards}, sorted_handcards={bot.sorted_handcards}, towers_destroyed={bot.towers_destroyed}, towers_unhit={bot.towers_unhit}")

        units = bot.unit_detector.run(image)
        numbers = bot.number_detector.run(image)

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

        # draw health
        for i in range(6):
            name, (x1, y1, _, _) = TOWER_HP_BOXES[i]
            num = f"{numbers[name]['number']}"

            draw_.text((x1, y1-10), num, fill="black", font=font, anchor="lb")

        # final image will contain bot view and unit detector view
        conc_img = Image.new("RGB", (width*2, height))
        conc_img.paste(image_, (width, 0))

        del image_
        del draw_


        board = bot._get_board_state(units)
        context = state[EMB_SIZE:]

        draw = ImageDraw.Draw(image)

        draw.text((width-50, 50), f"{count}", fill="black", font=font)

        # draw overtime
        overtime = context[OVERTIME]
        draw.text((307, 40), f"{'overtime' if overtime else 'normal'}", fill="black", font=font, anchor="lt")

        # draw health
        nums = context[HEALTH_START:HEALTH_END]
        for i in range(HEALTH_END-HEALTH_START):
            for name, (x1, y1, _, _) in TOWER_HP_BOXES:
                if name == list(bot.towers_destroyed.keys())[i]:
                    break
            num = f"{nums[i]:.2f}"

            draw.text((x1, y1-10), num, fill="black", font=font, anchor="lb")

        # draw elixir
        elixir = f"{context[ELIXIR]}"
        draw.text((120, 628), elixir, anchor="lb", font=font, fill="black")

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
            draw.rectangle([x1, y1, x2, y2], outline=color, fill=color, width=1)
            
            # draw unit labels from state vector
            draw.text((x2, y1), UNIT_NAMES[c%8], fill=color, font=font, anchor="lt")

        # draw choosen action
        if action == -1:
            bbox = CARD_CONFIG[1]
            draw.rectangle(bbox, outline="red", width=2)
        else:  # highlight choosen slot
            bbox = CARD_CONFIG[action+1]
            draw.rectangle(bbox, outline="green", width=2)

        conc_img.paste(image, (0, 0))
        conc_img.save(f"{OUTPUT}/img_{count}.png")

        image = bot.screen.take_screenshot()
        count += 1

    image.save(os.path.join(OUTPUT, "img_-1.png"))

    print("Detected game end")
    while not bot.is_game_end(image):
        image = bot.screen.take_screenshot()

    time.sleep(2.0)

    victory = f"{'victory' if bot.is_victory(image) else 'loss'}"
    print(f"Detected outcome {victory}")

    draw = ImageDraw.Draw(image)
    draw.text(SCREEN_CONFIG["victory"][0][:2], victory, fill="black", font=font, anchor="lb")

    conc_img = Image.new("RGB", (width*2, height), color="black")
    conc_img.paste(image, (0, 0))

    conc_img.save(os.path.join(OUTPUT, f"img_{count}.png"))

    # create a gif of all images
    files = list(os.listdir(OUTPUT))
    files = sorted(filter(lambda x: (".png" in x and x.split("_")[1][:-4] != "-1"), files), key=lambda x: int(x.split("_")[1][:-4]))

    video = cv2.VideoWriter(os.path.join(OUTPUT, "debug_vision.avi"), 0, 1, (width*2, height))
    for file in files:
        img = cv2.imread(os.path.join(OUTPUT, file))
        video.write(img)
    
    video.release()
