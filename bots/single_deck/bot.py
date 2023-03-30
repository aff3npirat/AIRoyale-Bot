import time

import torch
import numpy as np

from bots.custom_bot import BotBase
from state.units import UnitDetector
from state.cards import BlueCardDetector
from state.numbers import NumberDetector
from bots.single_deck.nn import QNet
from timing import exec_time, intervall
from constants import TILES_X, TILES_Y



EMB_SIZE = 512
REMAINING_TIME = 0
HEALTH_START = REMAINING_TIME + 1
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

    def __init__(self, team, unit_model_path, number_model_path, side_model_path, deck_names, hash_size=8, king_levels=None, port=5555):
        super().__init__(hash_size=hash_size, port=port)

        if team == "blue":
            side = "left"
        else:
            side = "right"

        self.side = side

        deck_names = sorted(deck_names)

        self.unit_detector = UnitDetector(unit_model_path, side_model_path, [i for i in range(8)])
        self.number_detector = NumberDetector(number_model_path)
        self.card_detector = BlueCardDetector(card_names=deck_names)
        self.Q_net = QNet([512+NEXT_CARD_END, 128, 64, 5], activation="sigmoid", bias=True, feature_extractor=False)
        self.Q_net.eval()

        self.towers_destroyed = {k: False for k in ["enemy_king_hp", "ally_king_hp", f"{side}_ally_princess_hp", f"{side}_enemy_princess_hp"]}
        self.towers_unhit = {k: True for k in self.towers_destroyed}
        self.king_levels = king_levels if king_levels is not None else {"ally": 1, "enemy": 1}
        
        self.last_expense = 0
        self.approx_time = 10
        self.tic = None

    def init_model(self, state_dict):
        self.Q_net.load_state_dict(state_dict)

    @staticmethod
    def get_illegal_actions(state):
        _, context = state
        illegal_actions = []
        for i in range(4):
            card_idx = CARDS_START+i*8
            ready_idx = READY_START+i
            if torch.sum(context[card_idx:card_idx+8]) == 0 or context[ready_idx] == 0:
                illegal_actions.append(i+1)

        return illegal_actions

    @staticmethod
    def with_reward(episode, victory):
        experience = []
        (prev_board, prev_context), prev_action, prev_N_enemy = episode[0]
        for exp in episode[1:]:
            (board, context), action, N_enemy = exp

            reward = SingleDeckBot.get_reward(context, prev_context, prev_action, N_enemy, prev_N_enemy)

            experience.append(((prev_board, prev_context), prev_action, reward, False))
            (prev_board, prev_context), prev_action, prev_N_enemy = exp

        outcome_reward = 1 if victory else -1
        experience.append(((board, context), action, outcome_reward, True))

        return experience

    @staticmethod
    def get_reward(next_context, context, action, next_N_enemy, N_enemy):
        troop_reward = max(N_enemy - next_N_enemy, 0) * 2
        card_reward = -3 if action != -1 else 0
        
        tower_hp = next_context[HEALTH_START:HEALTH_END]
        prev_hp = context[HEALTH_START:HEALTH_END]
        enemy_towers = [0, 3]
        ally_towers = [1, 2]

        damage_reward = 0.0
        for idxs, scale in zip([enemy_towers, ally_towers], [20, -30]):
            damage_reward += scale * (torch.sum(prev_hp[idxs]>0.0) - torch.sum(tower_hp[idxs]>0.0))
            damage_reward += scale * torch.sum(prev_hp[idxs] - tower_hp[idxs])

        return troop_reward + card_reward + damage_reward

    def store_experience(self, state, actions):
        N_enemy = torch.sum(state[0][8:])
        self.replay_buffer.append((state, actions, N_enemy))

    @exec_time
    def _get_context(self, numbers, cards):
        context = torch.zeros((NEXT_CARD_END), dtype=torch.float32)  # time, turret health, elixir, handcards, handcards ready, next handcard

        rem_time = numbers["time"]["number"]
        if rem_time >= 0:
            rem_time /= 180
        else:
            rem_time /= 120
        context[REMAINING_TIME] = rem_time

        for i, name in enumerate(self.towers_destroyed):
            if self.towers_destroyed[name]:
                hp = 0.0
            elif self.towers_unhit[name]:
                hp = 1.0
            else:
                hp = numbers[name]["number"]

            context[HEALTH_START + i] = hp


        context[ELIXIR] = int(self.elixir)


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
    @intervall
    @torch.no_grad()
    def get_state(self, image):
        if self.tic is None:
            seconds_elapsed = 0
        else:
            seconds_elapsed = time.time() - self.tic
        self.tic = time.time()
        
        self.approx_time += seconds_elapsed

        units = self.unit_detector.run(image)  # label, tile, side
        numbers = self.number_detector.run(image)
        cards = self.card_detector.run(image)
        
        overtime = self.screen.detect_game_screen(image, "overtime")

        remaining_seconds = numbers["time"]["number"]
        if remaining_seconds == -1:
            remaining_seconds = 180 - self.approx_time
        elif overtime:
            remaining_seconds -= 121

        numbers["time"]["number"] = remaining_seconds

        elixir_gain = 2.8  # seconds per elixir
        if 0 <= remaining_seconds <= 60:
            elixir_gain /= 2
        elif remaining_seconds <= -61:
            elixir_gain /= 3

        if self.last_expense > 0:
            self.elixir = self.elixir - self.last_expense + seconds_elapsed/elixir_gain
        else:
            self.elixir = numbers["elixir"]["number"]
        for i in range(4):
            cards[i+1]["ready"] = (cards[i+1]["cost"]<=self.elixir)

        if numbers["enemy_king_hp"]["number"] == 1 and self.towers_unhit["enemy_king_hp"]:
            numbers["enemy_king_hp"]["number"] = -1

        for name in self.towers_destroyed:
            num = numbers[name]["number"]
            if num >= 0.0 and self.towers_unhit[name]:
                self.towers_unhit[name] = False
            elif num < 0 and not self.towers_unhit[name]:
                self.towers_destroyed[name] = True

        NumberDetector.relative_tower_hp(numbers, king_level=self.king_levels)

        self.handcards = cards[1:]
        context = self._get_context(numbers, cards)

        board = self._get_board_state(units)

        return board, context
    
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

        return action
    
    def play_actions(self, action):
        if action == 0:
            self.last_expense = 0
            return
        
        action -= 1
        names = [card["name"] for card in self.handcards]
        slot_idx = names.index(self.sorted_handcards[action]["name"])
        
        self.last_expense = self.handcards[slot_idx]["cost"]
        
        self.controller.select_place_unit(slot_idx, self.side)

    @exec_time
    def run(self, eps):
        image = self.controller.take_screenshot()
        while not self.screen.in_game(image):
            image = self.controller.take_screenshot()

        while self.screen.in_game(image):
            state = self.get_state(image)
            action = self.get_actions(state, eps=eps)
            self.play_actions(action)

            self.store_experience(state, action)

            image = self.controller.take_screenshot()

        while not self.screen.is_game_end(image):
            image = self.controller.take_screenshot()

        time.sleep(3.0)
        image = self.controller.take_screenshot()

        victory = self.screen.is_victory(image)

        return victory
    

def debug(id, team, port):
    import os
    import logging
    import time

    import cv2
    from PIL import ImageDraw, ImageFont, Image

    import timing
    from constants import TILE_WIDTH, TILE_HEIGHT, TOWER_HP_BOXES, CARD_CONFIG, SCREEN_CONFIG, PRINCESS_Y_OFFSET

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

    for i, (name, (x1, y1, x2, y2)) in enumerate(TOWER_HP_BOXES):
        if "princess" in name:
            TOWER_HP_BOXES[i][1] = (x1, y1-PRINCESS_Y_OFFSET, x2, y2-PRINCESS_Y_OFFSET)


    OUTPUT = f"./output/debug/single_deck_bot_{id}"
    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)
        os.makedirs(f"{OUTPUT}/raw")

    bot_logger = logging.getLogger("bot")
    bot_logger.setLevel(logging.INFO)
    handler_file = logging.FileHandler(os.path.join(OUTPUT, "bot.log"), mode="w+")
    bot_logger.addHandler(handler_file)
    bot_logger.info("Initialized bot logging")

    timing.init_logging(os.path.join(OUTPUT, "time.log"))


    deck_names = ["minions", "giant", "speargoblins", "musketeer", "minipekka", "knight", "archers", "arrows"]
    deck_names = sorted(deck_names)
    bot = SingleDeckBot(
        team=team,
        unit_model_path="./models/units_singledeck_cpu.onnx",
        number_model_path="./models/number_cpu.onnx",
        side_model_path="./models/side_cpu.onnx",
        deck_names=deck_names,
        port=port,
    )

    bot_logger.info(f"Ally units={bot.unit_detector.ally_units}")

    font = ImageFont.load_default()

    image = bot.controller.take_screenshot()
    while not bot.screen.in_game(image):
        image = bot.controller.take_screenshot()

    count = 0
    width, height = image.size
    while bot.screen.in_game(image):
        image.save(f"{OUTPUT}/raw/img_{count}.png")

        state = bot.get_state(image)
        action = bot.get_actions(state, eps=1.0)
        bot.play_actions(action)

        if action == 0:
            action_slot_idx = -1
        else:
            names = [card["name"] for card in bot.handcards]
            action_slot_idx = names.index(bot.sorted_handcards[action-1]["name"])
        bot_logger.info(f"[{count}] action={action}, slot_idx={action_slot_idx} handcards={bot.handcards}, sorted_handcards={bot.sorted_handcards}, towers_destroyed={bot.towers_destroyed}, towers_unhit={bot.towers_unhit}, approx_time={bot.approx_time}, last_expense={bot.last_expense}, elixir={bot.elixir}")

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

        # draw absolute time
        time_ = numbers["time"]["number"]
        if time_ >= 0:
            minutes = time_ // 60
            seconds = time_ % 60
            draw_.text((307, 40), f"{minutes}:{0 if seconds <= 9 else ''}{seconds}", fill="black", font=font, anchor="lt")
        else:
            draw_.text((307, 40), f"{time_}", fill="black", font=font, anchor="lt")


        # final image will contain bot view and unit detector view
        conc_img = Image.new("RGB", (width*2, height))
        conc_img.paste(image_, (width, 0))

        del image_
        del draw_


        board, context = state

        draw = ImageDraw.Draw(image)

        draw.text((width-50, 60), f"{count}", fill="black", font=font)

        # draw normed time
        time_ = context[REMAINING_TIME]
        draw.text((307, 40), f"{time_:.3f}", fill="black", font=font, anchor="lt")

        # draw overtime
        draw.text((307, 50), f"{'overtime' if time_ < 0 else 'normal'}", fill="black", font=font, anchor="lt")

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
            slot_idx = [card["name"] for card in bot.handcards].index(name)
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
        if action == 0:
            bbox = CARD_CONFIG[1]
            draw.rectangle(bbox, outline="red", width=2)
        else:  # highlight choosen slot
            bbox = CARD_CONFIG[action_slot_idx+1]
            draw.rectangle(bbox, outline="green", width=2)

        conc_img.paste(image, (0, 0))
        conc_img.save(f"{OUTPUT}/img_{count}.png")

        image = bot.controller.take_screenshot()
        count += 1

    image.save(os.path.join(OUTPUT, "img_-1.png"))

    print("Detected game end")
    while not bot.screen.is_game_end(image):
        image = bot.controller.take_screenshot()

    time.sleep(3.0)
    image = bot.controller.take_screenshot()

    victory = f"{'victory' if bot.screen.is_victory(image) else 'loss'}"
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

if __name__ == "__main__":
    # debugging purposes
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--team", type=str, default="blue")
    parser.add_argument("--id", type=str, default="")
    args = parser.parse_args()

    port = args.port
    team = args.team
    id_ = args.id
    debug(id_, team, port)
