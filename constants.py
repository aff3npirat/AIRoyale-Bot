DATA_DIR = "./data/"
ADB_PATH = r"..\adb\adb.exe"

# game screens
SCREEN_CONFIG = {
    "in_game": ((80, 630, 93, 645), 40),
    "victory": ((140, 304, 220, 314), 75),
    "game_end": ((143, 568, 225, 598), 20),
    "overtime": ((315, 8, 359, 13), 15),
    "clan": ((200, 595, 290, 650), 2.5),
}

# navigation
CLAN_MATCH = (152, 547)
CLAN_1V1 = (180, 468)
CLAN_1V1_ACCEPT = (291, 512)

# screenshot dimensions
SCREENSHOT_WIDTH = 368
SCREENSHOT_HEIGHT = 652

# tile data
TILES_X = 18
TILES_Y = 30
TILE_HEIGHT = 14.234
TILE_WIDTH = 17.667
TILE_INIT_X = 25
TILE_INIT_Y = 62
TILE_END_Y = 489

# size of image for side-detector
SIDE_W = 16
SIDE_H = 16

# size of image for unit detector
UNIT_Y_START = 32.6
UNIT_Y_END = 521.6
UNIT_W = 416
UNIT_H = 416

# side detector
BBOX_Y_OFFSET = 10  # enlarge bbox from unit detector to ensure healthbar/level is in crop

# Bounding box of elixir, turret helth bars, king level
ELIXIR_X = 120
ELIXIR_Y = 635
ELIXIR_DELTA_X = 25
ELIXIR_RED_THR = 80
ELIXIR_GREEN_THR = 230

HP_HEIGHT = 12
HP_WIDTH = 35
KING_HP_X = 189
ENEMY_KING_HP_Y = 11
ALLY_KING_HP_Y = 493
LEFT_PRINCESS_HP_X = 60
RIGHT_PRINCESS_HP_X = 278
ALLY_PRINCESS_HP_Y = 400
ENEMY_PRINCESS_HP_Y = 92
TOWER_HP_BOXES = [
    ['enemy_king_hp', (KING_HP_X, ENEMY_KING_HP_Y, KING_HP_X + HP_WIDTH, ENEMY_KING_HP_Y + HP_HEIGHT)],
    ['ally_king_hp', (KING_HP_X, ALLY_KING_HP_Y, KING_HP_X + HP_WIDTH, ALLY_KING_HP_Y + HP_HEIGHT)],
    ['right_ally_princess_hp',
     (RIGHT_PRINCESS_HP_X, ALLY_PRINCESS_HP_Y, RIGHT_PRINCESS_HP_X + HP_WIDTH, ALLY_PRINCESS_HP_Y + HP_HEIGHT)],
    ['left_ally_princess_hp',
     (LEFT_PRINCESS_HP_X, ALLY_PRINCESS_HP_Y, LEFT_PRINCESS_HP_X + HP_WIDTH, ALLY_PRINCESS_HP_Y + HP_HEIGHT)],
    ['right_enemy_princess_hp',
     (RIGHT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y, RIGHT_PRINCESS_HP_X + HP_WIDTH, ENEMY_PRINCESS_HP_Y + HP_HEIGHT)],
    ['left_enemy_princess_hp',
     (LEFT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y, LEFT_PRINCESS_HP_X + HP_WIDTH, ENEMY_PRINCESS_HP_Y + HP_HEIGHT)],
]

NUMBER_HEIGHT = 64
NUMBER_WIDTH = 64

# time bounding box
TIME_BOX = (312, 15, 312+HP_WIDTH*2, 15+20)

# HP
KING_HP = [2400, 2568, 2736, 2904, 3096, 3312, 3528, 3768, 4008, 4392, 4824, 5304, 5832, 6408]
PRINCESS_HP = [1400, 1512, 1624, 1750, 1890, 2030, 2184, 2352, 2534, 2786, 3052, 3346, 3668, 4032]

# hand cards
HAND_SIZE = 5
DECK_SIZE = 8
CARD_Y = 557
CARD_INIT_X = 87
CARD_WIDTH = 55
CARD_HEIGHT = 61
CARD_DELTA_X = 69
CARD_CONFIG = [
    (22, 615, 47, 643),  # next card
    (CARD_INIT_X, CARD_Y, CARD_INIT_X + CARD_WIDTH, CARD_Y + CARD_HEIGHT),  # handcard 1
    (CARD_INIT_X + CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + CARD_DELTA_X, CARD_Y + CARD_HEIGHT),  # handcard 2
    (CARD_INIT_X + 2 * CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + 2 * CARD_DELTA_X, CARD_Y + CARD_HEIGHT),
    (CARD_INIT_X + 3 * CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + 3 * CARD_DELTA_X, CARD_Y + CARD_HEIGHT),
]

# Multihash coefficients
MULTI_HASH_SCALE = 0.3246317505836487
MULTI_HASH_INTERCEPT = 160.90733337402344
