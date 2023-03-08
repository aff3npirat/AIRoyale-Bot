DATA_DIR = "./data/"

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

# Bounding box of elixir, turret helth bars, king level
ELIXIR_X = 100
ELIXIR_Y = 628
ELIXIR_W = 250
ELIXIR_H = 15
ELIXIR_BOUNDING_BOX = (ELIXIR_X, ELIXIR_Y, ELIXIR_X + ELIXIR_W, ELIXIR_Y + ELIXIR_H)

_W = 28
_H = 7
KING_HP_X = 188
ALLY_KING_HP_Y = 15
ENEMY_KING_HP_Y = 495
LEFT_PRINCESS_HP_X = 74
RIGHT_PRINCESS_HP_X = 266
ALLY_PRINCESS_HP_Y = 401
ENEMY_PRINCESS_HP_Y = 93
ENEMY_KING_LEVEL_Y = 19
ALLY_KING_LEVEL_Y = 487
KING_LEVEL_X = 134
KING_LEVEL_2_X = KING_LEVEL_X + _W
TOWER_HP_BOXES = [
    ['enemy_king_hp', (KING_HP_X, ALLY_KING_HP_Y, KING_HP_X + _W, ALLY_KING_HP_Y + _H)],
    ['ally_king_hp', (KING_HP_X, ENEMY_KING_HP_Y, KING_HP_X + _W, ENEMY_KING_HP_Y + _H)],
    ['right_ally_princess_hp',
     (RIGHT_PRINCESS_HP_X, ALLY_PRINCESS_HP_Y, RIGHT_PRINCESS_HP_X + _W, ALLY_PRINCESS_HP_Y + _H)],
    ['left_ally_princess_hp',
     (LEFT_PRINCESS_HP_X, ALLY_PRINCESS_HP_Y, LEFT_PRINCESS_HP_X + _W, ALLY_PRINCESS_HP_Y + _H)],
    ['right_enemy_princess_hp',
     (RIGHT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y, RIGHT_PRINCESS_HP_X + _W, ENEMY_PRINCESS_HP_Y + _H)],
    ['left_enemy_princess_hp',
     (LEFT_PRINCESS_HP_X, ENEMY_PRINCESS_HP_Y, LEFT_PRINCESS_HP_X + _W, ENEMY_PRINCESS_HP_Y + _H)],
]
KING_LEVEL_BOXES = [
    ['enemy_king_level', (KING_LEVEL_X, ENEMY_KING_LEVEL_Y, KING_LEVEL_X + _W, ENEMY_KING_LEVEL_Y + _H)],
    ['enemy_king_level_2', (KING_LEVEL_2_X, ENEMY_KING_LEVEL_Y, KING_LEVEL_2_X + _W, ENEMY_KING_LEVEL_Y + _H)],
    ['ally_king_level', (KING_LEVEL_X, ALLY_KING_LEVEL_Y, KING_LEVEL_X + _W, ALLY_KING_LEVEL_Y + _H)],
    ['ally_king_level_2', (KING_LEVEL_2_X, ALLY_KING_LEVEL_Y, KING_LEVEL_2_X + _W, ALLY_KING_LEVEL_Y + _H)],
]
NUMBER_HEIGHT = 16
NUMBER_WIDTH = 64

# HP
KING_HP = [2400, 2568, 2736, 2904, 3096, 3312, 3528, 3768, 4008, 4392, 4824, 5304, 5832, 6408]
PRINCESS_HP = [1400, 1512, 1624, 1750, 1890, 2030, 2184, 2352, 2534, 2786, 3052, 3346, 3668, 4032]

# hand cards
HAND_SIZE = 5
DECK_SIZE = 8
CARD_Y = 545
CARD_INIT_X = 87
CARD_WIDTH = 55
CARD_HEIGHT = 65
CARD_DELTA_X = 69
CARD_CONFIG = [
    (19, 605, 51, 645),  # next card
    (CARD_INIT_X, CARD_Y, CARD_INIT_X + CARD_WIDTH, CARD_Y + CARD_HEIGHT),  # handcard 1
    (CARD_INIT_X + CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + CARD_DELTA_X, CARD_Y + CARD_HEIGHT),  # handcard 2
    (CARD_INIT_X + 2 * CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + 2 * CARD_DELTA_X, CARD_Y + CARD_HEIGHT),
    (CARD_INIT_X + 3 * CARD_DELTA_X, CARD_Y, CARD_INIT_X + CARD_WIDTH + 3 * CARD_DELTA_X, CARD_Y + CARD_HEIGHT),
]

# Multihash coefficients
MULTI_HASH_SCALE = 0.355
MULTI_HASH_INTERCEPT = 163

# card names
UNIT_NAMES = [
    'archer',
    'arrows', 
    'babydragon', 
    'cagegoblin', 
    'fireball', 
    'giant', 
    'goblin', 
    'goblincage',
    'goblinhut',
    'hunter', 
    'knight', 
    'minion', 
    'minipekka', 
    'musketeer', 
    'prince', 
    'skeleton'
    'speargoblin', 
    'tombstone', 
    'valkyrie', 
    'wallbreaker', 
]

CARD_TO_UNITS = {
    "archers": "archer",
    "minions": "minion",
    "goblins": "goblin",
}