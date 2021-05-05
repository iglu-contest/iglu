block_map_ = {
    'cwc_minecraft_blue_rn': 'lapis_block',
    'cwc_minecraft_yellow_rn': 'gold_block',
    'cwc_minecraft_green_rn': 'emerald_block',
    'cwc_minecraft_orange_rn': 'brick_block',
    'cwc_minecraft_purple_rn': 'packed_ice',
    'cwc_minecraft_red_rn': 'redstone_block',
}

block_map = {
    'cwc_minecraft_blue_rn': 'malmomod:iglu_minecraft_blue_rn',
    'cwc_minecraft_yellow_rn': 'malmomod:iglu_minecraft_yellow_rn',
    'cwc_minecraft_green_rn': 'malmomod:iglu_minecraft_green_rn',
    'cwc_minecraft_orange_rn': 'malmomod:iglu_minecraft_orange_rn',
    'cwc_minecraft_purple_rn': 'malmomod:iglu_minecraft_purple_rn',
    'cwc_minecraft_red_rn': 'malmomod:iglu_minecraft_red_rn',
}

block2id = {
    b: i for i, b in enumerate(['air'] + list(block_map.values()))
}

BUILD_ZONE_SIZE_X = 11
BUILD_ZONE_SIZE_Y = 9
BUILD_ZONE_SIZE_Z = 11
GROUND_LEVEL = 226