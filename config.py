FILTERS = [32, 64, 96, 128, 196, 256]
FILTERS_128 = [32, 64, 80, 96, 128, 196, 256]
FILTERS_D6 = [32, 64, 128, 196, 256, 320]
FILTERS_D6_SMALL = [32, 64, 96, 128, 196, 256]
FILTERS_SMALL = [8, 16, 32, 48, 64, 96]

REGIONS_PIN = [(67, 10, 112, 69), (277, 10, 322, 69)]

FILTERS_PIN = [32, 64, 96, 128, 196]

OBJECT_J0602 = {
    'main': {'width': 128, 'height': 128, 'region': 'all', 'filters': FILTERS_128, 'scope': 'main', 'reuse': False},
}
OBJECT_J0602_PIN = {
    'pin1': {'width': 64, 'height': 64, 'region': (67, 10, 112, 69), 'filters': FILTERS_SMALL, 'scope': 'up', 'reuse': False},
    'pin2': {'width': 64, 'height': 64, 'region': (277, 10, 322, 69), 'filters': FILTERS_SMALL, 'scope': 'up', 'reuse': True},
    'pin3': {'width': 64, 'height': 64, 'region': (105, 275, 285, 320), 'filters': FILTERS_SMALL, 'scope': 'down', 'reuse': False},
    'main': OBJECT_J0602['main']
}
OBJECT_J0601 = {
    'main': {'width': 320, 'height': 128, 'region': 'all', 'filters': FILTERS_D6, 'scope': 'main', 'reuse': False}
}
OBJECT_J0601_SPLIT = {
    'main': {'width': 320, 'height': 64, 'region': [(0, 0, 694, 63), (0, 216, 694, 279)], 'filters': FILTERS_D6_SMALL, 'scope': 'main', 'reuse': False}
}


def get_region(region_name):
    regions = {
        'default': OBJECT_J0602,
        'J0602': OBJECT_J0602,
        'J0602P': OBJECT_J0602_PIN,
        'J0601': OBJECT_J0601,
        'J0601_S': OBJECT_J0601_SPLIT
    }
    return regions[region_name]
