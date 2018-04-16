FILTERS = [32, 64, 96, 128, 196, 256]
FILTERS_128 = [32, 64, 80, 96, 128, 196, 256]
FILTERS_SMALL = [8, 16, 32, 48, 64, 96]

REGIONS_PIN = [(67, 10, 112, 69), (277, 10, 322, 69)]

FILTERS_PIN = [32, 64, 96, 128, 196]

MAIN_OBJECT = {
    'main': {'width': 128, 'height': 128, 'region': 'all', 'filters': FILTERS_128, 'scope': 'main', 'reuse': False},
}
PIN_OBJECT = {
    'pin1': {'width': 64, 'height': 64, 'region': (67, 10, 112, 69), 'filters': FILTERS_SMALL, 'scope': 'up', 'reuse': False},
    'pin2': {'width': 64, 'height': 64, 'region': (277, 10, 322, 69), 'filters': FILTERS_SMALL, 'scope': 'up', 'reuse': True},
    'pin3': {'width': 64, 'height': 64, 'region': (105, 275, 285, 320), 'filters': FILTERS_SMALL, 'scope': 'down', 'reuse': False},
    'main': MAIN_OBJECT['main']
}