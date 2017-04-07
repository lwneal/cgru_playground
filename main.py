"""
This script parses arguments into a dict **params, and calls module_name.main(**params)

Usage:
        main.py <module_name> [options]

Arguments:
        module_name: Name of the Python module to run

Options:
      --model_filename=<model>          Name of saved .h5 parameter files after each epoch. [default: None]
      --epochs=<epochs>                 Number of epochs to train [default: 2000].
      --batches_per_epoch=<b>           Number of batches per epoch [default: 32].
      --batch_size=<size>               Batch size for training [default: 16]
      --width=<width>                   Width of generated training bitmaps [default: 224].
      --cgru_size_1=<size>              Number of units in first CGRU layer [default: 128].
      --cgru_size_2=<size>              Number of units in second CGRU layer [default: 256].
      --validate=<validate>             Validate instead of training [default: False].
"""
from docopt import docopt
from pprint import pprint

def get_params():
    args = docopt(__doc__)
    return {argname(k): argval(args[k]) for k in args}


def argname(k):
    return k.strip('<').strip('>').strip('--').replace('-', '_')


def argval(val):
    if hasattr(val, 'lower') and val.lower() in ['true', 'false']:
        return val.lower().startswith('t')
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val == 'None':
        return None
    return val


if __name__ == '__main__':
    params = get_params()
    module = __import__(params['module_name'].rstrip('.py'))
    module.main(**params)
