import logging
import logging.config
from pathlib import Path
from ultocr.utils.utils_function import read_json


def setup_logging(save_dir, log_config='logger_config.json', default_level=logging.INFO):
    log_config = Path(__file__).parent.joinpath(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])
        logging.config.dictConfig(config)
    else:
        print('Warning: logging configuration file is not found in {}'.format(log_config))
        logging.basicConfig(level=default_level)
