import logging
import os

def _get_logger_level(logger_level:str) -> int:
    if logger_level == 'DEBUG':
        return logging.DEBUG
    elif logger_level == 'INFO':
        return logging.INFO
    elif logger_level == 'WARNING':
        return logging.WARNING
    elif logger_level == 'ERROR':
        return logging.ERROR
    elif logger_level == 'CRITICAL':
        return logging.CRITICAL
    else:
        raise ValueError(f'Unknown logger level : {logger_level}')


def logger_init(logger, output_dir:str=None, logger_level:str='INFO', save_as_file:bool=True):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    logger_level = _get_logger_level(logger_level=logger_level)

    logger.setLevel(logger_level)

    if save_as_file: 
        assert output_dir is not None
        
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        logging_output_file = os.path.join(output_dir, "output.log")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler = logging.FileHandler(logging_output_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f'LOGGING TO OUTPUT FILE : {logging_output_file}')
