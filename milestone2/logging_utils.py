import logging

def enable_cloud_log():
    """ Enable logs using default StreamHandler """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
