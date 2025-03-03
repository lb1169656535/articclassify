import logging
def setup_logger():
    """配置日志记录器"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename="experiment_log.log",
        filemode="w"
    )
    return logging.getLogger()
