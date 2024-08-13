import logging

logger = None


def log(*args, logtype="info", sep=" "):
    getattr(logger, logtype)(sep.join(str(a) for a in args))


def initializeLog():
    global logger
    logging.basicConfig(
        level=logging.INFO,
        filename="app.log",
        format="[%(levelname)s] %(message)s",
        filemode="w",
    )
    logging.info("This will get logged")
    logger = logging.getLogger("root")
