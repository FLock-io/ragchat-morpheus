import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


class Logger:
    def __init__(self, log_name="", level="DEBUG", console_level="DEBUG", log_dir='logs'):

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.DEBUG)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if log_name == "":
            self.log_name = "ragchat_" + sys.argv[0].rsplit("/", 1)[-1].split(".")[0]
        else:
            self.log_name = "ragchat_" + log_name

        handler = TimedRotatingFileHandler("{}/{}-server-biz.log".format(log_dir, self.log_name),
                                           when='midnight',
                                           interval=1,
                                           backupCount=10)

        handler.setLevel(level.upper())

        handler_formatter = logging.Formatter(
            "[entry][ts]%(asctime)s.%(msecs)03d[/ts][lv]%(levelname)s[/lv][th]%(threadName)s[/th][lg]%(pathname)s[/lg]"
            "[cl]%(filename)s[/cl][m]%(funcName)s[/m][ln]%(lineno)s[/ln][bsid]%(process)d[/bsid][esid]{}[/esid]"
            "[cmid][/cmid][txt]%(message)s[/txt][ex][/ex][/entry]".format(log_name), "%Y-%m-%d$%H:%M:%S")
        # handler_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s",
        #                                       "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(handler_formatter)
        console = logging.StreamHandler()

        console.setLevel(console_level.upper())

        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)s - %(message)s",
                                              "%Y-%m-%d %H:%M:%S")
        console.setFormatter(console_formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def getlog(self):
        return self.logger


if __name__ == "__main__":
    log = Logger().getlog()
    log.info("asd")
    log2 = Logger().getlog()
    log2.info("fdghnd")
