import os
import sys
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

INFO_SLACK = 25
logging.addLevelName(INFO_SLACK, "INFO_SLACK")

class SlackHandler(logging.StreamHandler):

    def __init__(self, token, user_id=None, channel_id=None, run_name=None):
        super().__init__()
        assert user_id or channel_id, "user_id or channel_id must be specified."
        self.client = WebClient(token=token)
        if channel_id:
            self.channel_id = channel_id
        else:
            res = self.client.conversations_open(users=user_id)
            self.channel_id = res["channel"]["id"]
        
        if run_name is None:
            run_name = "undefined"
        
        # create thread:
        res = self.send_message("run_name: " + run_name)
        self.thread_ts = res["ts"]
        print(res)
            
        print(self.channel_id)

    def emit(self, record):
        msg = self.format(record)
        # self.send_message(msg)
        self.send_message_to_thread(msg)

    def send_message_to_thread(self, text):
        try:
            res = self.client.chat_postMessage(
                channel=self.channel_id,
                text=text,
                thread_ts=self.thread_ts
            )
        except SlackApiError as e:
            print(f"Failed to send message: {e.response['error']}")

    def send_message(self, text):
        try:
            res = self.client.chat_postMessage(
                channel=self.channel_id,
                text=text
            )
            return res
        except SlackApiError as e:
            print(f"Failed to send message: {e.response['error']}")
        



def init_logging(logger_name: str, log_dir:str, filename = "info.log", reset=False, run_name=None, sh=None, slack=True) -> logging.Logger:
    '''
    logging levels:
        DEBUG
        INFO
        WARNING
        ERROR
        CRITICAL
    
    output INFO, WARNING, ERROR, CRITICAL to console
    output ALL to file
    '''
    logger = logging.getLogger(logger_name)


    def info_slack_parent(self):
        def info_slack(message, *args, **kws):
            if self.isEnabledFor(INFO_SLACK):
                self._log(INFO_SLACK, message, args, **kws)
        return info_slack
    logger.info_slack = info_slack_parent(logger)
    
    logger.setLevel(logging.DEBUG)

    file_path = os.path.join(log_dir, filename)
    if reset and os.path.exists(file_path):
        os.remove(file_path)

    if logger.handlers == []:

        os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s/%(levelname)s/%(name)s/%(funcName)s():%(lineno)s\n%(message)s \n"
        )

        # file handler
        fh = logging.FileHandler(str(log_dir) + "/" + filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)


        # slack handler
        if slack:
            if sh is None:
                sh = SlackHandler(
                    token=os.environ["SLACK_LOGGER_TOKEN"],
                    user_id=os.environ["SLACK_USER_ID_TNLP"],
                    run_name=run_name
                )
                sh.setLevel(logging._nameToLevel["INFO_SLACK"])
                sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        if slack:
            logger.addHandler(sh)

        logger.propagate = False

    return logger, sh

if __name__ == "__main__":
    logger, sh = init_logging(__name__, log_dir='logs', filename='test_logger.log', reset=True, slack=False)
    logger.debug("debug")
    logger.info("info")
    logger.info_slack("info_slack")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")

    logger, _ = init_logging('second', log_dir='logs', filename='test_logger.log', reset=False)
    logger.debug("debug")
    logger.info("info")
    logger.info_slack("info_slack")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")

