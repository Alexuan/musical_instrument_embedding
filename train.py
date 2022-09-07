import sys
import shutil

from musyn.utils import Config
from musyn.utils import mkdir_or_exist
from musyn.engine import create_engine


if __name__ == "__main__":

    cfg = Config.fromfile(sys.argv[1])

    # log
    mkdir_or_exist(cfg.log_config.dir)
    shutil.copy(sys.argv[1], cfg.log_config.dir)

    # engine
    trainer = create_engine(cfg)
    trainer.train()
