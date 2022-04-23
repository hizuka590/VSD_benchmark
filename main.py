import json
import os

import yaml
from absl import app
from absl import flags, logging
from absl.logging import info
from easydict import EasyDict

from imenh.launch.launch import ParallelLaunch

FLAGS = flags.FLAGS

flags.DEFINE_string("yaml_file", None, "The config file.")
flags.DEFINE_string("RESUME_PATH", None, "The RESUME.PATH")
flags.DEFINE_string("RESUME_TYPE", None, "The RESUME.PATH")


def init_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    # 0. logging
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    config["SAVE_DIR"] = FLAGS.log_dir
    # 1. Resume
    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
    if FLAGS.RESUME_TYPE:
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
    # 2. info
    info(f"config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)


def main(args):
    config = init_config(FLAGS.yaml_file)
    # 0. logging
    # 1. init launcher
    launcher = ParallelLaunch(config)
    launcher.run()


if __name__ == "__main__":
    app.run(main)
