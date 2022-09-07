"""This package includes all the modules related to engine

 To add a custom engine class called 'dummy', you need to add a file called 'dummy_trainer.py' and define a subclass 'DummyTrainer' inherited from BaseTrainer.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseTrainer.__init__(self, opt).
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the engine class by specifying flag '--engine_mode dummy'.
See our template engine class 'template_trainer.py' for more details.
"""

import importlib
from musyn.engine.base_trainer import BaseTrainer


def find_engine_using_name(engine_name):
    """Import the module "engine/[engine_name]_trainer.py".

    In the file, the class called DummyTrainer() will
    be instantiated. It has to be a subclass of BaseTrainer,
    and it is case-insensitive.
    """
    engine_filename = "musyn.engine." + engine_name + "_trainer"
    enginelib = importlib.import_module(engine_filename)
    trainer = None
    target_trainer_name = engine_name.replace('_', '') + 'trainer'
    for name, cls in enginelib.__dict__.items():
        if name.lower() == target_trainer_name.lower() \
           and issubclass(cls, BaseTrainer):
            trainer = cls

    if trainer is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (engine_filename, target_trainer_name))
        exit(0)

    return trainer


def create_engine(opt):
    """Create a engine given the option.

    This is the main interface between this package and 'train.py'/'inference.py'

    Example:
        >>> from engine import create_engine
        >>> trainer = create_engine(opt)
    """
    engine = find_engine_using_name(opt.engine)
    instance = engine(opt)
    print("engine [%s] was created" % type(instance).__name__)
    return instance