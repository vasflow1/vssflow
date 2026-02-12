import argparse, os, glob, importlib, wandb
from omegaconf import OmegaConf

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import Callback


def get_func_from_str(string):
    package_name, function_name = string.rsplit(".", 1)
    try:
        package = importlib.import_module(package_name)
        function = getattr(package, function_name)
        return function
    except ImportError:
        print(f"Package {package_name} not found.")
    except AttributeError:
        print(f"Function {function_name} not found in package {package_name}.")


def get_obj_from_str(string, reload=False):
    module, obj_class = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), obj_class)

def get_class_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-n",   "--name",       type=str,       nargs="?",  const=True,     default="",     help="postfix for logdir",)
    parser.add_argument("-r",   "--resume",     type=str,       nargs="?",  const=True,     default="",     help="resume from logdir or checkpoint in logdir",)
    parser.add_argument("-p",   "--project",    type=str,       nargs="?",                  default="",     help="name of new or path to existing project")
    parser.add_argument("-f",   "--postfix",    type=str,       nargs="?",                  default="",     help="post-postfix for default name",)
    parser.add_argument("-t",   "--train",      type=str2bool,  nargs="?",  const=True,     default=False,  help="train",)
    parser.add_argument("-i",   "--infer",      type=str2bool,  nargs="?",  const=True,     default=False,  help="infer",)
    parser.add_argument("-nt",  "--no-test",    type=str2bool,  nargs="?",  const=True,     default=False,  help="disable test",)
    parser.add_argument("-d",   "--debug",      type=str2bool,  nargs="?",  const=True,     default=False,  help="enable post-mortem debugging",)
    parser.add_argument("-c",   "--custom_train",type=str2bool, nargs="?",  const=True,     default=False,  help="enable custom training",)
    parser.add_argument("-s",   "--seed",       type=int,       nargs="?",                  default=42,     help="seed for seed_everything",)
    parser.add_argument("-b",   "--base",       nargs="*",      metavar="base_config.yaml", default=list(),
                        help="Paths to base configs. Loaded from left-to-right. "
                        "Parameters can be overwritten or added with command line options of the form `--key_value`.",)
    return parser


def prepare_args(now, args):
    """
    resume:
        args.resume_from_checkpoint : /path/log/{setting_name}/ckpt/{ckpt_name}.ckpt
        args.base                   : sorted(/path/log/{setting_name}/config/*.yaml) + +args.base
    retrun: 
        nowname: resume     : setting_name (/path/log/{setting_name})
                 not resume : {now}-{args.name OR args.base[0] cfg_name}{args.postfix}
        log_dir: resume     : /path/log/{setting_name}
                 not resume : /path/log/nowname
    """
    if args.name and args.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if args.resume:
        if not os.path.exists(args.resume):
            raise ValueError("Cannot find {}".format(args.resume))
        if os.path.isfile(args.resume):
            paths = args.resume.split("/")
            # MODIFIED
            try: idx = len(paths)-paths[::-1].index("log")+1
            except: idx = len(paths)-paths[::-1].index("log_backup")+1
            log_dir = "/".join(paths[:idx])
            ckpt = args.resume
        else:
            assert os.path.isdir(args.resume), args.resume
            log_dir = args.resume.rstrip("/")
            ckpt = os.path.join(log_dir, "ckpt", "last.ckpt")
        args.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(log_dir, "config/*.yaml")))
        args.base = base_configs+args.base          # resume config <= args.base config 
        _tmp = log_dir.split("/")
        # MODIFIED
        try: nowname = _tmp[_tmp.index("log")+1]
        except: nowname = _tmp[_tmp.index("log_backup")+1]
    else:
        if args.name:
            name = "-"+args.name
        elif args.base:
            cfg_fname = os.path.split(args.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "-"+cfg_name
        else:
            name = ""
        nowname = now+name+args.postfix
        log_dir = os.path.join("log", nowname)
    
    return nowname, log_dir


def prepare_config(args, unknown, nowname, log_dir):
    ckpt_dir = os.path.join(log_dir, "ckpt")
    cfg_dir = os.path.join(log_dir, "config")
    
    # merge: merge configs from args.base and cli to config
    configs = [OmegaConf.load(cfg) for cfg in args.base]
    cli = OmegaConf.from_dotlist(unknown)
    print(f"=> Command line arguments: {cli}")
    configs = configs + [cli]
    config = OmegaConf.merge(*configs)
    # pop: pop lightning_config
    lightning_config = config.pop("lightning", OmegaConf.create())

    # set: set trainer_config: merge args to config, set devices
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    for k in nondefault_trainer_args(args):
        trainer_config[k] = getattr(args, k)
    if trainer_config["devices"]:
        # regarded as gpu (not cpu tpu ipu) enviroment by default
        # with ddp strategy and devices gpus
        trainer_config["accelerator"] = "gpu"
        parsed_devices = pl.accelerators.CUDAAccelerator.parse_devices(trainer_config["devices"])
        assert trainer_config["strategy"] in ["ddp", "ddp_find_unused_parameters_false", ], "Only support ddp strategy for now"
        print(f"=> Running on GPU devices {parsed_devices}")
    else:
        # regarded as cpu enviroment by default
        trainer_config["accelerator"] = "cpu"
        trainer_config["strategy"] = None
        print(f"=> Running on CPU")
    trainer_arg = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    trainer_kwargs = dict()     # for logger and callbacks

    # logger
    default_logger_cfgs = {
        "wandb": {
            "target": "pytorch_lightning.loggers.wandb.WandbLogger",
            "params": {
                "dir"       : os.path.abspath(log_dir),
                "save_dir"  : os.path.abspath(log_dir),
                "id"        : nowname,
                "offline"   : args.debug,
                # "config"    : OmegaConf.to_container(cfg, resolve=True),
                # "resume"    : cfg.wandb.resume,         # TODO
            }
        },
        "csv": {
            "target": "pytorch_lightning.loggers.csv_logs.CSVLogger",
            "params": {
                "name"      : "csvlogger",
                "save_dir"  : log_dir,
            }
        },
    }
    _tmp = lightning_config["logger"]["target"].split(".")
    default_logger_choice = _tmp[_tmp.index("loggers")+1].lower().split("_")[0]
    default_logger_cfg = default_logger_cfgs[default_logger_choice]
    logger_cfg = lightning_config.get("logger", OmegaConf.create())
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    # WandB==0.13.5 do not makedirs for "dir" 
    if default_logger_choice == "wandb":
        os.makedirs(logger_cfg["params"]["dir"], exist_ok=True) 
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # add callback which sets up log directory
    default_callbacks_cfg = {
        "setup_callback": {
            "target": "worker.base.SetupCallback",
            "params": {
                "resume"    : args.resume,
                "nowname"   : nowname,
                "log_dir"   : log_dir,
                "ckpt_dir"  : ckpt_dir,
                "cfg_dir"   : cfg_dir,
                "config"    : config,
                "custom_train"      : args.custom_train,
                "lightning_config"  : lightning_config,
            }
        },
        "modelcheckpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath"   : ckpt_dir,
                "filename"  : "{epoch:04}-{step:.2e}",
            }
        },
    }
    callbacks_cfg = lightning_config.get("callbacks", OmegaConf.create())
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    return config, lightning_config, trainer_arg, trainer_kwargs


class SetupCallback(Callback):
    def __init__(self, resume, nowname, log_dir, ckpt_dir, cfg_dir, config, lightning_config, custom_train=False):
        super().__init__()
        self.resume = resume
        self.nowname = nowname
        self.log_dir = log_dir
        self.ckpt_dir = ckpt_dir
        self.cfg_dir = cfg_dir
        self.config = config
        self.lightning_config = lightning_config
        
        if custom_train:
            # Create logdirs and save configs
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.cfg_dir, exist_ok=True)

            print("=> Project config")
            print(OmegaConf.to_yaml(self.config, resolve=False, sort_keys=False))
            OmegaConf.save(self.config,
                           os.path.join(self.cfg_dir, "{}-project.yaml".format(self.nowname)))

            print("=> Lightning config")
            print(OmegaConf.to_yaml(self.config, resolve=False, sort_keys=False))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfg_dir, "{}-lightning.yaml".format(self.nowname)))

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.cfg_dir, exist_ok=True)
            
            if not self.resume:
                print("=> Project config")
                print(OmegaConf.to_yaml(self.config, resolve=False, sort_keys=False))
                OmegaConf.save(self.config,
                            os.path.join(self.cfg_dir, "{}-project.yaml".format(self.nowname)))

                print("=> Lightning config")
                print(OmegaConf.to_yaml(self.config, resolve=False, sort_keys=False))
                OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                            os.path.join(self.cfg_dir, "{}-lightning.yaml".format(self.nowname)))
        else:
            pass


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, persistent_workers=False, prefetch_factor=2, num_workers=None, collate_fn=None,
                 train=None, validation=None, test=None, data_name_to_cfg={}):
        super().__init__()
        self.batch_size         = batch_size
        self.dataset_configs    = dict()
        self.dataset_names      = dict()
        self.persistent_workers = persistent_workers
        self.prefetch_factor    = prefetch_factor
        self.num_workers        = num_workers if num_workers is not None else batch_size*2
        self.data_name_to_cfg   = data_name_to_cfg
        self.collate_fn         = None if collate_fn is None else get_func_from_str(collate_fn)
        if train is not None:
            assert train in self.data_name_to_cfg, "Training dataset name does not match with registered datasets."
            self.dataset_configs["train"]   = self.data_name_to_cfg[train]["train"]
            self.dataset_names["train"]     = train
            self.train_dataloader           = self._train_dataloader
        if validation is not None:
            assert validation in self.data_name_to_cfg, "Validation dataset name does not match with registered datasets."
            self.dataset_configs["validation"]  = self.data_name_to_cfg[validation]["validation"]
            self.dataset_names["validation"]    = validation
            self.val_dataloader                 = self._val_dataloader
        if test is not None:
            assert test in self.data_name_to_cfg, "Test dataset name does not match with registered datasets."
            self.dataset_configs["test"]    = self.data_name_to_cfg[test]["test"]
            self.dataset_names["test"]      = test
            self.test_dataloader            = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict( (k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def _train_dataloader(self):
        if self.dataset_names["train"] in ["vggass", "asm-", "hits", "favd", "tavg", "avsync15", "vggsound", "vggss", "greatesthits", "landscape", "lipphone"]:
            return DataLoader(self.datasets["train"], batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn,
                              persistent_workers=self.persistent_workers, num_workers=self.num_workers, 
                              prefetch_factor=self.prefetch_factor,)
        else:
            raise ValueError(f"Training dataset does not support {self.dataset_names['train']}")
        
    def _val_dataloader(self):
        if self.dataset_names["validation"] in ["vggass", "asm-", "favd", "tavg", "avsync15", "vggsound", "vggss", "greatesthits", "landscape", "lipphone"]:
            # return DataLoader(self.datasets["validation"], batch_size=self.batch_size, shuffle=False, collate_fn=None,
            return DataLoader(self.datasets["validation"], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                              persistent_workers=self.persistent_workers, num_workers=self.num_workers,
                              prefetch_factor=self.prefetch_factor,)
        else:
            raise ValueError(f"Validation dataset does not support {self.dataset_names['validation']}")
        
    def _test_dataloader(self):
        if self.dataset_names["test"] in ["vggass", "asm-", "favd", "tavg", "avsync15", "vggsound", "vggss", "greatesthits", "landscape", "lipphone"]:
            return DataLoader(self.datasets["test"], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn,
                              persistent_workers=self.persistent_workers, num_workers=self.num_workers,
                              prefetch_factor=self.prefetch_factor,)
        else:
            raise ValueError(f"Test dataset does not support {self.dataset_names['test']}")
        
        
if __name__=="__main__":
    pass