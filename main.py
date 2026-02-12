import os, sys, wandb
from datetime import datetime, timezone, timedelta

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.tuner.tuning import Tuner

from worker.base import get_parser, prepare_args, prepare_config, instantiate_from_config, get_func_from_str




def main():
    now =  datetime.now(timezone(timedelta(hours=+8), 'BJ')).strftime("%Y_%m_%d-%H_%M_%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args, unknown = parser.parse_known_args()
    if args.debug:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    nowname, log_dir = prepare_args(now=now, args=args)
    config, lightning_config, trainer_arg, trainer_kwargs = prepare_config(args, unknown, nowname, log_dir)
    
    # For h100 tensor core TODO try different precision for different project
    torch.set_float32_matmul_precision(config.platform.matmul_precision)

    seed_everything(args.seed)

    if args.infer:
        model = instantiate_from_config(config.model)
        trainer = Trainer.from_argparse_args(trainer_arg, **trainer_kwargs)
        dataset = instantiate_from_config(config.infer_data)
        collate_fn = get_func_from_str(config.infer_data.collate_fn) if config.infer_data.collate_fn is not None else None
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=config.infer_data.batch_size, 
                                                 shuffle=False, 
                                                 num_workers=config.infer_data.num_workers, 
                                                 prefetch_factor=config.infer_data.prefetch_factor,
                                                 persistent_workers=config.infer_data.persistent_workers,
                                                 collate_fn=collate_fn)
        trainer.predict(model=model, dataloaders=dataloader, return_predictions=False, ckpt_path=config.infer_data.ckpt_path)
        return

    try:
        # Model
        model = instantiate_from_config(config.model)
        trainer = Trainer.from_argparse_args(trainer_arg, **trainer_kwargs)
        data = instantiate_from_config(config.data)
        # data.prepare_data()
        # data.setup()
        
        # Auto scale batch size 
        if trainer_arg.auto_scale_batch_size:
            print("=> Find batch size ...")
            if trainer_arg.strategy.startswith("ddp"):
                print("   Not support for ddp-like strategy for now pl version")
            else:
                tuner = Tuner(trainer)
                new_batch_size = tuner.scale_batch_size(model=model, datamodule=data, method='fit', mode=trainer_arg.auto_scale_batch_size)
                data.batch_size = new_batch_size
                print("   New batch size {}".format(new_batch_size))

        # Configure learning rate
        bs, base_lr = data.batch_size, config.model.base_learning_rate
        if not lightning_config.trainer["accelerator"]=="cpu":
            ngpu = len(pl.accelerators.CUDAAccelerator.parse_devices(lightning_config.trainer.devices))
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("=> Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
              model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))

        # Allow checkpointing via USR1
        def melk(*args, **kwargs):
            # Run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(f"{log_dir}/ckpt", "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()
        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # NOTE TODO For resume training, temporarily use following code 
        # try:
        #     trainer.fit(model, data)
        # except Exception:
        #     melk()
        #     raise
        
        # Run
        if args.train:
            try:
                if args.custom_train:
                    raise NotImplementedError("Not support custom training for now")
                else:
                    trainer.fit(model, data)
            except Exception:
                if args.custom_train:
                    pass
                else:
                    melk()
                raise
        
        # if not args.no_test and not trainer.interrupted:
        #     trainer.test(model, data)
        else:
            trainer.validate(model, data)
    except Exception:
        raise
    finally:
        pass


if __name__ == "__main__":
    main()
    sys.exit(0)