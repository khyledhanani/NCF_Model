import torch
import argparse
import logging
import os
from model import NCF
from train import train_ncf, cross_validate
from data import get_data_loader
import torch.multiprocessing as mp
import torch.distributed as dist

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Collaborative Filtering')
    
    # model parameters
    parser.add_argument('--num_users', type=int, default=1000, help='Number of users')
    parser.add_argument('--num_items', type=int, default=2000, help='Number of items')
    parser.add_argument('--embedding_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--layers', nargs='+', type=int, default=[64, 32, 16, 8], help='MLP layers')
    
    # training parameters
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    # cross-validation parameters
    parser.add_argument('--cross_validate', action='store_true', help='Whether to perform cross-validation')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of cross-validation folds')
    
    # distributed training parameters - gcp online
    parser.add_argument('--distributed', action='store_true', help='Whether to use distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='Distributed backend')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    
    return parser.parse_args()

def setup_distributed(rank, world_size, dist_backend, dist_url):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

def cleanup_distributed():
    dist.destroy_process_group()

def main(rank, world_size, args):
    logging.basicConfig(
        level=logging.INFO if rank in [-1, 0] else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    if rank in [-1, 0]:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if args.distributed:
        setup_distributed(rank, world_size, args.dist_backend, args.dist_url)
        logger.info(f'Initialized distributed training on rank {rank}')
    
    logger.info('Initializing the NCF model.')
    
    if args.cross_validate and not args.distributed:
        logger.info('Starting cross-validation.')
        dataset = get_data_loader(
            num_users=args.num_users,
            num_items=args.num_items,
            batch_size=args.batch_size,
            return_dataset=True
        )
        
        results = cross_validate(
            model_class=NCF,
            dataset=dataset,
            n_splits=args.n_splits,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            logger=logger,
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            layers=args.layers
        )
        
        logger.info('Cross-validation Results:')
        for metric, values in results.items():
            logger.info(f'{metric}: {values["mean"]:.4f} ± {values["std"]:.4f}')
    
    else:
        model = NCF(
            num_users=args.num_users,
            num_items=args.num_items,
            embedding_dim=args.embedding_dim,
            layers=args.layers
        )
        
        train_loader, val_loader = get_data_loader(
            num_users=args.num_users,
            num_items=args.num_items,
            batch_size=args.batch_size,
            num_samples=100000
        )
        
        logger.info('Starting training.')
        train_ncf(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=args.device,
            logger=logger,
            checkpoint_dir=args.checkpoint_dir,
            distributed=args.distributed,
            local_rank=rank if args.distributed else -1
        )
        logger.info('Training completed.')
    
    if args.distributed:
        cleanup_distributed()

if __name__ == "__main__":
    args = parse_args()
    
    if args.distributed:
        mp.spawn(
            main,
            args=(args.world_size, args),
            nprocs=args.world_size,
            join=True
        )
    else:
        main(-1, 1, args)
