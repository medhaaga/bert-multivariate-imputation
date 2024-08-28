# System & OS
import os
import sys
import argparse
import json

sys.path.append('../')
sys.path.append('../../')

import numpy as np
from tqdm import tqdm

# Torch
import torch
from torch.utils.data import DataLoader

# Script imports

from src.transformer_utils import (collate_fn,
                                MLMBERT)
from src.plot_utils import (plot_imputed_acc_data,
                            plot_missing_acc_data)
from src.os_utils import (get_results_path)
from src.data_utils import create_dataset
##############################################
# Arguments
##############################################

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=168)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--ratio", type=float, default=0.1)
    parser.add_argument("--mask_avg_len", type=int, default=2)
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=.0001)
    parser.add_argument("--weight_decay", type=float, default=.0001)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--static", type=int, default=0)


    return parser


def train(model, train_dataloader, val_dataloader, test_dataloader, device, config, args):

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=1e-4, steps_per_epoch=len(train_dataloader), epochs=args.num_epochs+1)
    mse_loss = torch.nn.MSELoss()

    train_losses = []
    valid_losses = []
    test_losses = []
    best_val_loss = 1e9


    for epoch in tqdm(range(args.num_epochs)):

        model.train()
        trl = 0
        tprog = enumerate(train_dataloader)

        for i, (batch, batch_mask, batch_static) in tprog:
  
            batch = batch.to(device)
            masked_batch = batch.clone()
            masked_batch[batch_mask] = config['mask_token_id']
            masked_batch = masked_batch.to(device)
            batch_static = batch_static.to(device)
    
            preds = model(masked_batch, batch_static)['mask_predictions']
            optim.zero_grad()

            loss = torch.mean(torch.square(preds - batch[batch_mask]))
            # loss = mse_loss(batch[batch_mask], preds)

            # print(f'predictions: {torch.isnan(preds).any().item()}\t, masked batch: {torch.isnan(masked_batch).any().item()},\t target static: {torch.isnan(batch_static).any().item()},\t loss: {loss}')

            loss.backward()

            optim.step()
            sched.step()

            trl += loss.item()
            # print(loss.item(), torch.max(torch.square(batch)).item(), torch.mean(torch.square(preds - batch[batch_mask])).item())

        train_losses.append(trl/len(train_dataloader))

        model.eval()
        with torch.no_grad():
            vrl = 0.
            vprog = enumerate(val_dataloader)
            for i, (batch, batch_mask, batch_static) in vprog:
                batch = batch.to(device)
                masked_batch = batch.clone()
                masked_batch[batch_mask] = config['mask_token_id']
                masked_batch = masked_batch.to(device)
                batch_static = batch_static.to(device)

                preds = model(masked_batch, batch_static)['mask_predictions']
                loss = mse_loss(batch[batch_mask], preds)
                vrl += loss.item()
                # vprog.set_description(f'valid step loss: {loss.item():.4f}')
            vloss = vrl/len(val_dataloader)
            valid_losses.append(vloss)

            
            if vloss < best_val_loss:
                best_val_loss = vloss
                torch.save(model.state_dict(), os.path.join(get_results_path(), 'model.pth'))

        with torch.no_grad():
            trl = 0.
            tprog = enumerate(test_dataloader)
            for i, (batch, batch_mask, batch_static) in tprog:
                batch = batch.to(device)
                masked_batch = batch.clone()
                masked_batch[batch_mask] = config['mask_token_id']
                masked_batch = masked_batch.to(device)
                batch_static = batch_static.to(device)

                preds = model(masked_batch, batch_static)['mask_predictions']
                loss = mse_loss(batch[batch_mask], preds)
                trl += loss.item()

            tloss = trl/len(test_dataloader)
            test_losses.append(tloss)

        if (epoch + 1)%1 == 0:
            print("")
            print(f'EPOCH {epoch+1}:')
            print("----------------------")
            print(f'train loss: {train_losses[-1]:.4f}\nvalidation loss: {valid_losses[-1]:.4f}\ntest loss: {test_losses[-1]:.4f}')

        torch.save(torch.tensor(train_losses), os.path.join(get_results_path(), 'train_losses.pt'))
        torch.save(torch.tensor(valid_losses), os.path.join(get_results_path(), 'valid_losses.pt'))
        torch.save(torch.tensor(test_losses), os.path.join(get_results_path(), 'test_losses.pt'))



def main():

    # parse arguments
    parser = parse_arguments()
    args = parser.parse_args()

    # set device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # set seeds
    np.random.seed(seed=args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # model config and modelx
    config = {
    'dim': args.embed_dim,
    'n_heads': args.n_heads,
    'attn_dropout': args.attn_dropout,
    'mlp_dropout': args.mlp_dropout,
    'depth': args.depth,
    'max_len': args.max_len,
    'pad_token_id': -100,
    'mask_token_id': 100.,
    }


    # collate function for creating dataloaders
    def collate_fn_collapsed(batch):
        return collate_fn(batch, config, args.mask_prob, args.mask_avg_len)

    # set datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_dataset(test_ratio=args.ratio, val_ratio=args.ratio)

    config['vocab_size'] = train_dataset[0][0].shape[-1]
    if args.static:
        config['num_static_features'] = train_dataset[0][1].shape[-1]
    else:
        config['num_static_features'] = 0


    with open(os.path.join(get_results_path(), 'model_config.json'), 'w') as f:
        json.dump(config, f)


    # dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_collapsed)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_collapsed)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_collapsed)
        
    model = MLMBERT(config).to(device)
    print('trainable:', sum([p.numel() for p in model.parameters() if p.requires_grad]))


    # train model
    train(model, train_dataloader, val_dataloader, test_dataloader, device, config, args)


if __name__ == '__main__':
    main()