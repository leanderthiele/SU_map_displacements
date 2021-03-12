import os
from time import time

import numpy as np

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_mp

import settings
import startup
from data_loader import DataModes, Batch, DataLoader
from network import Network
from train_utils import Loss, Optimizer

def do_diagnostic_output(training_loss, validation_loss, Nepochs, epoch_len, world_size) :
#{{{
    training_times = np.linspace(0, Nepochs, num=Nepochs*epoch_len).repeat(world_size)
    validation_times = np.linspace(1, Nepochs, num=Nepochs).repeat(world_size)

    np.savez(settings.LOSS_FILE, training_times=training_times, validation_times=validation_times,
                                 training_loss=training_loss, validation_loss=validation_loss)
#}}}

def save_model(model) :
    # this function will only be called from the rank=0 process
#{{{
    torch.save(model.state_dict(), settings.MODEL_FILE)
#}}}

def load_model(model, rank) :
    # this function will be called from any process, we need to make sure we map
    # the tensors to the correct devices
    # TODO we probably want to store other data as well, most importantly the optimizer state dict
    #      other things we can put in are the loss curves
#{{{
    map_location = { 'cuda:0' : 'cuda:%d'%rank }
    model.load_state_dict(torch.load(settings.MODEL_FILE, map_location=map_location))
#}}}

def setup_process(rank, world_size) :
    # to be called at the beginning of a child process
#{{{
    # new process needs to get a consistent view of the settings
    startup.main(DataModes.TRAINING)

    # these are taken from the example at https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    # note that we only call this because the documentation for all_gather_object says we need to
    # when using NCCL
    torch.cuda.set_device(rank)
#}}}

def cleanup_process() :
    # to be called at the end of a child process
#{{{
    torch.distributed.destroy_process_group()
#}}}

def training_process(rank, world_size) :
    """
    A single training process, working on its own data.

    training_loss and validation_loss are in shared memory,
    the individual processes write into separate locations according to their rank.
    The root process is responsible for keeping us updated about training progress
    by periodically giving some loss output.
    """
#{{{
    setup_process(rank, world_size)

    model = Network().to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    
    loss_fn = Loss()
    optimizer = Optimizer(ddp_model.parameters())

    training_loader = DataLoader(DataModes.TRAINING, rank, world_size)
    validation_loader = DataLoader(DataModes.VALIDATION, rank, world_size)

    for epoch in range(settings.EPOCHS) :
        
        if rank == 0 :
            start_time_epoch = time()
        
        # create the variables we will later share with the root thread
        # note that it is useful to set the training loss to an impossible state initially
        # because we can catch bugs more easily
        training_loss = np.full(len(training_loader), -1.0)
        validation_loss = 0.0

        # set model into training mode
        ddp_model.train()

        # we need to reseed the global random number generator
        # so we get different augmentations each epoch
        np.random.seed()

        # loop once through the training data
        for t, data in enumerate(training_loader) :
            
            if rank == 0 :
                start_time_sample = time()
            
            assert isinstance(data, Batch)

            # do the forward pass and compute loss
            optimizer.zero_grad()

            prediction = ddp_model(data.inputs, data.styles)

            this_training_loss = loss_fn(prediction, data.targets)

            # update the loss storage
            training_loss[t] = this_training_loss.item()

            # update weights -- this is an implicit synchronization point!
            this_training_loss.backward()

            optimizer.step()
            
            if rank == 0 :
                print('\tSample %.3d / %d finished in epoch %d, '\
                      'took %f seconds'%(t+1, len(training_loader), epoch+1, time()-start_time_sample))

        # set model into evaluation mode
        ddp_model.eval()
        
        # loop once through the validation data
        with torch.no_grad() :
            for t, data in enumerate(validation_loader) :

                assert isinstance(data, Batch)

                prediction = ddp_model(data.inputs, data.styles)
                validation_loss += loss_fn(prediction, data.targets).item()

        # normalize (per data item)
        validation_loss /= len(validation_loader)

        # buffers for gathering
        all_training_loss = [np.empty(0), ] * world_size
        all_validation_loss = [0.0, ] * world_size

        # gather the loss values from all processes
        # note that only the rank=0 process actually needs them, but gather_object is not supported
        # when using NCCL
        torch.distributed.all_gather_object(all_training_loss, training_loss)
        torch.distributed.all_gather_object(all_validation_loss, validation_loss)

        if rank == 0 :
            # interleave the training loss arrays so the losses are temporally correctly ordered
            all_training_loss = np.vstack(all_training_loss).reshape((-1,), order='F')
            all_validation_loss = np.array(all_validation_loss)
            do_diagnostic_output(all_training_loss, all_validation_loss,
                                 epoch+1, len(training_loader), world_size)

        if (world_size == 1 and rank == 0) or rank == 1 :
            save_model(ddp_model)

        if rank == 0 :
            print('Epoch %d finished, took %f seconds'%(epoch+1, time()-start_time_epoch))

    # we're done, let's release resources
    cleanup_process()
#}}}

def main() :
    """
    launches a couple of training_process's
    """
#{{{

    world_size = torch.cuda.device_count()

    torch_mp.spawn(training_process,
                   args=(world_size, ),
                   nprocs=world_size)
#}}}

if __name__ == '__main__' :
    startup.main(DataModes.TRAINING)
    main()
