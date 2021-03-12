import os
from time import time

import numpy as np

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_mp

import settings
from data_loader import DataModes, Batch, DataLoader
from network import Network
from train_utils import Loss, Optimizer

def do_diagnostic_output(training_loss, validation_loss, Nepochs, epoch_len, world_size) :
    # we do not make assumptions about the length or data type of the *_loss arrays,
    # we only need to know that we can subscript them to get a float
#{{{
    training_times = np.linspace(0, Nepochs, num=Nepochs*epoch_len).repeat(world_size)
    validation_times = np.linspace(1, Nepochs, num=Nepochs).repeat(world_size)

    np.savez(settings.LOSS_FILE, training_times=training_times, validation_times=validation_times,
                                 training_loss=np.array(training_loss),
                                 validation_loss=np.array(validation_loss_arr))
#}}}

def save_model(model) :
    # TODO
#{{{
    pass
#}}}

def setup_process(rank, world_size) :
    # to be called at the beginning of a child process
#{{{
    # these are taken from the example at https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)
#}}}

def cleanup_process() :
    # to be called at the end of a child process
#{{{
    torch.distributed.destroy_process_group()
#}}}

def training_process(rank, world_size, diagnostic_barrier, diagnostic_pipes) :
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

    if rank == 0 :
        # root holds the data for all epochs (so we can easily produce plots etc.)
        all_training_loss = []
        all_validation_loss = []

    for epoch in range(settings.EPOCHS) :
        
        if rank == 0 :
            start_time_epoch = time()
        
        # create the variables we will later share with the root thread
        training_loss = []
        validation_loss = 0.0

        # set model into training mode
        ddp_model.train()

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
            training_loss.append(this_training_loss.item())

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

        # now give diagnostic output -- we need to wait for all threads to reach this point so
        # we have the validation loss complete
        diagnostic_barrier.wait()

        # non-root processes send their loss statistics to root
        if rank != 0 :
            diagnostic_pipes[rank-1][1].send((training_loss, validation_loss))
        else :
            len_my_training_loss = len(training_loss) # for consistency checks
            training_loss = [training_loss, ] # list of lists
            validation_loss = [validation_loss, ] # list of scalars

            # collect data from the other processes
            for ii in range(world_size-1) :
                t, v = diagnostic_pipes[ii][0].recv()
                assert len(t) == len_my_training_loss
                training_loss.append(t)
                validation_loss.append(v)
            assert len(training_loss) == len(validation_loss) == world_size

            # now write into the long lists (with some reshuffling)
            for ii in range(len_my_training_loss) :
                for jj in range(world_size) :
                    all_training_loss.append(training_loss[jj][ii])
            for ii in range(world_size) :
                all_validation_loss.append(validation_loss[ii])

            do_diagnostic_output(training_loss, validation_loss, epoch+1, len(training_loader), world_size)

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

    # create some objects that we need to share between processes
    diagnostic_barrier = torch_mp.Barrier(world_size)
    diagnostic_pipes = [torch_mp.Pipe(False) for ii in range(world_size-1)]

    torch_mp.spawn(training_process,
                   args=(world_size, diagnostic_barrier, diagnostic_pipes, ),
                   nprocs=world_size)
#}}}

if __name__ == '__main__' :
    main()
