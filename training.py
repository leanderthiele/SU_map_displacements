import numpy as np

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as torch_mp

import settings
import startup
from data_loader import DataModes, DataLoader
from network import Network
from train_utils import Loss, Optimizer

def do_diagnostic_output(training_loss, validation_loss, Nepochs, epoch_len, world_size) :
    # we do not make assumptions about the length or data type of the *_loss arrays,
    # we only need to know that we can subscript them to get a float
#{{{
    training_times = np.linspace(0, Nepochs, num=Nepochs*epoch_len).repeat(world_size)
    validation_times = np.linspace(1, Nepochs, num=Nepochs).repeat(world_size)

    training_loss_arr = np.empty(len(training_times))
    validation_loss_arr = np.empty(len(validation_times))
    for ii in range(len(training_loss_arr)) :
        training_loss_arr[ii] = training_loss[ii]
    for ii in range(len(validation_loss_arr)) :
        validation_loss_arr[ii] = validation_loss[ii]

    np.savez(settings.LOSS_FILE, training_times=training_times, validation_times=validation_times,
                                 training_loss=training_loss_arr, validation_loss=validation_loss_arr)
#}}}

def save_model(model) :
    # TODO
#{{{
    pass
#}}}

def training_process(rank, world_size, training_loss, validation_loss) :
    """
    A single training process, working on its own data.

    training_loss and validation_loss are in shared memory,
    the individual processes write into separate locations according to their rank.
    The root process is responsible for keeping us updated about training progress
    by periodically giving some loss output.
    """
#{{{
    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    model = Network(**settings.NETWORK_ARGS).to(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    
    # TODO write Loss function and Optimizer
    loss_fn = Loss()
    optimizer = Optimizer()

    training_loader = DataLoader(DataModes.TRAINING, rank, world_size)
    validation_loader = DataLoader(DataModes.VALIDATION, rank, world_size)

    for epoch in range(settings.EPOCHS) :

        # set model into training mode
        ddp_model.train()

        # loop once through the training data
        for t, data in enumerate(training_loader) :
            
            # send the batch to this device
            data.to(rank)

            # do the forward pass and compute loss
            optimizer.zero_grad()
            prediction = ddp_model(data.inputs, data.styles)
            this_training_loss = loss_fn(prediction, data.outputs)

            # update the loss storage
            training_loss[epoch*world_size*len(training_loader) + t*world_size + rank] \
                = this_training_loss.item()

            # update weights -- this is an implicit synchronization point!
            this_training_loss.backward()
            optimizer.step()

        # set model into evaluation mode
        ddp_model.eval()
        
        # loop once through the validation data
        this_validation_loss = 0.0
        with torch.no_grad() :
            for t, data in enumerate(validation_loader) :
                
                # send the batch to this device
                data.to(rank)

                prediction = ddp_model(data.inputs, data.styles)
                this_validation_loss += loss_fn(prediction, data.outputs).item()

        validation_loss[epoch*world_size + rank] = this_validation_loss / len(validation_loader)

        # now give diagnostic output -- we need to wait for all threads to reach this point so
        # we have the validation loss complete
        idx = settings.DIAGNOSTIC_BARRIER.wait()
        if idx == 0 :
            do_diagnostic_output(training_loss, validation_loss, epoch+1, epoch_len, world_size)
        if (world_size == 1 and idx == 0) or idx == 1 :
            save_model(ddp_model)
#}}}

def main() :
    """
    launches a couple of training_process's
    """
#{{{
    world_size = settings.NUM_GPU

    training_loss = torch_mp.Array('d', settings.EPOCHS * 100 * 48)
    validation_loss = torch_mp.Array('d', settings.EPOCHS * world_size)

    torch_mp.spawn(training_process,
                   args=(world_size, training_loss, validation_loss, ),
                   nprocs=world_size)
#}}}

if __name__ == '__main__' :
    startup.main()
    main()
