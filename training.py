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
import train_utils
from train_utils import Loss, Optimizer



def training_process(rank) :
    """
    A single training process, working on its own data.

    training_loss and validation_loss are in shared memory,
    the individual processes write into separate locations according to their rank.
    The root process is responsible for keeping us updated about training progress
    by periodically giving some loss output.
    """
#{{{
    train_util.setup_process(rank)

    model = Network().to(settings.DEVICE_IDX).to_ddp()

    train_utils.load_model(model)
    
    loss_fn = Loss()
    optimizer = Optimizer(model.parameters())

    # reset the optimizer -- not sure if it is necessary here but can't hurt
    optimizer.zero_grad()

    training_loader = DataLoader(DataModes.TRAINING)
    validation_loader = DataLoader(DataModes.VALIDATION)

    # load previous loss if it exists
    if train_utils.is_output_responsible() :
        start_epoch, all_epochs_training_loss, all_epochs_validation_loss = train_utils.load_loss()
        start_epoch_list = [start_epoch, ]
    else :
        start_epoch_list = [-1, ]

    # tell the other processes which epoch they should start training on
    torch.distributed.broadcast_object_list(start_epoch_list, src=settings.RANK)
    
    if train_utils.is_output_responsible() :
        assert start_epoch_list[0] == start_epoch
    else :
        start_epoch == start_epoch_list[0]


    # keep track of whether we encounter infinities / nans
    global_inf = False
    inf_list = [False, ] * settings.WORLD_SIZE

    # initialize outside the epoch loop so we can carry samples over from one
    # epoch to the other if a batch is incomplete
    idx_in_batch = 0
        

    for epoch in range(start_epoch, start_epoch + settings.EPOCHS) :
        
        if settings.RANK == 0 :
            start_time_epoch = time()

        # create the variables we will later share with the root thread
        # note that it is useful to set the training loss to an impossible state initially
        # because we can catch bugs more easily
        training_loss = np.full(len(training_loader), -1.0)
        validation_loss = 0.0

        # set model into training mode
        model.train()

        # loop once through the training data
        for t, data in enumerate(training_loader) :
            
            if settings.RANK == 0 :
                start_time_sample = time()

            idx_in_batch += 1
            
            assert isinstance(data, Batch)

            inputs, targets, styles = data.get_on_device()

            # do the forward pass and compute loss

            # note the >, it guards us against the possibility that a
            # full batch and an invalid loss coincide
            batch_done = idx_in_batch > settings.BATCH_SIZE

            prediction = model(inputs, styles)

            this_training_loss = loss_fn(prediction, targets)

            # update the loss storage
            training_loss[t] = this_training_loss.item()

            # check whether we need to disable synchronization because
            # at least one process had a problem
            this_inf = not np.isfinite(this_training_loss.item())
            torch.distributed.all_gather_object(inf_list, this_inf)
            global_inf = any(inf_list)

            # we can do a synchronous weight update if both the batch is full
            # and no invalid loss value has been found
            should_update_weights = batch_done and not global_inf

            if should_update_weights :
                # update gradients synchronously -- this is an implicit synchronization point!
                this_training_loss.backward()
            else :
                if this_inf :
                    # we encountered infinity/nan and should throw this loss away
                    print(f'encountered invalid loss in rank {settings.RANK}')
                else :
                    # update gradients asynchronously in those processes that did not have a problem
                    with model.no_sync() :
                        this_training_loss.backward()

            if should_update_weights :
                # each process sees the same gradients and can therefore do the same weight updates
                optimizer.step()
                optimizer.zero_grad()
                idx_in_batch = 0
            
            if settings.RANK == 0 and settings.VERBOSE :
                print('\tSample %.3d / %d finished in epoch %d, '\
                      'took %f seconds'%(t+1, len(training_loader), epoch+1, time()-start_time_sample))

        # set model into evaluation mode
        model.eval()

        if settings.RANK == 0 :
            start_time_validation = time()
        
        # loop once through the validation data
        with torch.no_grad() :
            for t, data in enumerate(validation_loader) :

                assert isinstance(data, Batch)

                inputs, targets, styles = data.get_on_device()

                prediction = model(inputs, styles)
                validation_loss += loss_fn(prediction, targets).item()

        # normalize (per data item)
        validation_loss /= len(validation_loader)

        if settings.RANK == 0 and settings.VERBOSE :
            print('\tLoop through validation set took %f seconds'%(time()-start_time_validation))

        if train_utils.is_output_responsible() :
            start_time_diagnostic = time()

        # buffers for gathering
        all_training_loss = [np.empty(0), ] * settings.WORLD_SIZE
        all_validation_loss = [0.0, ] * settings.WORLD_SIZE

        # gather the loss values from all processes
        # note that only the is_output_responsible() process actually needs them,
        # but gather_object is not supported when using NCCL
        torch.distributed.all_gather_object(all_training_loss, training_loss)
        torch.distributed.all_gather_object(all_validation_loss, validation_loss)


        if train_utils.is_output_responsible() :
            # interleave the training loss arrays so the losses are temporally correctly ordered
            all_training_loss = np.vstack(all_training_loss).reshape((-1,), order='F')
            all_validation_loss = np.array(all_validation_loss)

            all_epochs_training_loss = np.concatenate((all_epochs_training_loss,
                                                       all_training_loss))
            all_epochs_validation_loss = np.concatenate((all_epochs_validation_loss,
                                                         all_validation_loss))

            train_utils.do_diagnostic_output(all_epochs_training_loss, all_epochs_validation_loss,
                                             epoch+1, len(training_loader))

            train_util.save_model(model)


        if train_utils.is_output_responsible() and settings.VERBOSE :
            print('\tGathering of losses and diagnostic output took %f seconds'%(time()-start_time_diagnostic))


        if settings.RANK == 0 :
            print('Epoch %d finished, took %f seconds'%(epoch+1, time()-start_time_epoch))

    # we're done, let's release resources
    train_utils.cleanup_process()
#}}}


def main() :
    """
    launches a couple of training_process's
    (or one if each MPI_RANK gets its own GPU)
    """
#{{{
    startup.main(DataModes.TRAINING)

    if settings.MPI_ENV_TYPE is settings.MPIEnvTypes.MULTIGPU_SINGLERANK :
        torch_mp.spawn(training_process,
                       nprocs=settings.VISIBLE_GPUS)
    else :
        training_process(0)
#}}}

if __name__ == '__main__' :
    main()
