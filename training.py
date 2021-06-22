"""
This is currently the only executable script within the entire code.
"""


import os
from time import time

import numpy as np

import torch
import torch.distributed
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

    The root process is responsible for keeping us updated about training progress
    by periodically giving some loss output and saving the trained model to disk.
    """
#{{{
    train_utils.setup_process(rank)

    # construct the model, load from disk if exists, and put into DDP mode
    model = Network()   #.sync_batchnorm() TODO at the moment we are not using batch normalization -- this is not a good solution!
    model = model.to(settings.DEVICE_IDX).to_ddp()

    # NOTE it is important that we push the model to the GPU first and establish the DDP stuff,
    #      because otherwise the optimizer may refer to invalid parameters.
    optimizer = Optimizer(model.parameters())

    # if we have a previously trained model on disk, now is the time to load it into memory
    train_utils.load_model(model, optimizer)
    
    loss_fn = Loss()

    training_loader = DataLoader(DataModes.TRAINING)
    validation_loader = DataLoader(DataModes.VALIDATION)

    # reset the optimizer -- not sure if it is necessary here but can't hurt
    optimizer.zero_grad()

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-1,
                                                    steps_per_epoch=len(training_loader),
                                                    epochs=settings.EPOCHS)

    # load previous loss if it exists
    if settings.RANK == 0 :
        start_epoch, all_epochs_training_loss, all_epochs_validation_loss, \
            all_epochs_training_loss_guess, all_epochs_validation_loss_guess, \
            all_epochs_training_loss_guess_rescaled, all_epochs_validation_loss_guess_rescaled \
                = train_utils.load_loss()
        start_epoch_list = [start_epoch, ]
    else :
        start_epoch_list = [-1, ]

    # tell the other processes which epoch they should start training on
    torch.distributed.broadcast_object_list(start_epoch_list, src=0)
    
    if settings.RANK == 0 :
        assert start_epoch_list[0] == start_epoch
    else :
        start_epoch = start_epoch_list[0]

    # keep track of whether we encounter infinities / nans
    # the basic idea here is that if one of the processes computes an invalid loss function,
    # we don't want to have nans propagate into our weight updates.
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
        training_loss_guess = np.full(len(training_loader), -1.0)
        training_loss_guess_rescaled = np.full(len(training_loader), -1.0)
        validation_loss = 0.0
        validation_loss_guess = 0.0
        validation_loss_guess_rescaled = 0.0

        # set model into training mode
        model.train()

        # loop once through the training data
        for t, data in enumerate(training_loader) :
            
            if settings.RANK == 0 :
                start_time_sample = time()

            idx_in_batch += 1

            assert isinstance(data, Batch)

            if False :
                # FIXME for intuition / debugging
                np.savez('test_%d.npz'%settings.RANK,
                         inputs=data.inputs.numpy(), targets=data.targets.numpy(),
                         guesses=data.guesses.numpy(), styles=data.styles.numpy())
                torch.distributed.barrier()
                raise RuntimeError('Wrote data to file.')
            
            inputs, targets, guesses, styles = data.get_on_device()

            # do the forward pass and compute loss

            # note the >, it guards us against the possibility that a
            # full batch and an invalid loss coincide
            batch_done = idx_in_batch > settings.BATCH_SIZE

            # do the forward pass
            guesses_rescaled, prediction = model(inputs, styles, guesses)

            if settings.NORM_LOSS :
                # note the dim kwarg, we do not want to reduce over the batch dimension
                guesses_std = torch.std(guesses - targets, dim=(1,2,3,4))
            else :
                guesses_std = 1.0

            this_training_loss = loss_fn(prediction/guesses_std, targets/guesses_std)
            this_training_loss_guess = loss_fn(guesses/guesses_std, targets/guesses_std)
            this_training_loss_guess_rescaled = loss_fn(guesses_rescaled/guesses_std, targets/guesses_std)

            # update the loss storage
            training_loss[t] = this_training_loss.item()
            training_loss_guess[t] = this_training_loss_guess.item()
            training_loss_guess_rescaled[t] = this_training_loss_guess_rescaled.item()

            # check whether we need to disable synchronization because
            # at least one process had a problem
            this_inf = not np.isfinite(this_training_loss.item())
            torch.distributed.all_gather_object(inf_list, this_inf)
            global_inf = any(inf_list)

            # we can do a synchronous weight update if both the batch is full
            # and no invalid loss value has been found
            should_update_weights = batch_done and not global_inf

            if should_update_weights :
                # do the backward pass
                # updates gradients synchronously -- this is an implicit synchronization point!
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
                scheduler.step()
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

                inputs, targets, guesses, styles = data.get_on_device()

                guesses_rescaled, prediction = model(inputs, styles, guesses)

                if settings.NORM_LOSS :
                    # note the dim kwarg, we do not want to reduce over the batch dimension
                    guesses_std = torch.std(guesses - targets, dim=(1,2,3,4))
                else :
                    guesses_std = 1.0

                validation_loss += loss_fn(prediction/guesses_std, targets/guesses_std).item()
                validation_loss_guess += loss_fn(guesses/guesses_std, targets/guesses_std).item()
                validation_loss_guess_rescaled += loss_fn(guesses_rescaled/guesses_std, targets/guesses_std).item()

        # normalize (per data item)
        validation_loss /= len(validation_loader)
        validation_loss_guess /= len(validation_loader)
        validation_loss_guess_rescaled /= len(validation_loader)

        if settings.RANK == 0 and settings.VERBOSE :
            print('\tLoop through validation set took %f seconds'%(time()-start_time_validation))

        if settings.RANK == 0 :
            start_time_diagnostic = time()

        # buffers for gathering
        all_training_loss = [np.empty(0), ] * settings.WORLD_SIZE
        all_training_loss_guess = [np.empty(0), ] * settings.WORLD_SIZE
        all_training_loss_guess_rescaled = [np.empty(0), ] * settings.WORLD_SIZE
        all_validation_loss = [0.0, ] * settings.WORLD_SIZE
        all_validation_loss_guess = [0.0, ] * settings.WORLD_SIZE
        all_validation_loss_guess_rescaled = [0.0, ] * settings.WORLD_SIZE

        # gather the loss values from all processes
        # note that only the 0th process actually needs them,
        # but gather_object is not supported when using NCCL
        torch.distributed.all_gather_object(all_training_loss, training_loss)
        torch.distributed.all_gather_object(all_training_loss_guess, training_loss_guess)
        torch.distributed.all_gather_object(all_training_loss_guess_rescaled, training_loss_guess)
        torch.distributed.all_gather_object(all_validation_loss, validation_loss)
        torch.distributed.all_gather_object(all_validation_loss_guess, validation_loss_guess)
        torch.distributed.all_gather_object(all_validation_loss_guess_rescaled, validation_loss_guess)


        if settings.RANK == 0 :
            # interleave the training loss arrays so the losses are temporally correctly ordered
            all_training_loss = np.sqrt(np.vstack(all_training_loss).reshape((-1,), order='F'))
            all_training_loss_guess = np.sqrt(np.vstack(all_training_loss_guess).reshape((-1,), order='F'))
            all_training_loss_guess_rescaled = np.sqrt(np.vstack(all_training_loss_guess_rescaled).reshape((-1,), order='F'))
            all_validation_loss = np.sqrt(np.array(all_validation_loss))
            all_validation_loss_guess = np.sqrt(np.array(all_validation_loss_guess))
            all_validation_loss_guess_rescaled = np.sqrt(np.array(all_validation_loss_guess_rescaled))

            all_epochs_training_loss = np.concatenate((all_epochs_training_loss,
                                                       all_training_loss))
            all_epochs_training_loss_guess = np.concatenate((all_epochs_training_loss_guess,
                                                             all_training_loss_guess))
            all_epochs_training_loss_guess_rescaled = np.concatenate((all_epochs_training_loss_guess_rescaled,
                                                                      all_training_loss_guess_rescaled))
            all_epochs_validation_loss = np.concatenate((all_epochs_validation_loss,
                                                         all_validation_loss))
            all_epochs_validation_loss_guess = np.concatenate((all_epochs_validation_loss_guess,
                                                               all_validation_loss_guess))
            all_epochs_validation_loss_guess_rescaled = np.concatenate((all_epochs_validation_loss_guess_rescaled,
                                                                        all_validation_loss_guess_rescaled))

            train_utils.do_diagnostic_output(all_epochs_training_loss, all_epochs_validation_loss,
                                             all_epochs_training_loss_guess, all_epochs_validation_loss_guess,
                                             all_epochs_training_loss_guess_rescaled, all_epochs_validation_loss_guess_rescaled,
                                             epoch+1, len(training_loader))

            train_utils.save_model(model, optimizer)


        if settings.RANK == 0 and settings.VERBOSE :
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
