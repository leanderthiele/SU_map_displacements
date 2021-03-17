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
                                 training_loss=np.sqrt(training_loss),
                                 validation_loss=np.sqrt(validation_loss))
#}}}


def save_model(model) :
    # this function will only be called from the rank=0 process
    # TODO we probably want to store other data as well, most importantly the optimizer state dict
    #      other things we can put in are the loss curves
#{{{
    torch.save(model.state_dict(), settings.MODEL_FILE)
#}}}


def load_model(model, rank) :
    # this function will be called from any process, we need to make sure we map
    # the tensors to the correct devices
    # TODO we probably want to store other data as well, most importantly the optimizer state dict
    #      other things we can put in are the loss curves
    # TODO this function may not be correct in mpi mode
#{{{
    map_location = { 'cuda:0' : 'cuda:%d'%rank }
    model.load_state_dict(torch.load(settings.MODEL_FILE, map_location=map_location))
#}}}


def setup_process(rank, world_size, host_name) :
    # to be called at the beginning of a child process
#{{{
    # new process needs to get a consistent view of the settings
    startup.main(DataModes.TRAINING)

    # these are taken from the example at https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    os.environ['MASTER_ADDR'] = host_name
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group('nccl', rank=rank, world_size=world_size)

    torch.cuda.set_device(0 if settings.MPI else rank)
#}}}


def is_output_responsible(rank, world_size) :
#{{{
    # we don't want to store more data than really necessary on the 0th rank
    # which has to collect all the gradients already

    if world_size == 1 :
        return rank == 0
    else :
        return rank == 1
#}}}


def cleanup_process() :
    # to be called at the end of a child process
#{{{
    torch.distributed.destroy_process_group()
#}}}


def training_process(rank, world_size, host_name) :
    """
    A single training process, working on its own data.

    training_loss and validation_loss are in shared memory,
    the individual processes write into separate locations according to their rank.
    The root process is responsible for keeping us updated about training progress
    by periodically giving some loss output.
    """
#{{{

    setup_process(rank, world_size, host_name)

    model = Network().to(0 if settings.MPI else rank).to_ddp(rank, world_size)
    
    loss_fn = Loss()
    optimizer = Optimizer(model.parameters())

    # reset the optimizer -- not sure if it is necessary here but can't hurt
    optimizer.zero_grad()

    training_loader = DataLoader(DataModes.TRAINING, rank, world_size)
    validation_loader = DataLoader(DataModes.VALIDATION, rank, world_size)

    if is_output_responsible(rank, world_size) :
        all_epochs_training_loss = np.empty(0)
        all_epochs_validation_loss = np.empty(0)

    # keep track of whether we encounter infinities / nans
    global_inf = False
    inf_list = [False, ] * world_size

    # initialize outside the epoch loop so we can carry samples over from one
    # epoch to the other if a batch is incomplete
    idx_in_batch = 0
        

    for epoch in range(settings.EPOCHS) :
        
        if rank == 0 :
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
            
            if rank == 0 :
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
                    print('encountered invalid loss in rank %d'%rank)
                else :
                    # update gradients asynchronously in those processes that did not have a problem
                    with model.no_sync() :
                        this_training_loss.backward()

            if should_update_weights :
                # each process sees the same gradients and can therefore do the same weight updates
                optimizer.step()
                optimizer.zero_grad()
                idx_in_batch = 0
            
            if rank == 0 and settings.VERBOSE :
                print('\tSample %.3d / %d finished in epoch %d, '\
                      'took %f seconds'%(t+1, len(training_loader), epoch+1, time()-start_time_sample))

        # set model into evaluation mode
        model.eval()
        
        # loop once through the validation data
        with torch.no_grad() :
            for t, data in enumerate(validation_loader) :

                assert isinstance(data, Batch)

                inputs, targets, styles = data.get_on_device()

                prediction = model(inputs, styles)
                validation_loss += loss_fn(prediction, targets).item()

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

        if is_output_responsible(rank, world_size) :
            # interleave the training loss arrays so the losses are temporally correctly ordered
            all_training_loss = np.vstack(all_training_loss).reshape((-1,), order='F')
            all_validation_loss = np.array(all_validation_loss)

            all_epochs_training_loss = np.concatenate((all_epochs_training_loss,
                                                       all_training_loss))
            all_epochs_validation_loss = np.concatenate((all_epochs_validation_loss,
                                                         all_validation_loss))

            do_diagnostic_output(all_epochs_training_loss, all_epochs_validation_loss,
                                 epoch+1, len(training_loader), world_size)

            save_model(model)

        if rank == 0 :
            print('Epoch %d finished, took %f seconds'%(epoch+1, time()-start_time_epoch))

    # we're done, let's release resources
    cleanup_process()
#}}}


def get_mpi_env() :
    """
    either we're working on a single node or using srun
    In the latter case, we need to figure out how many friends we have.

    We are returning
        [0] SLURM_NTASKS
        [1] SLURM_PROCID
        [2] SLURMD_NODENAME
    """
#{{{
    assert settings.MPI

    env_vars_str = ['SLURM_NTASKS', 'SLURM_PROCID', 'SLURMD_NODENAME']
    env_vars = []
    
    try :
        for env_var_str in env_vars_str :
            env_vars.append(os.environ[env_var_str])
        env_vars[0] = int(env_vars[0])
        env_vars[1] = int(env_vars[1])
        print('on %s : found SLURM_NTASKS=%d, SLURM_PROCID=%d'%(env_vars[2], env_vars[0], env_vars[1]))
    except KeyError :
        if len(env_vars) != 0 :
            raise RuntimeError('Found only some of the expected slurm environment variables, We better stop')

        env_vars = [ 1, 0, 'localhost' ]


    return env_vars
#}}}


def get_host_name(mpi_rank, node_name) :
    """
    returns the name of the host
    not called -- and thus no mpi import -- if not run with srun
    """
#{{{
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    assert comm.Get_rank() == mpi_rank

    if mpi_rank == 0 :
        host_name = node_name
    else :
        host_name = None

    host_name = comm.bcast(host_name, root=0)

    return host_name
#}}}
        

def main() :
    """
    launches a couple of training_process's
    """
#{{{

    world_size = torch.cuda.device_count()

    if settings.MPI :
        mpi_world_size, mpi_rank, node_name = get_mpi_env()
        host_name = get_host_name(mpi_rank, node_name)
        assert world_size == 1 # only implement training on a single device per process
        print('on node %s, rank %d : host = %s'%(node_name, mpi_rank, host_name))
    else :
        host_name = 'localhost'

    assert mpi_world_size > 0
    assert mpi_rank < mpi_world_size


    if settings.MPI :
        training_process(mpi_rank, mpi_world_size, host_name)
    else :
        torch_mp.spawn(training_process,
                       args=(world_size, host_name, ),
                       nprocs=world_size)
#}}}

if __name__ == '__main__' :
    startup.main(DataModes.TRAINING)
    main()
