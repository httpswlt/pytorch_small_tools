
# timeout is 30 minutes
_default_pg_timeout = 30 * 60
# Default process group state
_default_pg = None
_default_pg_init_method = None


# Process group count for default naming
_group_count = 0




def init_process_group(backend,
                       init_method=None,
                       timeout=_default_pg_timeout,
                       world_size=-1,
                       rank=-1,
                       store=None,
                       group_name=''):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.
        If neither is specified, ``init_method`` is assumed to be "env://".


    Arguments:
        backend (str or Backend): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``"gloo"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
            multiple processes per machine with ``nccl`` backend, each process
            must have exclusive access to every GPU it uses, as sharing GPUs
            between processes can result in deadlocks.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process.
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is only applicable for the ``gloo`` backend.
        group_name (str, optional, deprecated): Group name.

    To enable ``backend == Backend.MPI``, PyTorch needs to built from source
    on a system that supports MPI. The same applies to NCCL as well.

    """
    global _pg_group_ranks
    global _backend
    global _default_pg
    global _default_pg_init_method

    if not isinstance(timeout, timedelta):
        raise RuntimeError("Expected timeout argument to be of type"
                           "datetime.timedelta")

    if _default_pg is not None:
        raise RuntimeError("trying to initialize the default process group "
                           "twice!")

    assert (store is None) or (init_method is None), \
        "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = "env://"

    backend = Backend(backend)

    if backend == Backend.MPI:
        _default_pg = _new_process_group_helper(
            -1,
            -1,
            [],
            Backend.MPI,
            None,
            group_name=group_name,
            timeout=timeout)
    else:
        # backward compatible API
        if store is None:
            url = init_method
            if world_size != -1 and rank != -1:
                url += "?rank={}&world_size={}".format(rank, world_size)
            elif rank != -1:
                url += "?rank={}".format(rank)
            elif world_size != -1:
                url += "?world_size={}".format(world_size)

            store, rank, world_size = next(rendezvous(url))
            store.set_timeout(timeout)

        _default_pg = _new_process_group_helper(
            world_size,
            rank,
            [],
            backend,
            store,
            group_name=group_name,
            timeout=timeout)

    _pg_group_ranks[_default_pg] = {i: i for i in range(_default_pg.size())}
    _backend = _pg_map[_default_pg][0]
    _default_pg_init_method = init_method