def batching(list_of_iterables, n=1, infinite=False, return_incomplete_batches=False):
    """
    Batch computation helper. Given a list of iterable objects, the function chops the iterables in
    equal size batches and yields them using a generator structure.
    :param list_of_iterables: list of iterable objects of the same length for which the function
    will create the batches (list of iterable objects; e.g. list of lists)
    :param n: size of the batches to generate (int)
    :param infinite: controls if it should finish or it should start from the begining at the end of
    the iterable (bool)
    :param return_incomplete_batches: controls if the function should return the last batch (which may
    be incomplete in case n is not multiple of the length of the lists) (bool)
    :return: [generator] it yields a list of lists representing a batch with the same structure as 
    list_of_iterables but instead, the lists contained in the main list have length n (list of lists)
    """
    list_of_iterables = [list_of_iterables] if type(list_of_iterables) is not list else list_of_iterables
    assert(len({len(it) for it in list_of_iterables}) == 1)
    l = len(list_of_iterables[0])
    while 1:
        for ndx in range(0, l, n):
            if not return_incomplete_batches:
                if (ndx+n) > l:
                    break
            yield [iterable[ndx:min(ndx + n, l)] for iterable in list_of_iterables]

        if not infinite:
            break