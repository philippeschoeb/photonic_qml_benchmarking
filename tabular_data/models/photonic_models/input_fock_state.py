def get_input_fock_state(type, m, n):
    assert type in ['standard', 'spaced'], f'Type ({type}) is not supported'
    if type == 'standard':
        return get_standard_state(m, n)
    elif type == 'spaced':
        return get_spaced_state(m, n)
    else:
        raise NotImplementedError


def get_standard_state(m, n):
    return [1] * n + [0] * (m - n)


def get_spaced_state(m, n):
    state = [0] * m
    jump = m // n
    for i in range (0, m, jump):
        state[i] = 1
    return state


def print_simple_state(state):
    print(f'Fock state: {state}')
    return


def get_amp_input_fock_state(m, input_size, encoding_type):
    if encoding_type == 'per_state':
        #TODO when amplitude encoding will be implemented
        state = None
        return state
    elif encoding_type == 'per_mode':
        #TODO when amplitude encoding will be implemented
        state = None
        return state
    else:
        raise NotImplementedError(f'Unknown encoding type: {encoding_type}')