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
