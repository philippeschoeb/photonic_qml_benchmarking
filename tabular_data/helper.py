

def architecture_help(model):
    if model == 'dressed_quantum_circuit' or model == 'dressed_quantum_circuit_reservoir':
        message = f'Here is the architecture string formatting of the {model}:\n\n'
        message += ('--architecture m_{i}_n_{j}\n\nWhere i is the number of modes,\nand j is the number of photons.\n')
        return message
    #TODO
    else:
        message = f'There is no current help for the architecture string formatting of the {model}\n'
        return message