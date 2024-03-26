import argparse

def parse_list_arg(arg):
    try:
        # Try to parse the argument as a single integer
        return [str((arg))]
    except ValueError:
        # If it's not a single integer, try to parse it as a list of integers
        try:
            return [str(x) for x in arg.split(',')]
        except ValueError:
            # If parsing as a list fails, raise an error
            raise argparse.ArgumentTypeError("Invalid format for 'id' argument. It should be either a single integer or a comma-separated list of integers.")

