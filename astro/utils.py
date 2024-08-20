import random

def signed_rand_float(val_range):
    min_val, max_val = val_range
    val = random.uniform(min_val,max_val)
    if not random.randint(0,1):
        val = val*-1

    return val