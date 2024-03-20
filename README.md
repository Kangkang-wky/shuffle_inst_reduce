# shuffle_inst_reduce

Multidimensional vector reduction gradient update, reducing the number of shuffle instructions.

From nsight compute, we can see that the number of instructions has been reduced from 400 to 174, and the number of shuffle instructions and fadd instructions has been reduced, reducing invalid instructions during multi-dimensional reduction.