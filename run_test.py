import torch

# very sophisticated test selector
# uncomment which test you want to run
# :)

from test.sample_gran import test
# from test.test_gran_node_labels import test
# from test.test_gran_edge_labels import test
# from test.test_gran_darts_medium import test
# from test.test_gran_nb301_sampling import test
# from test.test_oracle import test
# from test.test_oneshot import test

print("starting test")

print(f"cuda? {torch.cuda.is_available()}")

test()

print("ending test")
