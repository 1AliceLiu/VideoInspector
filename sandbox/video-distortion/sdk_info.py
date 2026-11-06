from pyservicemaker import Pipeline
from pyservicemaker._pydeepstream import Tensor
print("PIPELINE",[x for x in dir(Pipeline) if not x.startswith("_")])
print("START",Pipeline.start.__doc__)
print("CLONE",Tensor.clone.__doc__)
print("TENSOR",[x for x in dir(Tensor) if not x.startswith("_")])
