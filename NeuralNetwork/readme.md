Author: Józef Melańczuk

Synopsis:
```
Neural Netowrk framework with configurable architecture and activation functions. The network
does calculations on CPU and is optimized in this direction.
```

Available activation functions:
```c
SIGMOID
RELU
SOFTMAX
```

The framework is tested by modeling given problems:
```
Digit recognition using MNIST Dataset
Parabole function for number in range (-20, 20)
Calculating Hamming length of 7 bit-number
Determining parity of 8-bit number
Simulating a XOR gate
```

Building:
```
Project uses CMake to compile all sources.
Since the framework is exported to a python module
using Pybind11Extension no library is produced for the
project and the resulting executable is for running tests
on the framework. The project is designed for Windows operating system
since the target game for reinforcement learning is only playable on this platform
```

The compilation is done in following steps:
```
from powershell enter bash terminal(MSYS for example)
source gitenv.sh
cd build
./build.sh
make
./MainBuild.exe
```
