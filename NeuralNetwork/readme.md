# NeuralNetwork

A CPU-based neural network project written in C++20 for an engineering thesis. It explores two ways to build and train networks: configurable dense layers trained with backpropagation, and graph-based models whose structure and parameters evolve through a genetic algorithm.

Author: **Józef Melańczuk**

[Models](#two-modeling-approaches) · [Getting started](#getting-started) · [Tests](#tests-and-learning-examples) · [Evolution](#evolving-a-network) · [Data](#training-data) · [Logging](#logging-and-profiling)

## Two modeling approaches

| | Fixed topology | Dynamic topology |
|---|---|---|
| Representation | Dense layers backed by the custom `FastMatrix` class | A directed graph of neurons and synapses |
| Training | Mini-batch backpropagation, with optional gradient clipping | Population evolution through speciation, selection, crossover, and mutation |
| Architecture | Layer sizes chosen when constructing the model | Hidden neurons and connections can be added or removed during evolution |
| Activations | Sigmoid, ReLU variant, softmax, and no activation | Sigmoid, ReLU, and no activation |
| Main API | [`Model`](src/FixedTopologyModel/Model/model.h) | [`DTModel`](src/DynamicTopologyModel/DTModel/dtmodel.h) and [`DTMGeneticAlgorithm`](src/DynamicTopologyModel/DTMGeneticAlgorithm/DTMAlgorithm/dtmgeneticAlgorithm.h) |

The numerical operations are implemented directly in C++, including matrix arithmetic and forward propagation. Shared utilities provide training-data loading, normalization, and timestamp logging. The fixed-topology module also contains experimental, strategy-based genetic algorithm components; the complete evolution loop is implemented in the dynamic-topology module.

## Getting started

### Requirements

- CMake **3.30 or newer**.
- A C++20 compiler; the commands below use GCC and a Bash shell.
- A build tool supported by CMake, such as Make or Ninja.
- Internet access during the initial configuration to download GoogleTest through CMake's `FetchContent`.

The current CMake target is **`MainBuild`**, a GoogleTest executable. It does not produce an installable library or Python extension.

### Build and run the quick checks

From the repository root:

```bash
cd NeuralNetwork
source gitenv.sh

cmake -S . -B build -DCMAKE_CXX_COMPILER=g++ -DDEBUG_MODE=0 -DLOG_PRIO=1
cmake --build build --parallel

cd build
./MainBuild --gtest_filter='-DTMGeneticAlgorithmTest.hammingLengthTest:DTMGeneticAlgorithmTest.digitRecognitionTest'
```

Source `gitenv.sh` **from the `NeuralNetwork` directory** in each new shell before running data-dependent tests. It sets `GIT_REPOSITORY_NEURAL_NETWORK_PATH`, which the tests use to locate datasets independently of their working directory.

The filter above excludes the two full evolutionary training tests. On Windows, use a GCC-based environment such as MSYS2/MinGW and the generated `MainBuild.exe` executable; the current compiler flags are GCC-style, not MSVC-style.

## Tests and learning examples

The test suite covers matrix arithmetic, data preprocessing, graph edits, topological sorting, forward propagation, crossover, mutation, fitness calculations, and the evolution loop. Test headers are registered through [`test/runTests.cpp`](test/runTests.cpp).

Run these commands from `NeuralNetwork/build`, after setting the environment as shown above:

```bash
# Discover the registered tests.
./MainBuild --gtest_list_tests

# Run everything, including the longer training tests.
./MainBuild

# Run one evolutionary learning example.
./MainBuild --gtest_filter=DTMGeneticAlgorithmTest.hammingLengthTest
./MainBuild --gtest_filter=DTMGeneticAlgorithmTest.digitRecognitionTest
```

CTest is also configured: `ctest --output-on-failure` runs the registered tests, including the training tests.

### Dynamic-topology learning tests

Both examples are defined in [`dtmgeneticAlgorithmTest.h`](test/dtmgeneticAlgorithmTest.h):

| Example | Inputs → outputs | Data | Current configuration |
|---|---|---|---|
| Hamming count | 7 bits → 3-bit count of set bits | All 128 inputs in `hammingLengthTest.txt` | 100 individuals, 4,000 generations, sigmoid outputs |
| Digit recognition | 16 pen-stroke features → one numeric digit label | 7,494 training and 3,498 test samples in `pendigits.tra` / `pendigits.tes` | 50 individuals, 3,000 generations, ReLU output |

These are full training runs, so expect them to take substantially longer than the structural and numerical checks. Both assert a **training cost of at most `0.05`**. The digit test uses min–max normalization and records held-out cost separately; it does not assert a held-out threshold. This cost is a squared-error metric, **not a classification accuracy percentage**. The digit data is pen-feature data, not MNIST images.

### Fixed-topology examples

[`modelTest.h`](test/modelTest.h) contains learning examples for XOR, a quadratic function, parity of an 8-bit number, Hamming count, and digit recognition. Its include in `runTests.cpp` is currently commented out. Enable that include and rebuild to register those tests; they also perform longer training runs.

## Evolving a network

`DTMGeneticAlgorithm::run(numberOfGenerations)` initializes a fresh population and returns the **best evaluated individual across all generations**.

Each generation performs:

1. **Speciation** — group models by their maximum depth.
2. **Fitness evaluation** — evaluate every individual and preserve the best seen so far.
3. **Tournament selection** — choose parents within each species, accounting for mutation grace periods.
4. **Crossover** — combine shared elements and inherit the fitter parent's non-shared elements.
5. **Mutation** — attempt the configured structural and parameter changes.

Crossover matches neurons by `depthId`, not their unique IDs. Synapses match when both endpoint neurons have corresponding `depthId` values. Offspring models are topologically sorted for forward propagation.

### Example configuration

The following snippet uses the project's headers and mirrors the Hamming-count test. The dataset path assumes execution from `NeuralNetwork`; adjust it when running elsewhere.

```cpp
#include "dtmgeneticAlgorithm.h"
#include "meanSquaredEval.h"
#include "trainingData.h"
#include <cstdlib>

DTIndividual evolveHammingCount()
{
    TrainingData data("test/TestData/hammingLengthTest.txt");

    Hyperparameters parameters{};
    parameters.inputSize = 7;
    parameters.outputSize = 3;
    parameters.outputActivation = DTMUtils::ActivationE::SIGMOID;
    parameters.populationSize = 100;
    parameters.maxNumberOfNeurons = 40;
    parameters.tournamentSize = 5;
    parameters.gracePeriodLength = 2;

    using DTMUtils::MutationE;
    parameters.mutationTypes = {
        MutationE::ADD_NEURON, MutationE::REMOVE_NEURON,
        MutationE::ADD_SYNAPSE, MutationE::REMOVE_SYNAPSE,
        MutationE::ADJUST_WEIGHT, MutationE::ADJUST_BIAS,
        MutationE::CHANGE_ACTIVATION
    };
    parameters.numberOfMutations = {2, 1, 4, 1, 10, 10, 1};
    parameters.weightMutationStrength = 2.0;
    parameters.biasMutationStrength = 2.0;

    std::srand(2026);
    MeanSquaredEval fitnessEvaluation(data);
    DTMGeneticAlgorithm algorithm(parameters, &fitnessEvaluation);
    return algorithm.run(4000);
}
```

`mutationTypes` and `numberOfMutations` must have matching lengths. Each count specifies how many distinct individuals are selected for that mutation type, not how many mutations are guaranteed to succeed. An individual may be selected for multiple different mutation types. An impossible mutation is skipped without retrying; a successful mutation resets the individual's grace period.

The neuron limit includes input and output neurons. Weight and bias mutation strengths bound the magnitude of their additive changes. Activation mutations affect hidden neurons only.

### Fitness

[`MeanSquaredEval`](src/DynamicTopologyModel/DTMGeneticAlgorithm/DTMAlgorithm/Fitness/MeanSquaredEval/meanSquaredEval.h) implements dataset-based fitness. Its constructor validates and stores an immutable copy of the supplied `TrainingData`. It sums squared errors across outputs, then averages over samples:

```text
MSE = sum_over_samples(sum_over_outputs((prediction - target)^2)) / sample_count
fitness = 1 / MSE
```

Other fitness strategies can derive from [`DTMFitnessEvaluation`](src/DynamicTopologyModel/DTMGeneticAlgorithm/DTMAlgorithm/Fitness/dtmfitnessEvaluation.h) and override `evaluateIndividual`. The base implementation throws `std::logic_error`. Population evaluation uses OpenMP, so a shared evaluator must support concurrent calls on distinct individuals. Evaluator exceptions are rethrown on the calling thread after the workers finish. The algorithm itself has no dependency on training data.

## Training data

[`TrainingData`](src/TrainingData/trainingData.h) loads samples into `FastMatrix` objects and provides normalization, min–max scaling, and standardization.

The text format is:

```text
number_of_samples
input_size
output_size
<all input rows: one row per sample>
<all output rows: one row per sample>
```

Values within a row are space-separated. Inputs and outputs are stored in **separate blocks**, not alternating rows. For example, [`xorData.txt`](test/TestData/xorData.txt) contains:

```text
4
2
1
0 0
1 0
0 1
1 1
0
1
1
0
```

The bundled datasets are in [`test/TestData`](test/TestData). The quadratic-function example generates its data in code.

## Logging and profiling

The [`Logger`](src/Logger/logger.h) module provides priority-based logs and paired `TIME_MEASURE_BEGIN` / `TIME_MEASURE_END` events. Instrumented operations include evolution, initialization, speciation, selection, crossover, and model edits.

- `LOG_PRIO=1`: essential logs and standard timing events.
- `LOG_PRIO=2`: adds normal-detail logs.
- `LOG_PRIO=3`: adds heavy logs, including per-call forward-propagation timing. This can generate very large files and affect measured performance.
- `DEBUG_MODE=1`: selects an unoptimized build with debug symbols; use `DEBUG_MODE=0` for optimized performance measurements.

Ordinary logs go to **`logs.log`**, while timestamps go to **`timeStampLog.log`**, both in the process's working directory. Each process opens these files in overwrite mode, so preserve logs before starting another run and avoid parallel test processes sharing these files when collecting timings.

Timestamp logging uses a dedicated worker and two reusable buffers of 4,096 entries. The main thread captures events without per-event file I/O, switching buffers when full and waiting only when the worker still owns the next buffer. `test/runTests.cpp` starts the worker and drains all remaining entries before joining it at shutdown. This is a single-producer interface: timing macros must be called on the main thread.

Timestamp records use `[TIME_STAMP]: EVENT_BEGIN <name> <microseconds>` and the corresponding `EVENT_END` form. The `TIME_MEASURE_BEGIN` / `TIME_MEASURE_END` call sites are unchanged.

Timestamp values are microseconds relative to the logger's clock origin. Subtract an event's start from its matching end to get elapsed time. Nested events include their children's time, so summing all event durations does not give total runtime.

Use the [timestamp analyzer](tools/timeStampAnalyzer/readme.md) to report event totals and percentages separately for each `DTM_GENETIC_RUN`. From `NeuralNetwork`:

```bash
python3 tools/timeStampAnalyzer/timeStampAnalyzer.py build/timeStampLog.log
```

## Project layout

```text
NeuralNetwork/
├── src/
│   ├── FixedTopologyModel/   # Dense models, matrix operations, training, GA components
│   ├── DynamicTopologyModel/ # Graph models and the topology-evolving genetic algorithm
│   ├── TrainingData/         # Dataset loading and preprocessing
│   ├── Logger/               # Logging and timing macros
│   └── AgentControl/         # Legacy Trackmania/Python integration
├── test/                     # GoogleTest suites and bundled TestData/
├── tools/timeStampAnalyzer/  # Streaming Python reports for timestamp logs
├── CMakeLists.txt            # C++20 build and GoogleTest configuration
├── gitenv.sh                 # Dataset-path environment setup
└── setup.py                  # Legacy Python-extension build configuration
```

The dynamic genetic algorithm is split into initialization, speciation, selection, crossover, mutation, fitness, and orchestration source files under [`DTMAlgorithm`](src/DynamicTopologyModel/DTMGeneticAlgorithm/DTMAlgorithm).

### Legacy Python integration

[`src/AgentControl`](src/AgentControl) and [`setup.py`](setup.py) retain an earlier pybind11/Trackmania integration. They are not part of the current CMake build, and their paths and API usage need updating before the Python extension can be used with the current code. The supported build workflow documented here is the C++ test executable.
