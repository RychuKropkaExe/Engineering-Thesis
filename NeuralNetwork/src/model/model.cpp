#include "model.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>      

Model::Model(vector<size_t> arch, size_t archSize, vector<ActivationFunctionE> actFunctions, size_t actFunctionsSize, bool randomize){
    assert(archSize == actFunctionsSize);
    this->activationFunctions.resize(actFunctionsSize);
    for(size_t i = 0; i < actFunctionsSize; ++i){
        this->activationFunctions[i] = actFunctions[i];
    }

    this->layers.resize(archSize - 1);
    this->numberOfLayers = archSize - 1;
    for(size_t i = 0; i < archSize - 1; ++i){
        pair<size_t, size_t> weightsDimensions;
        pair<size_t, size_t> biasesDimensions;
        pair<size_t, size_t> outputDimensions;
        SET_ROWS_IN_PAIR(weightsDimensions, arch[i]);
        SET_COLS_IN_PAIR(weightsDimensions, arch[i+1]);
        SET_ROWS_IN_PAIR(biasesDimensions, 1);
        SET_COLS_IN_PAIR(biasesDimensions, arch[i+1]);
        SET_ROWS_IN_PAIR(outputDimensions, 1);
        SET_COLS_IN_PAIR(outputDimensions, arch[i+1]);
        this->layers[i] = Layer(outputDimensions, weightsDimensions , biasesDimensions, actFunctions[i], randomize);
    }

    this->arch = arch;
    this->archSize = archSize;
}

Model::Model(){
    this->layers.resize(1);
    this->activationFunctions.resize(1);
}

void Model::setLearningRate(float val){
    this->learningRate = val;
}

void Model::setEps(float val){
    this->eps = val;
}

float Model::cost(){

    float totalCost = 0;
    std::cout << "COST FUNCTION!!! " << "\n";
    std::cout << "NUMBER OF SAMPLES: " << "\n";
    std::cout << trainingData.numOfSamples << "\n";
    trainingData.printTrainingData();
    for(size_t i = 0; i < trainingData.numOfSamples; ++i){

        FastMatrix result = run(trainingData.inputs[i]);

        for(size_t j = 0; j < result.cols; ++j){

            float d = MAT_ACCESS(result, 0, j) - MAT_ACCESS(trainingData.outputs[i], 0, j);
            totalCost += d*d;

        }

    }

    return totalCost/(this->trainingData.numOfSamples);

}

void Model::learn(TrainingData& trainingDataIn, size_t iterations){

    std::cout << "STARTED LEARNING" << "\n";
    this->trainingData = trainingDataIn;
    std::cout << "NUMBER OF SAMPLES: " << this->trainingData.numOfSamples << "\n";
    std::cout << "COST FUNCTION VALUE: " << cost() << "\n";
    float percentage = 0.1f;
    for(size_t i = 0; i < iterations; ++i){
        if( ((float)i / (float)iterations) >= percentage ){
            std::cout << "LEARNING COMPLETION: " << 100*percentage << "\n";
            std::cout << "COST FUNCTION VALUE: " << cost() << "\n";
            percentage += 0.1f;
        }
        backPropagation();

    }

}

FastMatrix Model::run(FastMatrix input){

    FastMatrix result = input;

    for(size_t i = 0; i < numberOfLayers; i++){

        result = layers[i].forward(result);

    }

    return result;

}

void Model::finiteDifference(){

    Model fakeGradient(arch, archSize, activationFunctions, archSize, false);

    float saved;
    float curCost = cost();

    for(size_t i = 0; i < numberOfLayers; ++i){

        for(size_t j = 0; j < layers[i].weights.rows; ++j){
            for(size_t k = 0; k < layers[i].weights.cols; k++){
                saved = MAT_ACCESS(layers[i].weights, j, k);
                MAT_ACCESS(layers[i].weights, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].weights, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].weights, j, k) = saved;
            }
        }

        for(size_t j = 0; j < layers[i].biases.rows; ++j){
            for(size_t k = 0; k < layers[i].biases.cols; k++){
                saved = MAT_ACCESS(layers[i].biases, j, k);
                MAT_ACCESS(layers[i].biases, j, k) += this->eps;
                float newCost = cost();
                MAT_ACCESS(fakeGradient.layers[i].biases, j, k) = (newCost - curCost) / this->eps;
                MAT_ACCESS(layers[i].biases, j, k) = saved;
            }
        }

    }
    for(size_t i = 0; i < this->numberOfLayers; ++i){

        for(size_t j = 0; j < this->layers[i].weights.rows; ++j){
            for(size_t k = 0; k < this->layers[i].weights.cols; ++k){
                MAT_ACCESS(layers[i].weights, j, k) -= learningRate*MAT_ACCESS(fakeGradient.layers[i].weights, j, k);
            }
        }

        for(size_t j = 0; j < this->layers[i].biases.rows; ++j){
            for(size_t k = 0; k < this->layers[i].biases.cols; ++k){
                MAT_ACCESS(layers[i].biases, j, k) -= learningRate*MAT_ACCESS(fakeGradient.layers[i].biases, j, k);
            }
        }

    }

}

void Model::backPropagation(){
    size_t n = trainingData.numOfSamples;

    Model gradient(arch, archSize, activationFunctions, archSize, false);

    for(size_t i = 0; i < gradient.numberOfLayers; ++i){
        gradient.layers[i].weights.set(0.f);
        gradient.layers[i].biases.set(0.f);
        gradient.layers[i].output.set(0.f);
    }

    // i - current sample
    // l - current layer
    // j - current activation
    // k - previous activation

    for (size_t i = 0; i < n; ++i) {

        run(trainingData.inputs[i]);

        for (size_t j = 0; j < numberOfLayers; ++j) {
            gradient.layers[j].output.set(0);
        }

        for (size_t j = 0; j < trainingData.outputSize; ++j) {
            MAT_ACCESS(gradient.layers[numberOfLayers-1].output, 0, j) = MAT_ACCESS(layers[numberOfLayers-1].output, 0, j) - MAT_ACCESS(trainingData.outputs[i], 0, j);
        }
        
        // Basically, my model calss does not consider input as a proper
        // member of the first layer, but consider the output as a proper
        // member of the last layer. This create a sort of problem. Since we want
        // to iterate backwords in, well, BACKWORDS propagation, we need access to
        // previous layer at they full. This means that previous actviation is stored
        // in layer[l-1].output, but at the same time the 0th activation is not present in the model.
        // It looks for example model like this:
        //
        // layer1 weight, layer1 bias, layer1 output
        // layer2 weight, layer2 bias, layer2 output
        // layer3 weight, layer3 bias, layer3 output(model output)
        //
        // And when doing backwords propagation we pair them like this:
        //
        // l3 out(model output), l3 bias, l3 weights, l2 out(l3 input)
        // l2 out, l2 bias, l2 wieghts, l1 out(l2 input)
        // l1 out, l1 bias, l1 weights, l0 out(model input) <-- missing
        //
        // We both need it to look like that, and don't at the same time.
        // We could abstract it out, but this provides us with easy access,
        // and use of intermediate buffers which are layers outputs. 
        // This also binds the values together in imaginable way, because
        // They are connected. 
        // Hence the process is divided into iterating over all layers until first
        // And then iterating over first layer separetly.

        for (size_t l = numberOfLayers - 1; l > 0; --l) {
            for (size_t j = 0; j < layers[l].output.cols; ++j) {

                float a = MAT_ACCESS(layers[l].output, 0, j);
                float da = MAT_ACCESS(gradient.layers[l].output, 0, j);
                MAT_ACCESS(gradient.layers[l].biases, 0, j) += 2*da*a*(1-a);

                for (size_t k = 0; k < layers[l-1].output.cols; ++k) {
                    // j - weight matrix col
                    // k - weight matrix row
                    float pa = MAT_ACCESS(layers[l-1].output, 0, k);
                    float w = MAT_ACCESS(layers[l].weights, k, j);
                    MAT_ACCESS(gradient.layers[l].weights, k, j) += 2*da*a*(1 - a)*pa;
                    MAT_ACCESS(gradient.layers[l-1].output, 0, k) += 2*da*a*(1 - a)*w;
                }
            }
        }

        FastMatrix inp;
        inp.cols = trainingData.inputs[i].cols;
        inp.rows = trainingData.inputs[i].rows;
        inp.mat = trainingData.inputs[i].mat;

        // Here the iteration over first layer happens

        for (size_t j = 0; j < layers[0].output.cols; ++j) {

            float a = MAT_ACCESS(layers[0].output, 0, j);
            float da = MAT_ACCESS(gradient.layers[0].output, 0, j);
            MAT_ACCESS(gradient.layers[0].biases, 0, j) += 2*da*a*(1-a);

            for (size_t k = 0; k < inp.cols; ++k) {
                // j - weight matrix col
                // k - weight matrix row
                float pa = MAT_ACCESS(inp, 0, k);
                float w = MAT_ACCESS(layers[0].weights, k, j);
                MAT_ACCESS(gradient.layers[0].weights, k, j) += 2*da*a*(1 - a)*pa;
                MAT_ACCESS(inp, 0, k) += 2*da*a*(1 - a)*w;
            }

        }



    }

    //gradient.printModel();

    for (size_t i = 0; i < numberOfLayers; ++i) {
        for (size_t j = 0; j < gradient.layers[i].weights.rows; ++j) {
            for (size_t k = 0; k < gradient.layers[i].weights.cols; ++k) {
                MAT_ACCESS(gradient.layers[i].weights, j, k) /= n;
            }
        }
        for (size_t j = 0; j < gradient.layers[i].biases.rows; ++j) {
            for (size_t k = 0; k < gradient.layers[i].biases.cols; ++k) {
                MAT_ACCESS(gradient.layers[i].biases, j, k) /= n;
            }
        }
    }

    //gradient.printModel();
    for(size_t i = 0; i < numberOfLayers; ++i){

        for(size_t j = 0; j < layers[i].weights.rows; ++j){
            for(size_t k = 0; k < layers[i].weights.cols; ++k){
                MAT_ACCESS(layers[i].weights, j, k) -= learningRate*MAT_ACCESS(gradient.layers[i].weights, j, k);
            }
        }

        for(size_t j = 0; j < this->layers[i].biases.rows; ++j){
            for(size_t k = 0; k < this->layers[i].biases.cols; ++k){
                MAT_ACCESS(layers[i].biases, j, k) -= learningRate*MAT_ACCESS(gradient.layers[i].biases, j, k);
            }
        }

    }

}

void Model::printModel(){
    
    std::cout << "MODEL PARAMETERS: " << "\n";
    std::cout << "LEARNING RATE: " << learningRate << "\n";
    std::cout << "EPS: " << eps << "\n";
    std::cout << "NUMBER OF LAYERS: " << numberOfLayers << "\n";
    std::cout << "ACTIVATION FUNCTIONS: " << activationFunctions[0] << "\n";
    std::cout << "LAYERS: " << "\n";

    for(size_t i = 0; i < numberOfLayers; ++i){
        std::cout << "LAYER NUMBER: " << i << "\n";
        std::cout << "WEIGHTS: " << "\n";
        printFastMatrix(layers[i].weights);
        std::cout << "BIASES: " << "\n";
        printFastMatrix(layers[i].biases);
        std::cout << "INTERMIEDIATE: " << "\n";
        printFastMatrix(layers[i].output);
        std::cout << "ACTIVATION FUNCTION: " << layers[i].functionType << "\n";
    }

}

void Model::printModelToFile(std::string filename){
    std::ofstream f(filename);

  // Write to the file
    f << numberOfLayers << "\n";
    f << learningRate << "\n";
    f << eps << "\n";
    for(size_t i = 0; i < numberOfLayers; ++i){

        f << layers[i].functionType << "\n";
        f << layers[i].weights.rows << "\n";
        f << layers[i].weights.cols << "\n";
        for(size_t j = 0; j < layers[i].weights.rows*layers[i].weights.cols; ++j){
            f << layers[i].weights.mat[j] << " ";
        }
        f << "\n";
        f << layers[i].biases.rows << "\n";
        f << layers[i].biases.cols << "\n";
        for(size_t j = 0; j < layers[i].biases.rows*layers[i].biases.cols; ++j){
            f << layers[i].biases.mat[j] << " ";
        }
        f << "\n";
    }

    // Close the file
    f.close();
}

Model parseModelFromFile(std::string filename){
    std::ifstream f(filename);
    std::string buffer;

    getline (f, buffer);
    size_t numberOfLayers = (size_t)stoi(buffer);

    getline (f, buffer);
    float learningRate = (float)stof(buffer);

    getline (f, buffer);
    float eps = (float)stof(buffer);

    vector<Layer> layers;
    layers.resize(numberOfLayers);
    for(size_t i = 0; i < numberOfLayers; ++i){

        getline (f, buffer);
        ActivationFunctionE funcType = (ActivationFunctionE)stoi(buffer);
        getline (f, buffer);
        size_t weightsRows = (size_t)stoi(buffer);
        getline (f, buffer);
        size_t weightsCols = (size_t)stoi(buffer);
        vector<float> wieghtsMat;
        wieghtsMat.resize(weightsRows*weightsCols);
        getline (f, buffer);
        std::stringstream check(buffer);

        size_t counter = 0;

        while(getline(check, buffer, ' '))
        {   
            wieghtsMat[counter] = stof(buffer);
            counter++;
        }

        getline (f, buffer);
        size_t biasesRows = (size_t)stoi(buffer);
        getline (f, buffer);
        size_t biasesCols = (size_t)stoi(buffer);
        vector<float> biasesMat;
        biasesMat.resize(biasesCols*biasesRows);
        
        counter = 0;
        getline (f, buffer);
        std::cout << buffer << "\n";

        std::stringstream check2(buffer);
        while(getline(check2, buffer, ' '))
        {
            std::cout << buffer << "\n";
            biasesMat[counter] = stof(buffer);
            counter++;
        }

        FastMatrix weights;
        FastMatrix biases;
        FastMatrix output;
        weights.cols = weightsCols;
        weights.rows = weightsRows;
        weights.mat = wieghtsMat;
        biases.cols = biasesCols;
        biases.rows = biasesRows;
        biases.mat = biasesMat;
        output.cols = biasesCols;
        output.rows = biasesRows;
        output.mat = biasesMat;

        Layer l;

        l.weights = weights;
        l.biases = biases;
        l.output = output;
        l.functionType = funcType;

        layers[i] = l;

    }

    Model model;

    model.layers = layers;
    model.numberOfLayers = numberOfLayers;
    model.archSize = numberOfLayers;
    model.learningRate = learningRate;
    model.eps = eps;

    return model;

}