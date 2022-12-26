from numpy import exp, array, random, dot, maximum

class NeuralNetwork():
    def __init__(self):
        
        random.seed(1)

        
        self.s_weights = 2 * random.random((3, 1)) - 1

   
    def __relu(self, x):
        return maximum(0.1*x,x)

    
    def __relu_derivative(self, x):
        for i, val in enumerate(x): 
            x[i] = 0.1 if val <= 0 else 1
        return x

   
    def train(self, tsi, tso, training_iterations):
        for iteration in range(training_iterations):
            
            output = self.think(tsi)

            
            error = tso - output

           
            adjustment = dot(tsi.T, 0.1 * error * self.__relu_derivative(output))

            
            self.s_weights += adjustment

    
    def think(self, inputs):
        
        return self.__relu(dot(inputs, self.s_weights))


if __name__ == "__main__":

    
    neural_network = NeuralNetwork()

    print ("Random starting synaptic weights: ")
    print (neural_network.s_weights)

    
    tsi = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tso = array([[0, 0, 1, 1]]).T

    
    neural_network.train(tsi, tso, 10000)

    print ("New synaptic weights after training: ")
    print (neural_network.s_weights)

    
    print ("Considering new situation [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))
