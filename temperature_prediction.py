import numpy as np
import tensorflow as tf

# define MLP regression model for temperature prediction
class TemperatureMLP(tf.keras.Model):
    def __init__(self, activation):
        super().__init__()
        self.layer1 = tf.keras.layers.Dense(4, activation=activation)
        self.layer2 = tf.keras.layers.Dense(4, activation=activation)
        self.temperature = tf.keras.layers.Dense(1, activation=activation)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.temperature(x)
        
        return x

    def model(self):
        inputs = tf.keras.Input(shape=(1,8))
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

# generate a single training example of shape (mu_vec, temp) from base materials
def generate_example(mu_mtx, mu_std_mtx, temperatures):
    rand_index = np.random.randint(low=0, high=mu_mtx.shape[0])
    mu_randtemp = mu_mtx[rand_index, :]
    mu_std = mu_std_mtx[rand_index, :] 

    mu_temp33 = mu_mtx[0,:] # attenuations at 33 C
    rand_noise = np.random.normal(np.zeros(4), mu_std, 4)
    delta = (mu_randtemp + rand_noise - mu_temp33) * 100
    input_vec = np.concatenate([mu_randtemp, delta], axis=0)

    return input_vec, temperatures[rand_index]

# predict temperature on all data for a material
def predict_on_material(mu_mtx, std_mtx, temps, model, verbose=False, noise=False):
    # dictionaries are mutable
    mu_mtx = np.copy(mu_mtx)
    std_mtx = np.copy(std_mtx)
    predictions = []
    temp33 = mu_mtx[0,:] 
    std33 = std_mtx[0,:]

    for i in range(len(temps)):
        input_vec = np.reshape(mu_mtx[i, :], (1,4))
        input_std = std_mtx[i,:]
        
        if noise: 
            temp33 += np.random.normal(np.zeros(4), std33, 4) / np.sqrt(10)
            input_vec += np.random.normal(np.zeros(4), input_std, 4) / np.sqrt(10)

        delta = (input_vec - temp33) * 100
        network_inp = np.concatenate([input_vec, delta], axis=1)

        pred = model(network_inp).numpy()[0][0]
        predictions.append(pred)

        if verbose:
            print(f"prediction: {pred:.2f} C, label: {temps[i]} C")
    print("")

    return predictions

# compute standard deviation of predictions
def get_pred_std(mu_mtx, std_mtx, temps, model, num):
    predictions = []
    
    for i in range(num):
        preds = predict_on_material(mu_mtx, std_mtx, temps, model, noise=True)
        predictions.append(preds)

    return np.std(np.array(predictions), axis=0)
