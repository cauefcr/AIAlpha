from get_data import GetData
from preprocessing import PreProcessing
from autoencoder import AutoEncoder
from data_processing import DataProcessing
from model import NeuralNetwork
from model_20_encoded import nnmodel
import tensorflow as tf
from keras.models import Model
import keras.layers as kl
import keras as kr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model,load_model
from keras import regularizers
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

import IPython

class PreProcessing:
    def __init__(self, split, feature_split, csv_, out_, out_test,log_train):
        self.split = split
        self.feature_split = feature_split
        self.stock_data = pd.read_csv(csv_)
        self.csv = csv_
        self.out = out_
        self.out_test = out_test
        self.log_train = log_train

    # wavelet transform and create autoencoder data
    def make_wavelet_train(self):
        train_data = []
        test_data = []
        log_train_data = []
        for i in range(22,(len(self.stock_data)//10)*10 - 11):
            train = []
            log_ret = []
            for j in range(1, 6):
                # if i > 11:
                x = np.array(self.stock_data.iloc[i-11:i,j])
                # IPython.embed()
                (ca, cd) = pywt.dwt(x, "haar")
                cat = pywt.threshold(ca, np.std(ca), mode="soft")
                cdt = pywt.threshold(cd, np.std(cd), mode="soft")
                tx = pywt.idwt(cat, cdt, "haar")
                log = np.diff(np.log(tx))*100
                macd = np.mean(x[5:]) - np.mean(x)
                # ma = np.mean(x)
                sd = np.std(x)
                log_ret = np.append(log_ret, log)
                x_tech = np.append(macd*10, sd)
                train = np.append(train, x_tech)
            train_data.append(train)
            log_train_data.append(log_ret)
        trained = pd.DataFrame(train_data)
        trained.to_csv("preprocessing/indicators.csv")
        log_train = pd.DataFrame(log_train_data, index=None)
        log_train.to_csv(self.log_train)
        # auto_train = pd.DataFrame(train_data[0:800])
        # auto_test = pd.DataFrame(train_data[801:1000])
        # auto_train.to_csv("auto_train.csv")
        # auto_test.to_csv("auto_test.csv")
        rbm_train = pd.DataFrame(log_train_data[0:int(self.split*self.feature_split*len(log_train_data))], index=None)
        rbm_train.to_csv(self.out)
        rbm_test = pd.DataFrame(log_train_data[int(self.split*self.feature_split*len(log_train_data))+1:
                                               int(self.feature_split*len(log_train_data))])
        rbm_test.to_csv(self.out_test)
        for i in range((len(self.stock_data) // 10) * 10 - 11):
            y = 100*np.log(self.stock_data.iloc[i + 11, 5] / self.stock_data.iloc[i + 10, 5])
            test_data.append(y)
        # test = pd.DataFrame(test_data)
        # test.to_csv("preprocessing/test_data.csv")

    def make_test_data(self):
        test_stock = []
        # stock_data_test = pd.read_csv("stock_data_test.csv", index_col=0)

        for i in range((len(self.stock_data) // 10) * 10 - 11):
            l = self.stock_data.iloc[i-11, 5]
            test_stock.append(l)
            test = pd.DataFrame(test_stock)
            test.to_csv("preprocessing/test_stock.csv")

        stock_test_data = np.array(test_stock)[int(self.feature_split*len(test_stock) +
                                               self.split*(1-self.feature_split)*len(test_stock)):]
        stock = pd.DataFrame(stock_test_data, index=None)
        stock.to_csv("stock_data_test.csv")

        # print(train_data[1:5])
        # print(test_data[1:5])
        # plt.plot(train_data[1])
        # plt.show()



class AutoEncoder:
    def __init__(self, encoding_dim,testing,in_,in_test,out_,log_train):
        self.encoding_dim = encoding_dim
        self.inp = in_
        self.inp_test = in_test
        self.out = out_
        self.testing = testing
        self.log_train = log_train

    def build_train_model(self, input_shape, encoded1_shape, encoded2_shape, decoded1_shape, decoded2_shape):
        input_data = Input(shape=(1, input_shape))

        encoded1 = Dense(encoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(input_data)
        encoded2 = Dense(encoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded1)
        encoded3 = Dense(self.encoding_dim, activation="relu", activity_regularizer=regularizers.l2(0))(encoded2)
        decoded1 = Dense(decoded1_shape, activation="relu", activity_regularizer=regularizers.l2(0))(encoded3)
        decoded2 = Dense(decoded2_shape, activation="relu", activity_regularizer=regularizers.l2(0))(decoded1)
        decoded = Dense(input_shape, activation="sigmoid", activity_regularizer=regularizers.l2(0))(decoded2)

        autoencoder = Model(inputs=input_data, outputs=decoded)

        
        encoder = Model(input_data, encoded3)

        # Now train the model using data we already preprocessed
        autoencoder.compile(loss="mean_squared_error", optimizer="adam")

        train = pd.read_csv(self.inp, index_col=0)
        ntrain = np.array(train)
        train_data = np.reshape(ntrain, (len(ntrain), 1, input_shape))

        # print(train_data)
        # autoencoder.summary()
        if self.testing:
            encoder = load_model("models/encoder.h5")
            encoder.compile(loss="mean_squared_error",optimizer="adam")
        else:
            autoencoder.fit(train_data, train_data, epochs=1000)
            encoder.save("models/encoder.h5")

        test = pd.read_csv(self.inp_test, index_col=0)
        ntest = np.array(test)
        test_data = np.reshape(ntest, (len(ntest), 1, 55))

        print(autoencoder.evaluate(test_data, test_data))
        # pred = np.reshape(ntest[1], (1, 1, 75))
        # print(encoder.predict(pred))

        log_train = pd.read_csv(self.log_train, index_col=0)
        coded_train = []
        for i in range(len(log_train)):
            data = np.array(log_train.iloc[i, :])
            values = np.reshape(data, (1, 1, 55))
            coded = encoder.predict(values)
            shaped = np.reshape(coded, (20,))
            coded_train.append(shaped)

        train_coded = pd.DataFrame(coded_train)
        train_coded.to_csv(self.out)


def nnmodel(epochs, regularizer1, regularizer2,encoded_train,encoded_test,train_y,log_test,test_price,predicted,price,return_acc):

    train_data = np.array(pd.read_csv(encoded_train, index_col=0))
    # length = len(train_data)
    train_data = np.reshape(train_data, (len(train_data), 20))
    print(np.shape(train_data))
    test_data = np.array(pd.read_csv(encoded_test, index_col=0))
    test_data = np.reshape(test_data, (len(test_data), 20))
    train_y = np.array(pd.read_csv(train_y, index_col=0))
    test_y = np.array(pd.read_csv(log_test, index_col=0))
    price = np.array(pd.read_csv(test_price, index_col=0))

    model = kr.models.Sequential()
    # model.add(kl.Dense(50, activation="sigmoid", activity_regularizer=kr.regularizers.l2(0)))
    model.add(kl.Dense(20, input_dim=20, activation="tanh", activity_regularizer=kr.regularizers.l2(regularizer1)))
    model.add(kl.Dense(20, activation="tanh", activity_regularizer=kr.regularizers.l2(regularizer1)))
    model.add(kl.Dense(20, activation="tanh", activity_regularizer=kr.regularizers.l2(regularizer2)))
    # model.add(kl.Dense(100))
    model.add(kl.Dense(1))

    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(train_data, train_y, epochs=epochs)
    model.save("models/final_model.h5")
    predicted_data = []
    predicted_price = []
    for i in range(len(test_data)):
        prediction = model.predict(np.reshape(test_data[i], (1, 20)))
        predicted_data.append(prediction)
        price_pred = np.exp(prediction)*price[i]
        predicted_price.append(price_pred)
        # print(test_data[i])

    # print(model.evaluate(test_data, test_y))
    pd.DataFrame(np.reshape(predicted_price, (len(predicted_price, )))).to_csv(predicted)
    pd.DataFrame(price).to_csv(price)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(predicted_data)), np.reshape(test_y, (len(test_y))),
             np.reshape(predicted_data, (len(predicted_data))))
    plt.title("Prediction vs Actual")
    plt.ylabel("Log Return")

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(predicted_price)), np.reshape(price, (len(price))),
             np.reshape(predicted_price, (len(predicted_price))))
    plt.xlabel("Time stamp")
    plt.ylabel("Market Price")
    plt.show()

    price_r_score = r2_score(np.reshape(predicted_price, (len(predicted_price))), price)
    return_r_score = r2_score(np.reshape(predicted_data, (len(predicted_data))), test_y)
    price_mse = mean_squared_error(np.reshape(predicted_price, (len(predicted_price))), price)
    return_mse = mean_squared_error(np.reshape(predicted_data, (len(predicted_data))), test_y)

    print(f"Regularizer for 1: {regularizer1} \nRegularizer for 2: {regularizer2} \nEpochs: {epochs}")
    print(f"Predicted Price r^2 value: {price_r_score} \nPredicted return r^2 value: {return_r_score}"
          f"\nPredict Price MSE: {price_mse} \nPredicted Return MSE: {return_mse}")
    dataset = []
    values = np.array([regularizer1, regularizer2, epochs, price_r_score, return_r_score, price_mse, return_mse])
    dataset.append(values)
    dataset = pd.DataFrame(dataset, columns=["regularizer1", "regularizer2", "epochs", "price_r_score", "return_r_score", "price_mse", "return_mse"])
    # print(dataset)
    accuracy = []
    for i in range(len(price)-1):
        acc = 100 - (np.abs(predicted_price[i] - price[i+1]))/price[i+1] * 100
        accuracy.append(acc)
    average = np.mean(accuracy)
    std = np.std(accuracy)
    ret_acc = []
    for i in range(len(test_y)-1):
        if test_y[i] != 0:
            acc = 100 - (np.abs(predicted_data[i] - test_y[i]))/test_y[i] * 100
            ret_acc.append(acc)
    ret_avg = np.mean(ret_acc)
    ret_std = np.std(ret_acc)
    pd.DataFrame(np.reshape(ret_acc, (len(ret_acc, )))).to_csv(return_acc)
    prediction = np.exp(model.predict(np.reshape(test_data[-2], (1, 20))))*price[-2]
    print(prediction)

    return dataset, average, std



# if __name__ == "__main__":
preprocess = PreProcessing(0.8, 0.25,"stock_data.csv","preprocessing/rbm_train.csv","preprocessing/rbm_test.csv","preprocessing/log_train.csv")
preprocess.make_wavelet_train()
preprocess.make_test_data()

# if __name__ == "__main__":
autoencoder = AutoEncoder(20,True,"preprocessing/rbm_train.csv","preprocessing/rbm_test.csv","features/autoencoded_data.csv","preprocessing/log_train.csv")
autoencoder.build_train_model(55, 40, 30, 30, 40)

# if __name__ == "__main__":
dataset, average, std = nnmodel(500, 0.05, 0.01,"features/autoencoded_data.csv","60_return_forex/encoded_return_test_data.csv","preprocessing/log_train.csv","forex_y/log_test_y.csv","forex_y/test_price.csv","60_return_forex/predicted_price.csv","60_return_forex/price.csv","60_return_forex/ret_acc.csv")
print(f"Price Accuracy Average = {average} \nPrice Accuracy Standard Deviation = {std}")
