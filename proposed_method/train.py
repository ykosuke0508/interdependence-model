from pathlib import Path 
import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from sklearn.preprocessing import MinMaxScaler

DATASET_BASE_PATH = "../download_datasets/data_home"

def adam(loss, all_params, alpha=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    updates = []
    all_grads = theano.grad(loss, all_params)
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g
        v = b2*v_previous + (1 - b2)*g**2
        m_hat = m / (1-b1**t)
        v_hat = v / (1-b2**t)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta))
    updates.append((t, t + 1.))
    return updates

def train_idd(features, labels, Wx=None, Wy=None, yhat=None, gamma_=0.1, rho_=1.0):
    # ../Thesis/Masters_Thesis/masters_paper/mthesis_yoshimura_for_wiki.pdf
    n_features = features.shape[1]
    n_labels = labels.shape[1]
    n_samples = features.shape[0]

    # features symbol
    x = T.matrix(name="x")
    x_T = x.dimshuffle((1,0))

    # labels symbol
    y = T.matrix(name="y")
    y_T = y.dimshuffle((1,0))

    if Wx is None:
        Wx = theano.shared(value=np.ones([n_labels, n_features]),name="Wx")
    
    if Wy is None:
        A = np.random.random([n_labels, n_labels])
        Wy = theano.shared(value=A - np.diag(np.diag(A)),name="Wy")
    
    if yhat is None:
        yhat = theano.shared(value=np.random.uniform(0,1,[n_labels, n_samples]),name="yhat")

    all_params = [Wx, Wy, yhat]
    fx = 1 / (1 + T.exp(-T.dot(Wx, x_T) - T.dot(Wy,yhat)))

    loss = -T.sum(y_T*T.log(fx) + (1 - y_T)*T.log(1 - fx)) + gamma_*(T.sum(Wx**2) + T.sum(Wy**2)) + rho_*T.sum((fx - yhat) ** 2)
    updates = adam(loss, all_params)
    f = theano.function(inputs=[x,y], outputs=[loss], updates=updates)

    loss_score = f(features, labels)[0]
    for i in range(100000):
        previous_loss_score = loss_score
        loss_score = f(features, labels)[0]
        if (previous_loss_score - loss_score) / previous_loss_score < 1e-5:
            break

        if i % 1000 == 0:
            print(f"Epoch {i}: {loss_score}")
    return Wx, Wy, yhat


def test(dataset_name):
    path = Path(DATASET_BASE_PATH)
    feature_data_file = path / f"{dataset_name}/{dataset_name}_features.csv"
    label_data_file = path / f"{dataset_name}/{dataset_name}_labels.csv"

    feature_data = pd.read_csv(feature_data_file).values
    feature_data = np.insert(feature_data, 0, values=1, axis=1)
    label_data = pd.read_csv(label_data_file).values

    ms = MinMaxScaler()
    feature_data= ms.fit_transform(feature_data)
    #for gamma in [0.01, 0.1, 1, 10, 100]:
    #    train_idd(feature_data, label_data, gamma_=gamma)

    hyper_params = {"gamma_": 0.1, "rho_": 1}
    Wx, Wy, yhat = train_idd(feature_data, label_data, gamma_=hyper_params["gamma_"], rho_=hyper_params["rho_"])
    hyper_params["rho_"] = 10
    Wx, Wy, yhat = train_idd(feature_data, label_data, Wx=Wx, Wy=Wy, yhat=yhat, gamma_=hyper_params["gamma_"], rho_=hyper_params["rho_"])
    hyper_params["rho_"] = 100
    Wx, Wy, yhat = train_idd(feature_data, label_data, Wx=Wx, Wy=Wy, yhat=yhat, gamma_=hyper_params["gamma_"], rho_=hyper_params["rho_"])
    print(yhat.get_value())

def main():
    test("scene")

if __name__ == "__main__":
    main()
