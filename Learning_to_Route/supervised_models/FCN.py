from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers import Dense

from Learning_to_Route.supervised_models.network_graph import NetworkGraph
from Learning_to_Route.supervised_models.model_input_generation import get_x_y_data, sample_tm_cyclic
from consts import Consts
from Learning_to_Route.supervised_models.trainer import trainer_function


def get_model_fcn(num_nodes, history_size=10):
    model = Sequential()

    model.add(Flatten(input_shape=(history_size, num_nodes, num_nodes)))
    model.add(Dense(150, activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    #     model.add(Dense(128, activation='sigmoid'))
    #     model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(num_nodes ** 2, activation='relu'))
    return model


if __name__ == "__main__":
    num_nodes = 16
    avg_degree = int(num_nodes / 2)
    graph = NetworkGraph(num_nodes=num_nodes, avg_degree=avg_degree)
    p = 0.3
    num_histories = 10
    history_size = 10
    max_history_len = 500
    X_train, Y_train = get_x_y_data(graph, p, num_histories=num_histories,
                                    get_xy=sample_tm_cyclic,
                                    tm_type=Consts.GRAVITY,
                                    max_history_len=max_history_len,
                                    history_window=history_size, )

    X_test, Y_test = get_x_y_data(graph, p, num_histories=3,
                                  get_xy=sample_tm_cyclic,
                                  tm_type=Consts.GRAVITY,
                                  max_history_len=max_history_len,
                                  history_window=history_size)

    model = get_model_fcn(num_nodes=num_nodes, history_size=history_size)

    trainer_function(model=model, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, num_epocs=100)
