import pandas as pd
import random
import helper
import numpy as np
import matplotlib.pyplot as plt


class server:
    state = 'server'
    name = 'global'
    x = 0
    y = 0
    model_size = 0
    comm1 = 0
    comm2 = 0

    def __init__(self, model_size, comm1, comm2 , x=None, y=None):

        if x is not None:
            self.x = x
        else:
            self.x = random.randint(0, 100)

        if y is not None:
            self.y = y
        else:
            self.y = random.randint(0, 100)

        self.model_size = model_size

        self.comm1 = comm1 / 8
        self.comm2 = comm2 / 8

    def info(self):
        print(f'-------{self.state} create-------')
        attribute = {'model size:':self.model_size, 'communication':[self.comm1, self.comm2]}

        for attribute_name, attribute_value in attribute.items():
            if isinstance(attribute_value, list):
                print(attribute_name + ":")
                for value in attribute_value:
                    print(f'    {value:.2f}')
            else:
                print(f'{attribute_name:10s}{attribute_value:.2f}')

    def DataFrame_global(self):
        df_global = pd.DataFrame(vars(self), index=['global'])
        return df_global

    def node_loc(self):
        #plt.figure()
        plt.xlim(-10, 110)
        plt.ylim(-10, 110)
        plt.scatter(self.x, self.y, color='red', marker='s', s=70 )


class edge( ):
    name = ''
    model_size = 0
    dataset_size = 0
    compute_rate = 0
    learning_time = 0
    x = 0
    y = 0
    comm1 = 0
    comm2 = 0

    def __init__(self, name, model_size, comm1, comm2, dataset_size, compute_rate,x=None, y=None):
        self.name = name
        self.model_size = model_size
        self.dataset_size = dataset_size
        self.compute_rate = compute_rate
        self.learning_time = self.dataset_size * self.compute_rate # ms -> s

        self.comm1 = comm1 / 8
        self.comm2 = comm2 / 8

        if x is not None:
            self.x = x
        else:
            self.x = random.randint(0, 100)

        if y is not None:
            self.y = y
        else:
            self.y = random.randint(0, 100)

        self.agent_list = []

    def info(self):
        attribute = {'data size': self.dataset_size, 'performance of device':self.compute_rate, 'Training time': self.learning_time}
        for attribute_name, attribute_value in attribute.items():
            print(f'{attribute_name:10s}{attribute_value:.2f}')

    def DataFrame_edge(self):
        now_client_dict = vars(self.copy())
        now_client_dict.pop('agent_list', None)
        df_edge = pd.DataFrame(now_client_dict, index=[now_client_dict['name']])
        df_edge.drop('name', axis=1, inplace=True)
        return df_edge

    def copy(self):
        copied_edge = edge(self.name, self.model_size, self.comm1 * 8, self.comm2 * 8, self.dataset_size, self.compute_rate)
        copied_edge.x = self.x
        copied_edge.y = self.y
        copied_edge.agent_list = self.agent_list.copy()
        return copied_edge

    def node_loc(self):
        plt.scatter(self.x, self.y, color='blue', marker='o',s = 70)


class learning_agent():
    name = ''
    model_size = 0
    dataset_size = 0
    compute_rate = 0
    learning_time = 0
    x = 0
    y = 0
    comm1 = 0
    comm2 = 0

    def __init__(self, name, model_size, comm1, comm2, dataset_size, compute_rate, p_idle, client_x = None, client_y = None,
                 x = None, y = None):

        self.name = name
        self.model_size = model_size
        self.dataset_size = dataset_size
        self.compute_rate = compute_rate
        self.learning_time = self.dataset_size * self.compute_rate

        self.comm1 = comm1 / 8
        self.comm2 = comm2 / 8

        if x is not None:
            self.x = x
        else:
            self.x = random.randint(client_x-10, client_x+10)  # near the client (+- 10)
            self.use_status = helper.random_integer(0, 1)  # 0=false / 1=true

        if y is not None:
            self.y = y
        else:
            self.y = random.randint(client_y-10, client_y+10)

        if(p_idle>=0 and p_idle<=0.3):
            self.use_status = 0
        else:
            self.use_status = 1

    def DataFrame_agent(self):
        now_agent_dict = vars(self)
        df_agent = pd.DataFrame(now_agent_dict, index=[now_agent_dict['name']])
        df_agent.drop('name', axis=1, inplace=True)  # name 삭제
        return df_agent

    def node_loc(self):
        plt.scatter(self.x, self.y, color='green', marker='X',s=70)

