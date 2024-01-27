import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd

random.seed(20)

def random_integer(start,end):
    return random.randint(start, end)

def custom_sigmoid(x,a,b):
    return a+(b-a)*(2*((1/(1+np.exp(-x)))-0.5))

def propagation_delay(node1, node2):
    point1 = np.array([node1.x, node2.x])
    point2 = np.array([node2.x, node2.y])
    distance = np.linalg.norm(point2 - point1)
    return distance / (3 * 10**8)

def total_node(server, edge_device, base=True, without_la= False):
    plt.figure()
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    if(base == True):
        plt.scatter(server.x, server.y, color='red', marker='s') #server node
        plt.text(server.x, server.y, server.name, ha='center', va='bottom', fontsize=8)

        for edge in edge_device:
            plt.scatter(edge.x, edge.y, color='gray', marker='o') #client node
            plt.text(edge.x, edge.y, edge.name, ha='center', va='bottom', fontsize=8)
            plt.plot([server.x, edge.x],[server.y, edge.y], color= 'gray', lw = 0.5)
            if len(edge.agent_list) > 0 and without_la == False:
                for i, agent in enumerate(edge.agent_list):
                    plt.scatter(agent.x, agent.y, color='gray', marker='X') #agent node
                    plt.text(agent.x, agent.y, 'agent'+str(i+1), ha='center', va='bottom', fontsize=8)
                    plt.plot([edge.x, agent.x], [edge.y, agent.y], color='gray', lw=0.5)

    else:
        plt.scatter(server.x, server.y, color='red', marker='s')  # server node
        plt.text(server.x, server.y, server.name, ha='center', va='bottom', fontsize=8)

        for edge in edge_device:
            plt.scatter(edge.x, edge.y, color='blue', marker='o')  # client node
            plt.text(edge.x, edge.y, edge.name, ha='center', va='bottom', fontsize=8)
            plt.plot([server.x, edge.x], [server.y, edge.y], color='blue', lw=0.7)

            if len(edge.agent_list) > 0:
                for i, agent in enumerate(edge.agent_list):
                    if(agent.use_status == 0):
                        plt.scatter(agent.x, agent.y, color='gray', marker='X')  # agent node
                        plt.plot([edge.x, agent.x], [edge.y, agent.y], color='gray', lw=0.7, linestyle= 'dotted')
                    else:
                        plt.scatter(agent.x, agent.y, color='green', marker='X')  # agent node
                        plt.plot([edge.x, agent.x], [edge.y, agent.y], color='green', lw=0.7)

                    plt.text(agent.x, agent.y, 'agent'+str(i+1), ha='center', va='bottom', fontsize=8)

        plt.title('Device Locations')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('./output_basic/Device_Location' + '.png')
        plt.show()

def gif_show(images):

    fig, ax = plt.subplots()
    img = ax.imshow(images[0], animated=True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    def update(i):
        img.set_array(images[i])
        return img,

    animation_fig = animation.FuncAnimation(fig, update, frames=len(images), interval=800, blit=True, repeat_delay=10, )
    plt.show()

def output_dataframe(history):
    df_round = pd.DataFrame()
    r = 1
    column_names = ['max_roundtime', 'total_roundtime', 'acc_aggre']
    all_round_dict = {}

    for now_round in history:
        round_dict = {'max_roundtime': now_round.max_roundtime, 'total_roundtime': now_round.total_roundtime, 'acc_aggre': now_round.acc_aggre}
        for now_target in now_round.RoundList:
            index = 'Round' + str(r)
            now_target_dict = vars(now_target)
            del now_target_dict['name']
            round_dict.update(now_target_dict)
            all_round_dict.update(now_target_dict)

            df_target = pd.DataFrame(round_dict, index=[index])

        df_round = pd.concat([df_round, df_target], join='outer', axis=0)

        r = r + 1

    # client1, client2 ... sorting
    round_dict_keys = all_round_dict.keys()
    round_dict_keys = sorted(round_dict_keys)
    round_dict_keys.extend(column_names)
    df_round = df_round.reindex(columns=round_dict_keys)

    return df_round

def accuracy_graph(history,file_name_graph):
    x_time = 0
    X_list = []
    y = []
    round_num = len(history)
    for r in history:
        totaltime = r.total_roundtime  # total round time
        x_time = x_time + totaltime
        X_time = round(x_time, 2)
        X_list.append(X_time)
        y.append(r.acc_aggre)

    X = np.arange(1, round_num + 1, 1)

    plt.subplot(2, 1, 1)  # nrows=2, ncols=1, index=1
    plt.plot(X, y, 'o-')
    plt.title('X=round')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)  # nrows=2, ncols=1, index=2
    plt.plot(X_list, y, '.-')
    plt.title('X=time')
    plt.xlabel('time (s)')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(file_name_graph + '.png')
    plt.show()

def roundtime_graph(history1, history2,path_type):
    round_num = range(len(history1))
    history1_roundtime = []
    history2_roundtime = []
    history1_totalroundtime = 0
    history2_totalroundtime = 0

    for r1 in history1:
        round_time1 = r1.total_roundtime
        history1_totalroundtime += round_time1
        history1_roundtime.append(history1_totalroundtime)
    for r2 in history2:
        round_time2 = r2.total_roundtime
        history2_totalroundtime += round_time2
        history2_roundtime.append(history2_totalroundtime)

    plt.figure()
    plt.plot(round_num, history1_roundtime,label= 'Without LA', marker= 'o')
    plt.plot(round_num, history2_roundtime,label= 'With LA', marker= 'o')


    plt.title(f'Roundtime comparision in {path_type}')
    plt.xlabel('round')
    plt.ylabel('Roundtime')
    plt.legend()

    plt.savefig(f'./output_basic/{path_type}/Roundtime comparision.png')
    plt.show()

def paths_maxdelay(transmission_device, receive_device, has_dataset= False):

    p1 = random.uniform(0.8,1) #path1_trans_p
    p2 = random.uniform(0.8,1) #path2_trans_p

    comm1, comm2 = transmission_device.comm1, transmission_device.comm2
    pd = propagation_delay(transmission_device, receive_device)
    Data = transmission_device.model_size
    if (has_dataset == True):
        Data = transmission_device.dataset_size + transmission_device.model_size

    #ratio = 0.5
    ratio = ((1/(comm2*p2)) - ((pd / Data)*(1/p1 - 1/p2))) * ((comm1*p1*comm2*p2)/(comm1*p1 + comm2*p2))

    path1_data, path2_data = Data*ratio, Data*(1-ratio)
    path1_delay, path2_delay = path1_data/comm1, path2_data/comm2

    path1_totaldelay = (path1_delay + pd) * (1/p1)
    path2_totaldelay = (path2_delay + pd) * (1/p2)
    # print('From',transmission_device.name,'to',receive_device.name)
    # print('path1_totaldelay',path1_totaldelay)
    # print('path2_totaldelay',path2_totaldelay)

    return max(path1_totaldelay, path2_totaldelay)

def delay(transmission_device,receive_device, has_dataset= False):
    p = random.uniform(0.8,1)
    pd = propagation_delay(transmission_device, receive_device)
    comm = transmission_device.comm1
    Data = transmission_device.model_size
    if (has_dataset == True):
        Data = transmission_device.dataset_size + transmission_device.model_size

    delay = ((Data/comm) + pd) * (1/p)
    # print('From', transmission_device.name, 'to', receive_device.name)
    # print(delay)
    return delay
