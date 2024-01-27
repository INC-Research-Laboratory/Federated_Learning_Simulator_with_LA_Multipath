import pandas as pd
import random
import matplotlib.pyplot as plt
import yaml
import os
from PIL import Image

import device
import helper

# path setting
with open('yaml/path_setting.yaml') as path_yaml:
    path_info = yaml.load(path_yaml, Loader=yaml.FullLoader)
    root_path = path_info['root_path']

os.makedirs(root_path, exist_ok = True)

## Setting
min_acc = 0  # minimum accuracy
max_acc = 0  # maximum accuracy
p = 0  # weight of local training epoch
q = 0  # weight of participated client

edge_num = 0
round_num = 0
agent_num = 0

with open('./yaml/custom_setting.yaml') as custom_setting_yaml:
    custom_setting = yaml.load(custom_setting_yaml, Loader=yaml.FullLoader)

    setting = custom_setting['setting']

    # parameter
    min_acc = setting['min_acc']
    max_acc = setting['max_acc']
    p = setting['w_of_epoch']
    q = setting['w_of_client']

    edge_num = setting['edge_num']
    round_num = setting['round_num']
    model_name = setting['model']

    server_spec = custom_setting['server_spec']
    edge_device_spec = custom_setting['edge_device_spec']

with open('./yaml/model.yaml') as model_yaml:
    model_load = yaml.load(model_yaml, Loader=yaml.FullLoader)
    model_load = model_load['model']

with open('./yaml/device.yaml') as device_yaml:
    device_load = yaml.load(device_yaml, Loader=yaml.FullLoader)
    device_gpu = device_load['GPU']
    device_soc = device_load['SoC']

with open('./yaml/communication.yaml') as communication_yaml:
    communication_load = yaml.load(communication_yaml, Loader=yaml.FullLoader)
    communication_load = communication_load['communication']

with open('./yaml/dataset.yaml') as dataset_yaml:
    dataset_load = yaml.load(dataset_yaml, Loader=yaml.FullLoader)
    dataset_load = dataset_load['dataset']

def Device():
    print(f'[Step1] Device Information')

    ###################################### 1. Server
    # server location
    server_location = server_spec['location']
    server_x = server_location['x']
    server_y = server_location['y']

    # model info
    model_spec = model_load[model_name]
    model_size = model_spec['size']

    # server comm info
    server_comm1_name = server_spec['communication_1']
    server_comm1 = communication_load[server_comm1_name]
    server_comm1_throughput = server_comm1['throughput']

    server_comm2_name = server_spec['communication_2']
    server_comm2 = communication_load[server_comm2_name]
    server_comm2_throughput = server_comm2['throughput']

    print(f'  1. Server Device Information')
    print(f'     Global Model: {model_name}')
    print(f'     Model Spec: {model_spec}')
    print(f'     Communication: (1) {server_comm1_name}, {server_comm1_throughput}')
    print(f'                    (2) {server_comm2_name}, {server_comm2_throughput}')

    # Create server object
    server_global = device.server(model_size, server_comm1_throughput, server_comm2_throughput,
                                          server_x, server_y)

    # server dataframe
    df_Device = server_global.DataFrame_global()

    ############################################# 2. Client(Edge device)
    print(f'  2. Edge Device({edge_num} Client) Information')

    edge_device = []  # 생성하는 device 객체를 담는 리스트 생성

    df_agent = pd.DataFrame()  # agent dataframe 틀 생성, 뒤에서 concat

    for c in range(1, edge_num + 1):
        print('')
        print(f'     * Client{c} Information')
        client_name = 'Client' + str(c)

        ### client spec
        client_spec = edge_device_spec['client' + str(c) + '_spec']

        # dataset
        dataset_name = client_spec['dataset']
        dataset_spec = dataset_load[dataset_name]
        dataset_size = dataset_spec['volume']
        print(f'       Dataset: {dataset_spec}')

        # client location
        client_loc = client_spec['location']
        client_x = client_loc['x']
        client_y = client_loc['y']

        # device
        client_device_name = client_spec['device']
        if client_device_name in list(device_gpu.keys()):  # gpu에 포함되면
            client_device_spec = device_gpu[client_device_name]
        elif client_device_name in list(device_soc.keys()):  # soc에 포함되면
            client_device_spec = device_soc[client_device_name]

        client_compute_rate = client_device_spec[model_name]
        print(f'       Coumputing Power: {client_device_name}, {client_compute_rate} ms')

        # client communication info
        client_comm1_name = client_spec['communication_1']
        client_comm1_spec = communication_load[client_comm1_name]
        client_comm1_throughput = client_comm1_spec['throughput']

        client_comm2_name = client_spec['communication_2']
        client_comm2_spec = communication_load[client_comm2_name]
        client_comm2_throughput = client_comm2_spec['throughput']
        print(f'       Communication: (1) {client_comm1_name}, {client_comm1_throughput}')
        print(f'                      (2) {client_comm2_name}, {client_comm2_throughput}')

        # Create client object
        now_client = device.edge(client_name, model_size, client_comm1_throughput, client_comm2_throughput,
                                            dataset_size, client_compute_rate, client_x, client_y)

        edge_device.append(now_client)

        ############################################# 3. Device (Learning Agent)
        agent_num = client_spec['agent_num']
        print(f'        * {client_name} {agent_num} Learning Agent Information')
        for a in range(1, agent_num + 1):
            print(f'       ** Learning Agent {a} Info')
            agent_name = now_client.name + '_agent' + str(a)
            # agent_name = 'agent' + str(a)

            # agent spec
            agent_spec = client_spec['agent' + str(a) + '_spec']

            # agent location
            agent_loc = agent_spec['location']
            agent_x = agent_loc['x']
            agent_y = agent_loc['y']

            # agent idle
            agent_p_idle = random.random()

            # device 설정
            agent_device_name = agent_spec['device']
            if agent_device_name in list(device_gpu.keys()):
                agent_device_spec = device_gpu[agent_device_name]
            elif agent_device_name in list(device_soc.keys()):
                agent_device_spec = device_soc[agent_device_name]

            agent_compute_rate = agent_device_spec[model_name]
            print(f'          Computing Power: {agent_device_name}, {agent_compute_rate} ms')

            # agent communication info
            agent_comm1_name = agent_spec['communication_1']
            agent_comm1_spec = communication_load[agent_comm1_name]
            agent_comm1_throughput = agent_comm1_spec['throughput']

            agent_comm2_name = agent_spec['communication_2']
            agent_comm2_spec = communication_load[agent_comm2_name]
            agent_comm2_throughput = agent_comm2_spec['throughput']

            print(f'          Communication: (1) {agent_comm1_name}, {agent_comm1_throughput}')
            print(f'                         (2) {agent_comm2_name}, {agent_comm2_throughput}')

            # Create agent object
            now_agent = device.learning_agent(agent_name, model_size, agent_comm1_throughput,
                                                         agent_comm2_throughput,dataset_size, agent_compute_rate,
                                                         agent_p_idle,x=agent_x, y=agent_y)

            now_client.agent_list.append(now_agent)

            # agent Dataframe
            df_now_agent = now_agent.DataFrame_agent()
            df_agent = pd.concat([df_agent, df_now_agent], axis=0)  # df_agent와 df_now_agent를 concatenate

        # client Dataframe
        df_edge = now_client.DataFrame_edge()

        df_Device = pd.concat([df_Device, df_edge], axis=0)

    df_Device = pd.concat([df_Device, df_agent], axis=0)
    df_Device.to_csv(root_path + 'Device_spec.csv')

    # node location visualize
    helper.total_node(server_global, edge_device, base=False)

    return server_global, edge_device

def target_list(edge_device, target_device_num=2): #usingClient_list
    ## Specify which clients to select by round
    total_target_list = [sorted(random.sample(edge_device, target_device_num), key=lambda x: x.name) for _ in range(round_num)]

    return total_target_list

def round_(server_global, edge_device, total_target_list, multipaths, without_la):
    import Record

    history = []  # List to append Round objects
    acc_x_total = 0
    images = []  # List to store png img

    for r in range(1, round_num + 1):
        if multipaths:
            if without_la:
                now_path = path_info['multi']['without_la']
                os.makedirs(now_path+'used_node', exist_ok = True)

                image_path = now_path + f'used_node/round{r}.png'
                gif_path = now_path + 'used_node/animation_without_la.gif'
                output_file_name = now_path + 'output_without_la'
                accgraph_file_name = now_path +'accuracy_Graph_without_la'

            else:
                now_path = path_info['multi']['with_la']
                os.makedirs(now_path + 'used_node', exist_ok=True)

                image_path = now_path + f'used_node/round{r}.png'
                gif_path = now_path + 'used_node/animation_with_la.gif'
                output_file_name = now_path + 'output_with_la'
                accgraph_file_name = now_path + 'accuracy_Graph_with_la'

        else: #singlepath
            if without_la:
                now_path = path_info['single']['without_la']
                os.makedirs(now_path + 'used_node', exist_ok=True)

                image_path = now_path + f'used_node/round{r}.png'
                gif_path = now_path + 'used_node/animation_without_la.gif'
                output_file_name = now_path + 'FL_output_without_la'
                accgraph_file_name = now_path + 'FL_accuracy_Graph_without_la'

            else:
                now_path = path_info['single']['with_la']
                os.makedirs(now_path + 'used_node', exist_ok=True)

                image_path = now_path + f'used_node/round{r}.png'
                gif_path = now_path + 'used_node/animation_with_la.gif'
                output_file_name = now_path + 'FL_output_with_la'
                accgraph_file_name = now_path + 'FL_accuracy_Graph_with_la'

        acc_x = 0  # for accuracy Calculation
        #output_index = []  # Round1, Round2 ...

        # 생성된 device 그리기 To draw the devices
        if(without_la == True):
            helper.total_node(server_global, edge_device, without_la= True)
        else:
            helper.total_node(server_global, edge_device)

        print(f'  Round {r}')

        ## 1. Edge device Select
        print(f'    1. Edge device Select - Total device: {[device.name for device in edge_device]}')
        now_target_list = total_target_list[r-1]
        print(f'       Selected device: {[device.name for device in now_target_list]}')

        # Create current round object
        now_round = Record.RoundRecord()

        ## 3. Select device local train
        print(f'    2. Local train & Send to server ({len(now_target_list)} device)')
        print(f'        -> Learning time(ms) = learning time(iter 1) * iteration')
        print(f'        -> Device(Edge/Learning Agent) to Server = Learning time(s) + send time(s)')
        print(f'        ---> Edge device to Server : learning time(s) + (edge device) send time(s)')
        print(
            f'        ---> Learning agent to Server : (edge device) send time(s) + learning time(s) + (learning agent) send time(s)')

        for index, target_client in enumerate(now_target_list):

            iteration = 500

            # Display used client node
            target_client.node_loc()
            plt.plot([server_global.x, target_client.x], [server_global.y, target_client.y], color='b', lw=0.8)


            # deploy : send global model from sever to edge device
            # (소요 시간(s) = Global model size(MB) / Communication(MB/s)
            if multipaths:
                deploy_time = helper.paths_maxdelay(server_global,target_client)
                edge_server_delay = helper.paths_maxdelay(target_client,server_global)
            else:
                deploy_time = helper.delay(server_global, target_client)
                edge_server_delay = helper.delay(target_client, server_global)

            print(f'        * server to {target_client.name} Model Deploy time : {deploy_time}')

            # without learning agent : train + send(edge -> server)
            edge_time = target_client.learning_time * iteration / 1000 \
                        + edge_server_delay

            print(f'        * {target_client.name} iteration: {iteration}')
            print(f'            Edge device to Server : {target_client.learning_time * iteration / 1000:.3f} + 'f'{edge_server_delay:.3f} '
                  f'= {edge_time:.3f} s')

            # use_agent : used learning agent
            # use_agent_time : Time spent using learning agent
            use_agent_time = 0
            use_agent = 'None'

            if without_la == False and len(target_client.agent_list) > 0:  # target device has learning agents
                for i in range(len(target_client.agent_list)):

                    use_status = target_client.agent_list[i].use_status  # use_status (0 or 1)

                    # learning agent is available : send(edge -> la) + training + send(la -> edge -> server)
                    if use_status == 1:

                        if multipaths:
                            edge_agent_delay = helper.paths_maxdelay(target_client, target_client.agent_list[i],
                                                                     has_dataset=True)
                            agent_edge_delay = helper.paths_maxdelay(target_client.agent_list[i], target_client)
                        else:
                            edge_agent_delay = helper.delay(target_client, target_client.agent_list[i],
                                                                     has_dataset=True)
                            agent_edge_delay = helper.delay(target_client.agent_list[i], target_client)

                        now_use_agent_time = edge_agent_delay \
                                            + target_client.agent_list[i].learning_time * iteration / 1000 \
                                            + agent_edge_delay + edge_server_delay

                        # Time comparison between learning agents
                        if use_agent_time == 0:  # (one la) or (the first la)
                            use_agent_time = now_use_agent_time
                            use_agent = target_client.agent_list[i]

                        else:
                            if use_agent_time > now_use_agent_time:  # Select shorter time
                                use_agent_time = now_use_agent_time
                                use_agent = target_client.agent_list[i]


                print(f'            Learning Agent to Server : {use_agent.name}, agent_time:{use_agent_time:.3f} s')

            # accuracy: x = round_dataset * round_iteration -> np.sum(x)
            acc_x += target_client.dataset_size * iteration * p


            # Time of edge device: Training time + sent time(egde -> server)
            # Time of LA: send time (edge -> la) + Training time + send time(la -> edge -> server)

            # If there is No learing agent
            if use_agent_time == 0:
                use_device = target_client
                final_time = edge_time

                target_client.node_loc()
                now_target = Record.TargetRecord(target_client.name, deploy_time, iteration, edge_time, use_agent,
                                                 use_agent_time, use_device.name, final_time)

            # If the learning agent has a longer time than edge device
            elif edge_time <= use_agent_time:
                use_device = target_client
                final_time = edge_time

                target_client.node_loc()
                now_target = Record.TargetRecord(target_client.name, deploy_time, iteration, edge_time, use_agent.name,
                                                 use_agent_time, use_device.name, final_time)

            # If the learning agent has a shorter time than edge device
            else:
                use_device = use_agent
                final_time = use_agent_time

                target_client.node_loc()
                use_agent.node_loc()
                plt.plot([target_client.x, use_agent.x], [target_client.y, use_agent.y], color='green', lw=0.8)

                now_target = Record.TargetRecord(target_client.name, deploy_time, iteration, edge_time, use_agent.name,
                                                 use_agent_time, use_device.name, final_time)

            print(f'            Use_device:{use_device.name} / Send to Server:{final_time:.3f} s')

            now_round.RoundList.append(now_target)

        # Append the round object in the history list
        history.append(now_round)

        # Round time of the last device to complete training and transmitting among the devices that participated in training
        for now_round in history:
            max_roundtime = 0
            sendtime = 0
            deploytime = 0
            for now_target in now_round.RoundList:
                for attr_name in dir(now_target):
                    if attr_name.endswith("_Finaltime"):
                        sendtime = getattr(now_target, attr_name)

                    if attr_name.endswith("_deploytime"):
                        deploytime = getattr(now_target, attr_name)

                    roundtime = sendtime + deploytime
                    if roundtime > max_roundtime:
                        max_roundtime = roundtime


        # total time
        total_time =  max_roundtime  # global 모델을 edge 디바이스에 배포하는 시간 + 엣지디바이스가 학습/전송을 완료하는 시간
        print('')
        print(f'        Total Round time = {total_time:.3f} s')

        # accuracy
        acc_x_weight_sum = acc_x * ((len(now_target_list) * q) / edge_num)

        acc_x_total = acc_x_total + acc_x_weight_sum
        acc_aggre = helper.custom_sigmoid(acc_x_total, min_acc, max_acc)

        print(f'        Aggregation accuracy: {acc_aggre:.3f}')

        setattr(history[r - 1], 'max_roundtime', max_roundtime)
        setattr(history[r - 1], 'total_roundtime', total_time)
        setattr(history[r - 1], 'acc_aggre', acc_aggre)

        # Save the used device node
        plt.title(f'Round{r} used device')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(image_path)
        # plt.show()
        plt.close()

        image = Image.open(image_path)
        images.append(image)

    ## images to gif
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)

    ## output dataframe
    df_round = helper.output_dataframe(history)
    df_round_copy = df_round.transpose()
    df_round.to_csv(output_file_name + '.csv')
    df_round_copy.to_csv(output_file_name + '_trans.csv')

    helper.accuracy_graph(history, accgraph_file_name)

    return history, images, df_round



def simulator_multipaths(server,edge_device,usingClient_list,multipaths):
    without_la_history, without_la_images, without_la_dataframe = round_(server, edge_device, usingClient_list, multipaths, without_la=True)
    helper.gif_show(without_la_images)

    with_la_history, with_la_images, with_la_dataframe = round_(server, edge_device, usingClient_list, multipaths, without_la=False)
    helper.gif_show(with_la_images)

    print("MULTI PATHS")
    print(f"Total Round time(Without la) : {without_la_dataframe['total_roundtime'].sum():.3f} s")
    print(f"Accuracy(Without la) : {without_la_dataframe['acc_aggre'][-1]:.3f}")

    print(f"Total Round time(With la) : {with_la_dataframe['total_roundtime'].sum():.3f} s")
    print(f"Accuracy(With la) : {with_la_dataframe['acc_aggre'][-1]:.3f}")

    helper.roundtime_graph(without_la_history, with_la_history,'multipaths')


def simulator_single(server,edge_device,usingClient_list,singlepath):
    without_la_history, without_la_images, without_la_dataframe = round_(server, edge_device, usingClient_list,singlepath,
                                                                         without_la=True)
    helper.gif_show(without_la_images)

    with_la_history, with_la_images, with_la_dataframe = round_(server, edge_device,usingClient_list, singlepath, without_la=False)
    helper.gif_show(with_la_images)

    print("SINGLE PATH")
    print(f"Total Round time(Without la) : {without_la_dataframe['total_roundtime'].sum():.3f} s")
    print(f"Accuracy(Without la) : {without_la_dataframe['acc_aggre'][-1]:.3f}")

    print(f"Total Round time(With la) : {with_la_dataframe['total_roundtime'].sum():.3f} s")
    print(f"Accuracy(With la) : {with_la_dataframe['acc_aggre'][-1]:.3f}")

    helper.roundtime_graph(without_la_history, with_la_history,'singlepath')