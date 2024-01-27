class RoundRecord():

    def __init__(self):
        self.RoundList = []


class TargetRecord():

    name = '' #target device name
    deploy_time = 0
    iteration = 0
    client_time = 0
    clientUsingLA = ''
    clientUsingLA_time = 0
    clientUsingDevice = ''
    clientUsingDevice_Finaltime = 0

    def __init__(self, name, deploy_time, iteration, client_time, clientUsingLA, clientUsingLA_time, clientUsingDevice, client_Finaltime):
        self.name = name
        setattr(self, f"{name}_deploytime", deploy_time)
        setattr(self, f"{name}_iteration", iteration)
        setattr(self, f"{name}_time", client_time)
        setattr(self, f"{name}_la", clientUsingLA)
        setattr(self, f"{name}_la_time", clientUsingLA_time)
        setattr(self, f"{name}_usingDevice", clientUsingDevice)
        setattr(self, f"{name}__Finaltime", client_Finaltime)