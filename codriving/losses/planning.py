from typing import Tuple, Dict, Any
import numpy as np
import torch
from torch import nn

from codriving import CODRIVING_REGISTRY


@CODRIVING_REGISTRY.register
class WaypointL1Loss(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(WaypointL1Loss, self).__init__()
        self.loss = l1_loss(reduction="none")
        # TODO: remove this hardcode
        # and make it extensible to variable trajectory length
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, Dict]:
            - first element: loss to be back propagated
            - second element: extra information
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        extra_info = dict()

        return torch.mean(loss), extra_info

@CODRIVING_REGISTRY.register
class WaypointL1Loss20(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(WaypointL1Loss20, self).__init__()
        self.loss = l1_loss(reduction="none")
        # TODO: remove this hardcode
        # and make it extensible to variable trajectory length
        weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
            0
        ]
        self.weights = []
        for i in range(len(weights)-1):
            self.weights.append(weights[i])
            self.weights.append((weights[i] + weights[i+1]) / 2)

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, Dict]:
            - first element: loss to be back propagated
            - second element: extra information
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        extra_info = dict()

        return torch.mean(loss), extra_info

@CODRIVING_REGISTRY.register
class ADE_FDE(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(ADE_FDE, self).__init__()
        self.id = 0

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, torch.Tensor]:
            - first element: ADE
            - second element: FDE
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']
        
        # from matplotlib import pyplot as plt
        # output_1 = output.detach().cpu().numpy()
        # target_1 = target.detach().cpu().numpy()
        # cmd_speed_dict = {8: 'Stop', 4: 'Slow down', 2: 'Hold', 1: 'Accelerate'}
        # cmd_direction_dict = {32: 'Left', 16: 'Right', 8: 'Straight', 4: 'lane follow', 2: 'lane change left', 1: 'lane change right'}
        
        # for i in range(output.shape[0]):                        
        #     x1, y1 = output_1[i, :, 0], output_1[i, :, 1]
        #     x2, y2 = target_1[i, :, 0], target_1[i, :, 1]

        #     y1 = np.abs(y1)
        #     y2 = np.abs(y2)

        #     plt.figure(figsize=(10, 6))
        #     plt.plot(x1, y1, marker='o', label='output', color='blue')
        #     plt.plot(x2, y2, marker='x', label='target', color='red')
        #     plt.xlabel('X Coordinate')
        #     plt.ylabel('Y Coordinate')
            
            
        #     # try:
        #     # import ipdb; ipdb.set_trace()
        #     speed_index = int(torch.dot(batch_data["cmd_speed"][i].detach().cpu(), torch.tensor([8, 4, 2, 1], dtype=torch.float32)).item())
        #     direction_index = int(torch.dot(batch_data["cmd_direction"][i].detach().cpu(), torch.tensor([32, 16, 8, 4, 2, 1], dtype=torch.float32)).item())
        #     plt.title(f'cmd speed: {cmd_speed_dict[speed_index]}\ncmd direction: {cmd_direction_dict[direction_index]}')
        #     # except Exception as e:
        #     # import ipdb; ipdb.set_trace()
        #     #     plt.title(f"cmd speed: {batch_data['cmd_speed'][i]}\ncmd direction: {batch_data['cmd_direction'][i]}")

        #     plt.xlim(-5, 5) 
        #     plt.ylim(0, 20) 

        #     plt.legend()
        #     plt.grid()
        #     plt.savefig(f'./vis_results/interpolate/{self.id}_{i}.png')
        #     plt.close()  

        # self.id += 1
        
        dis = (torch.sum((output - target)**2, (2)))**0.5 # shape: n, 10
        ADE = torch.mean(dis, (1)) # shape: n
        FDE = torch.mean(dis[:,-1:], (1)) # shape: n
        v_output = (output[:,1:,:] - output[:,:-1,:])/5 # shape: n, 9, 2
        v_target = (target[:,1:,:] - target[:,:-1,:])/5 # shape: n, 9, 2
        v_dis = (torch.sum((v_output - v_target)**2, (2)))**0.5 # shape: n, 9
        FVE = torch.mean(v_dis[:,-1:], (1)) # shape: n
        a_output_average = (v_output[:,-1,:] - v_output[:,0,:])/2 # 0.2sx10
        a_target_average = (v_target[:,-1,:] - v_target[:,0,:])/2
        AAE = torch.sum((a_output_average - a_target_average)**2, (1))**0.5 # shape: n
        # import ipdb; ipdb.set_trace()
        return ADE, FDE, FVE, AAE
    
@CODRIVING_REGISTRY.register
class WaypointL1ExpLoss(nn.Module):
    """Loss for supervising waypoint predictor
    """
    def __init__(self, l1_loss=torch.nn.L1Loss):
        super(WaypointL1ExpLoss, self).__init__()
        self.loss = l1_loss(reduction="none")
        # TODO: remove this hardcode
        # and make it extensible to variable trajectory length
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def forward(self,
                batch_data : Dict,
                model_output : Dict,
        ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss

        Args:
            batch_data: loaded batch data
            model_output: output from model

        Return:
            Tuple[torch.Tensor, Dict]:
            - first element: loss to be back propagated
            - second element: extra information
        """
        output = model_output['future_waypoints']
        target = batch_data['future_waypoints']
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.exp(loss) - 1
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        extra_info = dict()

        return torch.mean(loss), extra_info