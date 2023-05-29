import torch
import torch.nn as nn
import torchvision

# FRY. 
# use pretrained resnet18 as backbone
class ActorCritic(nn.Module):
    def __init__(self, N_grid, num_ops=12, img_size=224, RLprob=False):
        super(ActorCritic , self).__init__()

        self.RLprob = RLprob

        self.fea_extracter = torchvision.models.resnet18(pretrained=False)
        #self.fea_extracter.load_state_dict(torch.load('/model/bitahub/ResNet/resnet18.pth'))

        self.fea_extracter = nn.Sequential(*list(self.fea_extracter.children())[:-2])   # out shape : 7*7
        self.pool27x7 = None
        if (img_size == 448) and (N_grid != 14):
            self.pool27x7 = nn.MaxPool2d(2, 2)
        self.fea_last_conv = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32)
        )

        # N 14 2 2
        # 32*7*7
        if (img_size == 448) and (N_grid == 14):
            input_num = 32 * 14 * 14
        else: 
            input_num = 32 * 7 * 7
        self.value = nn.Sequential(
            nn.Linear(input_num, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        channels_1 = 64 
        channels_2 = 64
        channels_3 = 128
        self.policy_neck = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(32, channels_1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels_1),
                nn.ReLU(),
                nn.Conv2d(channels_1, channels_2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(channels_2),
                nn.ReLU(),  # shape: Nx64x7x7
            )
        if N_grid in [7, 14]:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, num_ops, kernel_size=3, stride=1, padding=1),
                nn.Softmax(dim=1)
            )
        # if N_grid == 6:
            # ! tobe done
        elif N_grid == 5:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, num_ops, kernel_size=3, stride=1, padding=0),
                nn.Softmax(dim=1)
            )        
        elif N_grid == 4:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, num_ops, kernel_size=3, stride=2, padding=1),
                nn.Softmax(dim=1)
            )
        elif N_grid == 3:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=1, padding=0),
                nn.BatchNorm2d(channels_3),
                nn.ReLU(),
                nn.Conv2d(channels_3, num_ops, kernel_size=3, stride=1, padding=0),
                nn.Softmax(dim=1)
            ) 
        elif N_grid == 2:
            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels_3),
                nn.ReLU(),
                nn.Conv2d(channels_3, num_ops, kernel_size=2, stride=2, padding=0),
                nn.Softmax(dim=1)
            )
        elif N_grid == 1:

            self.policy_head = nn.Sequential(
                nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(channels_3),
                nn.ReLU(),
                nn.Conv2d(channels_3, channels_3, kernel_size=2, stride=2, padding=0),
                nn.AvgPool2d(2, 2),
                nn.Conv2d(channels_3, num_ops, kernel_size=1, stride=1, padding=0),
                nn.Softmax(dim=1)
            )
        else:
            raise('choices of N_grid is (7, 5,) !')

        if RLprob:
            self.prob_neck = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(32, channels_1, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels_1),
                    nn.ReLU(),
                    nn.Conv2d(channels_1, channels_2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(channels_2),
                    nn.ReLU(),  # shape: Nx64x7x7
                )
            if N_grid in [7, 14]:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, 10, kernel_size=3, stride=1, padding=1),
                    nn.Softmax(dim=1)
                )
            # if N_grid == 6:
                # ! tobe done
            elif N_grid == 5:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, 10, kernel_size=3, stride=1, padding=0),
                    nn.Softmax(dim=1)
                )        
            elif N_grid == 4:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, 10, kernel_size=3, stride=2, padding=1),
                    nn.Softmax(dim=1)
                )
            elif N_grid == 3:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=1, padding=0),
                    nn.BatchNorm2d(channels_3),
                    nn.ReLU(),
                    nn.Conv2d(channels_3, 10, kernel_size=3, stride=1, padding=0),
                    nn.Softmax(dim=1)
                ) 
            elif N_grid == 2:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels_3),
                    nn.ReLU(),
                    nn.Conv2d(channels_3, 10, kernel_size=2, stride=2, padding=0),
                    nn.Softmax(dim=1)
                )
            elif N_grid == 1:
                self.prob_head = nn.Sequential(
                    nn.Conv2d(channels_2, channels_3, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(channels_3),
                    nn.ReLU(),
                    nn.Conv2d(channels_3, channels_3, kernel_size=2, stride=2, padding=0),
                    nn.AvgPool2d(2, 2),
                    nn.Conv2d(channels_3, 10, kernel_size=1, stride=1, padding=0),
                    nn.Softmax(dim=1)
                )
            else:
                raise('choices of N_grid is (7, 5,) !')

    def forward(self, image):

        state = self.fea_extracter(image)
        if self.pool27x7:
            state = self.pool27x7(state)
        state = self.fea_last_conv(state)

        state_value = self.value(state.view(state.size(0),-1))

        operation_dist = self.policy_neck(state)
        operation_dist = self.policy_head(operation_dist)

        if self.RLprob:
            prob_dist = self.prob_neck(state)
            prob_dist = self.prob_head(prob_dist)
            return operation_dist, prob_dist, state_value

        return operation_dist, state_value

if __name__ == "__main__":
    model = ActorCritic(14)
    print(model)

    input = torch.randn(8, 3, 224, 224)

    operation_dist, state_value = model(input)
    print(operation_dist.shape)
    print(state_value.shape)

    # print(output)
