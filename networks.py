import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.spectral_norm import spectral_norm


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.Mish()]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.Mish()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        # self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        # self.relu = nn.ReLU(True)

        # Self Attention
        # self.self_att = Self_Attn(ngf * mult, 'relu')
        # self.conv1x1 = nn.Conv2d(ngf * mult * 3, ngf * mult, kernel_size=1, stride=1, bias=True) # for multi-scale
        # self.relu = nn.ReLU(True)

        self.self_att = Stride_Self_Attn(ngf * mult, 'relu')
        self.conv1x1 = nn.Conv2d(ngf * mult * 3, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        self.mish = nn.Mish()

        # Conv CAM
        # self.g_conv_down = nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=2, padding=1, groups=ngf * mult, bias=False)
        # self.g_conv = nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, groups=ngf * mult, bias=False)
        # self.g_conv_3 = nn.Conv2d(ngf * mult, ngf * mult * 4, kernel_size=3, stride=1, padding=1, groups=ngf * mult, bias=False)
        # self.g_conv_2 = nn.Conv2d(ngf * mult * 2, ngf * mult * 4, kernel_size=3, stride=1, padding=1, groups=ngf * mult, bias=False)
        # self.g_conv_1 = nn.Conv2d(ngf * mult * 4, ngf * mult * 4, kernel_size=3, stride=1, padding=1, groups=ngf * mult, bias=False)
        # self.g_conv_0 = nn.Conv2d(ngf * mult * 4, ngf * mult, kernel_size=3, stride=1, padding=1, groups=ngf * mult, bias=False)
        self.conv1x1_logit = nn.Conv2d(ngf * mult, 1, kernel_size=1, stride=1, bias=True)
        self.conv_CAM_var = nn.Parameter(torch.tensor(0.5))
        
        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.Mish(),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.Mish()]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.Mish(),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.Mish()]
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(int(ngf * mult / 2)),
                         nn.Mish()]

        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]

        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock2 = nn.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        content_feature = x

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # x = torch.cat([gap, gmp], 1)
        # x = self.relu(self.conv1x1(x))

        self_att_map = self.self_att(x)

        self_att_map = self.mish(self.conv1x1(self_att_map))

        c_cam_logit = self.conv1x1_logit(self_att_map)

        # x = (self_att_map * (1.0 - self.conv_CAM_var)) + (self.conv_CAM_var * content_feature)
        x = self_att_map + content_feature
        # heatmap = torch.sum(g_conv, dim=1, keepdim=True)
        heatmap = torch.sum(self_att_map, dim=1, keepdim=True)

        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, c_cam_logit, heatmap, content_feature


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.Mish()]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, stride=1, padding=0))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1,  stride=1, padding=0))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1,  stride=1, padding=0))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) fx
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H) gx

        energy =  torch.bmm(proj_query,proj_key) # transpose check 

        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N hx

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x
        return out

class Stride_Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Stride_Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, stride=1, padding=0))
        self.key_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1,  stride=1, padding=0))
        self.value_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1,  stride=1, padding=0))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.pooling_h = nn.MaxPool2d((5,3), stride=(2,1), padding=(2,1))
        self.query_conv_3 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, stride=1, padding=0))
        self.key_conv_3 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1,  stride=1, padding=0))
        self.value_conv_3 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1,  stride=1, padding=0))
        self.gamma_3 = nn.Parameter(torch.zeros(1))
        self.upsample_3 = nn.Upsample(scale_factor=(2,1), mode='nearest')

        self.pooling_w = nn.MaxPool2d((3,5), stride=(1,2), padding=(1,2))
        self.query_conv_7 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1, stride=1, padding=0))
        self.key_conv_7 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1,  stride=1, padding=0))
        self.value_conv_7 = nn.utils.spectral_norm(nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1,  stride=1, padding=0))
        self.gamma_7 = nn.Parameter(torch.zeros(1))
        self.upsample_7 = nn.Upsample(scale_factor=(1,2), mode='nearest')

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) fx
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H) gx

        energy =  torch.bmm(proj_query,proj_key) # transpose check 

        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N hx

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x

        # 3x3
        x_h = self.pooling_h(x)
        m_batchsize,C,width ,height = x_h.size()

        proj_query_3  = self.query_conv_3(x_h).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) fx
        proj_key_3 =  self.key_conv_3(x_h).view(m_batchsize,-1,width*height) # B X C x (*W*H) gx

        energy_3 =  torch.bmm(proj_query_3,proj_key_3) # transpose check 

        attention_3 = self.softmax(energy_3) # BX (N) X (N) 
        proj_value_3 = self.value_conv_3(x_h).view(m_batchsize,-1,width*height) # B X C X N hx

        out_3 = torch.bmm(proj_value_3,attention_3.permute(0,2,1) )
        out_3 = out_3.view(m_batchsize,C,width,height)
        
        out_3 = self.gamma_3 * out_3 + x_h

        # 5x5
        x_w = self.pooling_w(x)
        m_batchsize,C,width ,height = x_w.size()
        proj_query_7  = self.query_conv_7(x_w).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N) fx
        proj_key_7 =  self.key_conv_7(x_w).view(m_batchsize,-1,width*height) # B X C x (*W*H) gx

        energy_7 =  torch.bmm(proj_query_7,proj_key_7) # transpose check 

        attention_7 = self.softmax(energy_7) # BX (N) X (N) 
        proj_value_7 = self.value_conv_7(x_w).view(m_batchsize,-1,width*height) # B X C X N hx

        out_7 = torch.bmm(proj_value_7,attention_7.permute(0,2,1) )
        out_7 = out_7.view(m_batchsize,C,width,height)
        
        out_7 = self.gamma * out_7 + x_w

        out_3 = self.upsample_3(out_3)
        out_7 = self.upsample_7(out_7)

        cat_out = torch.cat([out, out_3, out_7], 1)

        return cat_out

class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.Mish()

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2,True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        # Self Attention
        self.self_att = Self_Attn(ndf * mult, 'relu')

        # conv CAM
        # self.g_conv = nn.Conv2d(ndf * mult, ndf * mult, kernel_size=3, stride=1, padding=1, groups=ndf * mult, bias=False)
        # self.g_conv_3 = nn.Conv2d(ndf * mult, ndf * mult * 4, kernel_size=1, stride=1, padding=0, groups=ndf * mult, bias=False)
        # self.g_conv_2 = nn.Conv2d(ndf * mult * 2, ndf * mult * 4, kernel_size=1, stride=1, padding=0, groups=ndf * mult, bias=False)
        # self.g_conv_1 = nn.Conv2d(ndf * mult * 4, ndf * mult * 4, kernel_size=1, stride=1, padding=0, groups=ndf * mult, bias=False)
        # self.g_conv_down = nn.Conv2d(ndf * mult, ndf * mult, kernel_size=3, stride=2, padding=1, groups=ndf * mult, bias=False)
        self.conv_CAM_var = nn.Parameter(torch.tensor(0.5))

        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        # self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.conv1x1 = nn.Conv2d(ndf * mult, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.conv1x1_logit = nn.Conv2d(ndf * mult, 1, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.mish = nn.Mish()

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)
        

    def forward(self, input):
        x = self.model(input)
        content_feature = x

        # gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        # gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        # gap_weight = list(self.gap_fc.parameters())[0]
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        # gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        # gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        # gmp_weight = list(self.gmp_fc.parameters())[0]
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        # cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        # x = torch.cat([gap, gmp], 1)
        # x = self.leaky_relu(self.conv1x1(x))

        self_att_map = self.self_att(x)
        # g_conv_3 = self.mish(self.g_conv_3(self_att_map))
        # g_conv_2 = self.mish(self.g_conv_2(g_conv_3))
        # g_conv_1 = self.mish(self.g_conv_1(g_conv_2))

        # c_cam_logit = self.conv1x1_logit(g_conv_1)

        self_att_map = self.mish(self.conv1x1(self_att_map))
        c_cam_logit = self.conv1x1_logit(self_att_map)
        
        # x = (self_att_map * (1.0 - self.conv_CAM_var)) + (self.conv_CAM_var * content_feature)
        x = self_att_map + content_feature

        heatmap = torch.sum(self_att_map, dim=1, keepdim=True)

        # g_conv_down = self.g_conv_down(self_att_map) # legacy

        # g_conv = self.mish(self.g_conv(self_att_map))
        # c_cam_logit = self.conv1x1_logit(g_conv)
        
        # # x = self.relu(self.conv1x1(x))
        # x = self.mish(self.conv1x1(g_conv))
        # x = (x * (1.0 - self.conv_CAM_var)) + (self.conv_CAM_var * content_feature)

        # heatmap = torch.sum(g_conv, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, c_cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        self.c_clip_min = 0.1
        self.c_clip_max = 0.9
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

        elif hasattr(module, 'conv_CAM_var'):
            w = module.conv_CAM_var.data
            w = w.clamp(self.c_clip_min, self.c_clip_max)
            module.conv_CAM_var.data = w
