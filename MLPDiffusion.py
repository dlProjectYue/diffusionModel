import torch
import torch.nn as nn

class MLPDiffusion (nn.Module):
    def __init__(self,n_steps,num_units=128):
        super(MLPDiffusion,self).__init__()

        self.linears=nn.ModuleList(
            [
             nn.Linear(2,num_units),#原始是2x128
             nn.ReLU(),
             nn.Linear(num_units,num_units),
             nn.ReLU(),
             nn.Linear(num_units,num_units),
             nn.ReLU(),
             nn.Linear(num_units,2),
            ]
        )
        self.step_embeddings=nn.ModuleList(
            [
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
                nn.Embedding(n_steps,num_units),
            ]
        )
    def forward(self,x_0,t):
        x=x_0
        # x=x.to(device)
        for idx, embedding_layer in enumerate(self.step_embeddings):
            t_embedding=embedding_layer(t)
            x=self.linears[2*idx](x)
            x+=t_embedding
            x=self.linears[2*idx+1](x)
        x=self.linears[-1](x)
        return x
def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps,device):
    """ 对任意时刻t进行采样算Loss"""

    x_0 = x_0.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)
    batch_size=x_0.shape[0]#128
    #对一个batchsize样本生成随机的时刻t
    t=torch.randint(0,n_steps,size=(batch_size//2,)) #生成0~(n_steps-1)batchsize一半的长度数据
    t=torch.cat([t,n_steps-1-t],dim=0) #拼接后一半，后一半符合n_steps-1-t的规则 比如nsteps=10, [1,2,3]变成[1,2,3,9,8,7]
    t=t.unsqueeze(-1) #拼接一个维度，如果是-1就是在最后一个维度位置上添加1，变成(batchsize,1)的维度; (128,1)
    t=t.to(device)
    #取t时刻的alphasbarsqrt
    a=alphas_bar_sqrt[t]# 128个随机数代表128个时间点， 按照时间步数t的索引取出对应的元素，得到一个形状为(batch_size,1,2)的张量
    aml=one_minus_alphas_bar_sqrt[t]#(128,1)
    # print(f'aml.shape={aml.shape}')
    e=torch.randn_like(x_0)# (128,2) 生成随机噪声 每个元素都是从标准正态分布中采样的随机
    e = e.to(device)
    x=x_0*a+e*aml#(128,2)
    model=model.to(device)
    output=model(x,t.squeeze(-1))#（128，2）这个操作是为了将时间步数张量t从(batch_size, 1)的形状转化为(batch_size,)的形状，以便在后面的计算中能够正确地使用
    #随机采样一个时刻t,为了提高训练效率，确保t不重复
    return (e-output).square().mean()



def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt,device):
    """从x[T]采样t时刻的重构值"""
    t= torch.tensor([t]) #转换成tensor值 将时间步数t转化为PyTorch张量类型的目的是为了方便与其他张量进行计算，并能够在GPU上运行加速计算。
    t=t.to(device)
    x=x.to(device)
    betas[t]=betas[t].to(device)
    one_minus_alphas_bar_sqrt[t] = one_minus_alphas_bar_sqrt[t].to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    coeff=coeff.to(device)
    model=model.to(device)
    eps_theta = model (x,t)#预测逆向时 x在t时刻的噪声
    eps_theta=eps_theta.to(device)
    mean=(1/((1-betas[t]).to(device)).sqrt())*(x-(coeff*eps_theta))
    mean = mean.to(device)
    z=torch.randn_like(x)
    z = z.to(device)
    sigma_t=betas[t].sqrt()
    sigma_t = sigma_t.to(device)
    sample=mean+sigma_t*z
    return (sample)
def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt,device):
    """从x[T]恢复x[T-1] x[T-2] x[T-3]... x[0]"""
    cur_x=torch.randn(shape) #随机正态分布x(10000,2)
    x_seq=[cur_x]#最终有100个（10000，2）
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt,device=device)
        x_seq.append(cur_x)
    return x_seq
def p_sample1(model,x,t,alphas_bar_sqrt,one_minus_alphas_bar_sqrt):
    t=torch.tensor([t])
    coeff=alphas_bar_sqrt[t]/one_minus_alphas_bar_sqrt[t]
    eps_theta=model(x,t)
    mean=(1/(1-alphas_bar_sqrt[t])).sqrt()*(x-(coeff*eps_theta))
    z=torch.randn_like(x)
    sigma_t=alphas_bar_sqrt[t]
    sample=mean+sigma_t*z
    return sample
def p_sample_loop1(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
    """从x[T]恢复x[T-1] x[T-2] x[T-3]... x[0]"""
    cur_x=torch.randn(shape) #随机正态分布x
    x_seq=[cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x.cpu().numpy)
    return x_seq.reverse()



