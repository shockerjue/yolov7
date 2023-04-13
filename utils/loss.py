# Loss functions

import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.torch_utils import is_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False, kpt_label=False):
        super(ComputeLoss, self).__init__()
        self.kpt_label = kpt_label
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        # 定义分类损失和置信度损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        BCE_kptv = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors', 'nkpt':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):  # predictions, targets, model
        device = targets.device

        # 初始化对应的损失
        # lcls      分类损失
        # lbox      回归损失
        # lobj      置信度损失(目标置信度)
        # lkpt      关键点损失
        # lkptv     置信度损失(关键点置信度)
        lcls, lbox, lobj, lkpt, lkptv = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        sigmas = torch.tensor([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89], device=device) / 10.0

        # tcls      分类索引列表
        # tbox      目标坐标偏移值
        # tkpt      关键点数据信息
        # indeices  真实目标对应的图像索引、anchor所以、网格坐标
        # anch      对应的anchor值(真实的anchor大小)
        tcls, tbox, tkpt, indices, anchors = self.build_targets(p, targets)  # targets

        # 遍历特征信息，计算对应的损失
        for i, pi in enumerate(p):  # layer index, layer predictions
            # 取出图像索引、anchor索引，网格信息
            # b 图像索引
            # a anchor索引
            # gj,gi 网格信息
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx

            # 从特征层上获取分类信息
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            # 取出图像索引的数量
            n = b.shape[0]  # number of targets
            
            if n:
                # 精确得到第 b 张图片的第 a 个 feature map 的 grid_cell(gi, gj) 对应的预测值
                # 用这个预测值与我们筛选的这个 grid_cell 的真实框进行预测(计算损失)
                """
                主要是通过targets构建的anchor取出对应特征图上数据,也就是有目标的特征数据
                [561, 85] 其中[num_targets, xycwh]
                b是图像索引，对应的就是特征图上的batch_size索引
                a是anchor索引，对应的就是anchor序号(0,1,2)
                gj是特征图上的y坐标(比如80x80的y坐标)
                gi是特征图上的x坐标(比如80x80的x坐标)
                这样就可以对应到具体特征图上的某个像素点以及通道数了
                (16, 3, 80, 80, 136)
                (b,  a, gj, gi)
                """
                # 根据真实目标的信息从特征层上取出对应的数据
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                # 获取当前目标存在的head所预测的中心坐标pxy，pwh,以及类置信度 
                # 新的公式:  pxy = [-0.5 + cx, 1.5 + cx]    pwh = [0, 4pw]   这个区域内都是正样本
                pxy = ps[:, :2].sigmoid() * 2. - 0.5

                # 将归一化后的值转换成真实的wh
                # 其中wh就是要学习的参数，找到最符合或最接近使pbox接近tbox的偏移量
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box

                # 这里的tbox[i]中的xy是这个target对当前grid_cell左上角的偏移量[0,1]  而pbox.T是一个归一化的值
                # 就是要用这种方式训练传回loss,修改梯度让pbox越来越接近tbox(偏移量)
                # 其中函数中tbox也是进行转置运算，使其与pbox保持同样的shape(4, 543)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                
                # 主要是通过回归损失传回，修改梯度让pbox越来越接近真实tbox的偏移量
                # mean方法用于求平均数
                lbox += (1.0 - iou).mean()  # iou loss

                """
                以下是用于计算关键点损失以及关键点置信度损失
                """
                if self.kpt_label:
                    #Direct kpt prediction
                    # 从特征层上取出关键点对应的x,y,score值
                    # 取出方式是通过隔3三个取一个，主要是因为每一个关键点包含三个值(x,y,置信度)
                    pkpt_x = ps[:, 6::3] * 2. - 0.5
                    pkpt_y = ps[:, 7::3] * 2. - 0.5
                    pkpt_score = ps[:, 8::3]

                    #mask
                    # 计算关键点置信度损失，其中tkpt主要是取出x的值
                    kpt_mask = (tkpt[i][:, 0::2] != 0)
                    lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 

                    #l2 distance based loss
                    #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                    #oks based loss
                    # 下面主要是计算关键点损失

                    # 计算两点直接的距离
                    d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2

                    # 计算所有矩形的面积，也就是w*h
                    s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True)

                    
                    kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)

                    # 计算关键点损失
                    lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # 计算分类损失
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            # 取出置信度信息，计算置信度损失
            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        lkptv *= self.hyp['cls']
        lkpt *= self.hyp['kpt']
        bs = tobj.shape[0]  # batch size

        loss = lbox + lobj + lcls + lkpt + lkptv
        return loss * bs, torch.cat((lbox, lobj, lcls, lkpt, lkptv, loss)).detach()

    """
        这个函数是用来为所有GT筛选相应的anchor正样本
        筛选条件是比较GT和anchor的宽比和高比，大于一定的阈值就是负样本，反之正样本。
        筛选到的正样本信息（image_index, anchor_index, gridy, gridx, x1y1score1, x2y2score2, ...），传入 __call__ 函数，
        通过这个信息去筛选 pred 里每个 grid 预测得到的信息，保留对应 grid_cell 上的正样本。
        通过 build_targets 筛选的 GT 中的正样本和 pred 筛选出的对应位置的预测样本 进行计算损失。
        build_targets 函数用于获得在训练时计算 loss 所需要的目标框，也即正样本。与yolov3/v4的不同，yolov5支持跨网格预测。
        对于任何一个 GT bbox，三个预测特征层上都可能有先验框匹配，所以该函数输出的正样本框比传入的 targets （GT框）数目多
        具体处理过程:
            (1)首先通过 bbox 与当前层 anchor 做一遍过滤。对于任何一层计算当前 bbox 与当前层 anchor 的匹配程度，不采用IoU，
                而采用shape比例。如果anchor与bbox的宽高比差距大于4，则认为不匹配，此时忽略相应的bbox，即当做背景;
            (2)根据留下的bbox，在上下左右四个网格四个方向扩增采样（即对 bbox 计算落在的网格所有 anchors 都计算 loss(并不是直接和 GT 框比较计算 loss) )
        注意此时落在网格不再是一个，而是附近的多个，这样就增加了正样本数。
        用真实的bbox从对应的anchors中寻找正样本的anchors，并保留下来【用标签数据标记anchors中的正样本】。
    """
    # @param:p        来自特征提取层的输出，其中shape是(bs, 40, 40, 408)
    #                   其中后面的153存储就是关键点坐标信息(17 x 3 x 3)，其中每个关键点有三个值(x,y,score)，包含三个box
    # @param:targets    真实的目标信息（包含物体信息以及关键点的坐标信息，格式是：[image_index, x, y, w, h, x1y1score1, x2y2score2, ...]） 
    def build_targets(self, p, targets):
        """
        所有GT筛选相应的anchor正样本
        这里通过
        p       : list([16, 3, 80, 80, 136], [16, 3, 40, 40, 136],[16, 3, 20, 20, 136])
        targets : targets.shape[314, 40]  
        解析 build_targets(self, p, targets):函数
        :params p: p[i]的作用只是得到每个feature map的shape，如(16, 3, 20, 20, 136)
                   预测框,由模型构建中的三个检测头Detector返回的三个yolo特征层的输出
                   tensor格式 list列表 存放三个tensor 对应的是三个yolo特征层的输出
                   如: list([16, 3, 80, 80, 136], [16, 3, 40, 40, 136],[16, 3, 20, 20, 136])
                   [bs, anchor_num, grid_h, grid_w, classes+xywh+x1y1score1+x2y2score2+...]
                   可以看出来这里的预测值p是三个yolo特征层中每个grid_cell(每个grid_cell有三个预测值)的预测值,后面肯定要进行正样本筛选
        :params targets: 数据增强后的真实框 [63, 40] [num_target,  image_index+class+xywh+x1y1 + ...] xywh为归一化后的框
        :return tcls: 表示这个target所属的class index
                tbox: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
                tkpt: 存储对应关键点的信息
                indices: b: 表示这个target所属的image index
                         a: 表示这个target使用的anchor index
                        gj: gj表示这个网格的左上角y坐标
                        gi: 表示这个网格的左上角x坐标
                anch: 表示这个target所使用anchor的尺度（相对于这个feature map）  注意可能一个target会使用大小不同anchor进行计算
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, tkpt, indices, anch = [], [], [], [], []
        if self.kpt_label:
            # 初始化存储关键节点的信息，包含17个，每个信息是(xyscore)
            # [num_target,  image_index+class+xywh+x1y1+..., anchor_index]
            gain = torch.ones(41, device=targets.device)  # normalized to gridspace gain
        else:
            gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain

        # 初始化anchor索引结构
        """
        [
            [0,0,0,0,0,0,0,0,0,...],
            [1,1,1,1,1,1,1,1,1,...],
            [2,2,2,2,2,2,2,2,2,...],
        ]
        """
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        # 添加anchor索引到真实目标上，每个索引上都有所有的真实目标
        """
        [
            [x1,...........,0],
            [x2,...........,0],
            ...
            [x1,...........,1],
            [x2,...........,1],
            ...
            [x1,...........,2],
            [x2,...........,2],
        ]
        其中最有的0,1,2就是anchor索引，也就是特征层上像素点的box序号（每个像素点有三个会分为三个box）
        """
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.nl):
            # 获取特征层对应的anchor张量[[19,27], [44,40], [38,94]]
            anchors = self.anchors[i]
            if self.kpt_label:
                # 用特征层的wh初始化关键点的信息,也就是x1y1,...
                # [image_index, class, x, y, w, h, x1y1, x2y2, x3y3, ..., x17y17, anchor_index]
                gain[2:40] = torch.tensor(p[i].shape)[19*[3, 2]]  # xyxy gain
            else:
                gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            # 将真实目标关键信息与anchor关联，同时将其转换成特征上的坐标
            # 经过*运算以后，得到的就是特征层上的对应坐标
            # 注：真实目标的信息是经过归一化的，这里讲起映射到特征层上
            """
            [image_index, class, x, y, w, h, x1y1, x2y2, x3y3, ..., x17y17, anchor_index]
            tensor([[[ 0.00000,  0.00000,  3.36320, 16.90000,  6.72640, 29.20240,...,  0.00000],
            ...
            [ 0.00000,  0.00000, 29.00240, 19.61360, 17.59600, 31.63600,...,  0.00000]],
            [[ 0.00000,  0.00000,  3.36320, 16.90000,  6.72640, 29.20240,...,  1.00000],
            ...
            [ 0.00000,  0.00000, 29.00240, 19.61360, 17.59600, 31.63600,...,  1.00000]],
            [[ 0.00000,  0.00000,  3.36320, 16.90000,  6.72640, 29.20240,...,  2.00000],
            ...
            [ 0.00000,  0.00000, 29.00240, 19.61360, 17.59600, 31.63600,...,  2.00000]]])
            注：将真实目标与特征层进行联系起来，所以对应的值就是真实目标在特征层上的坐标信息
            (image_index, class, x, y, w, h, x1y1, x2y2, x3y3, ..., x17y17, anchor_index)
            (图像索引， 类别， 中心x坐标， 中心y坐标，宽度， 高度，关键点坐标， 对应的anchor索引)
            """
            t = targets * gain

            if nt:
                # Matches
                # 计算真实目标wh与anchor wh的比值
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                # 筛选条件  GT与anchor的宽比或高比超过一定的阈值 就当作负样本
                # flow.max(r, 1. / r)=[3, 314, 2] 筛选出宽比w1/w2 w2/w1 高比h1/h2 h2/h1中最大的那个
                # .max(2)返回宽比 高比两者中较大的一个值和它的索引  [0]返回较大的一个值
                # j.shape = [3, 314]  False: 当前anchor是当前gt的负样本  True: 当前anchor是当前gt的正样本
                # anchor的宽高与target的宽高比得到ratio1
                """
                假如此时得到的j【也就是和阈值相比较】均为False,就是表示没有匹配成功，也就是可以认为，
                此时的target中所有的目标不能被80*80的head所预测【或者说这些目标中没有小目标】
                tensor([[False, False, False],
                    [False, False, False],
                    [False, False, False]])
                假如知道target中目标可以被40*40这个head的第0与第2号anchor所匹配，这些anchor所匹配的样本就是我们要的正样本GT，那么
                tensor([[ True, False, False],
                    [False, False, False],
                    [ True,  True,  True]]) 
                """
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare

                # 根据筛选条件j, 过滤负样本, 得到所有gt的anchor正样本(batch_size张图片)
                # 知道当前gt的坐标 属于哪张图片 正样本对应的idx 也就得到了当前gt的正样本anchor
                # t: [3, 314, 7] -> [555, 7]  [num_Positive_sample, image_index+class+xywh+anchor_index]
                # 此时得到的j【也就是和阈值相比较】均为False,就是表示没有匹配成功，也就是可以认为，此时的target中所有的目标不能被80*80的head所预测【或者说这些目标中没有小目标】
                # 这里过滤以后就只有正样本了
                """如,过滤后保留下来的
                tensor([[ 0.00000,  0.00000,  1.68160,  8.45000,  3.36320, 14.60120,  0.00000],
                [ 0.00000,  0.00000,  1.68160,  8.45000,  3.36320, 14.60120,  2.00000],
                [ 0.00000,  0.00000,  5.98400,  9.76000,  9.54680, 14.60120,  2.00000],
                [ 0.00000,  0.00000, 14.50120,  9.80680,  8.79800, 15.81800,  2.00000]])
                """
                # 根据过滤后的信息，可以看出，保留下来的就是真实目标在特征层上的映射
                t = t[j]  # filter

                # Offsets
                # 得到过滤后目标的中心点坐标 
                # 筛选当前格子周围格子 找到 2 个离target中心最近的两个格子  
                # 可能周围的格子也预测到了高质量的样本 我们也要把这部分的预测信息加入正样本中
                # 除了target所在的当前格子外, 还有2个格子对目标进行检测(计算损失) 
                # 也就是说一个目标需要3个格子去预测(计算损失)
                # 首先当前格子是其中1个 再从当前格子的上下左右四个格子中选择2个
                # 用这三个格子去预测这个目标(计算损失)
                # feature map上的原点在左上角 向右为x轴正坐标 向下为y轴正坐标
                # 取target中心的坐标xy(相对feature map左上角的坐标)
                gxy = t[:, 2:4]  # grid xy

                # 得到中心点相对于边界的距离
                # 得到target中心点相对于右下角的坐标 
                # gxi.shape = [555, 2]
                gxi = gain[[2, 3]] - gxy  # inverse

                # jk和lm是判断gxy的中心点更偏向哪里
                # 筛选中心坐标距离当前grid_cell的左、上方偏移小于g=0.5 
                # 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # j: [555] bool 如果是True表示当前target中心点所在的格子的左边格子也对该target进行回归(后续进行计算损失)
                # k: [555] bool 如果是True表示当前target中心点所在的格子的上边格子也对该target进行回归(后续进行计算损失)
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T

                # 筛选中心坐标距离当前grid_cell的右、下方偏移小于g=0.5 且 中心坐标必须大于1(坐标不能在边上 此时就没有4个格子了)
                # l: [555] bool 如果是True表示当前target中心点所在的格子的右边格子也对该target进行回归(后续进行计算损失)
                # m: [555] bool 如果是True表示当前target中心点所在的格子的下边格子也对该target进行回归(后续进行计算损失)
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T

                # 得到的j如下，包含当前网格有五个cell，第一行保留所有的gtbox，第二行表示左边的cell中的gt，
                # 第三行是表示上方的cell中的gt，第四行是右边cell的网格，第五行是下方的cell中的gt。
                # 这里与v3和v4不同在于之前的yolo是目标落在哪个head的cell就由该cell进行预测，
                # 而v5通过增加邻近的cell来预测，这样就是相当于增加了正样本的数量。
                j = torch.stack((torch.ones_like(j), j, k, l, m))

                # 这里主要是进行数据的扩展，添加上下左右网格的检测，以增加正样本数量
                # 在yolov5中不仅仅用了中心点进行预测，还采用了距离中心点网格最近的两个网格，
                # 所以是有五种情况【四周的网格和当前中心的网格】同时用上面的j过滤，这样就可以得出哪些网格有目标
                # 得到筛选后所有格子的正样本 格子数<=3*555 都不在边上等号成立
                # t: [555, 7] -> 复制 5 份target[5, 555, 7]  分别对应当前格子和左上右下格子5个格子
                # 使用 j 筛选后 t 的形状: [1659, 7] 
                t = t.repeat((5, 1, 1))[j]

                # 将扩展的检测数据进行过滤，把不可能包含的网格排除
                # 在yolov5中不仅仅用了中心点进行预测，还采用了距离中心点网格最近的两个网格，
                # 所以是有五种情况【四周的网格和当前中心的网格】同时用上面的j过滤，这样就可以得出哪些网格有目标
                # flow.zeros_like(gxy)[None]: [1, 555, 2]   off[:, None]: [5, 1, 2]  => [5, 555, 2]
                # 得到所有筛选后的网格的中心相对于这个要预测的真实框所在网格边界
                # （左右上下边框）的偏移量，然后通过 j 筛选最终 offsets 的形状是 [1659, 2]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            # 获取对应的图像和类别信息
            b, c = t[:, :2].long().T  # image, class

            # 获取满足条件的xywh信息
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh

            # 预测真实框的网格所在的左上角坐标(有左上右下的网格)  
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            # 获取对应的anchors信息
            a = t[:, -1].long()  # anchor indices

            # 将图像索引、anchor索引以及网格中心坐标放入容器中
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices

            # tbix: xywh 其中xy为这个target对当前grid_cell左上角的偏移量
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box

            if self.kpt_label:
                # 计算关键点的坐标xy信息,并放入结果集中,这里的都是可信的
                for kpt in range(self.nkpt):
                    t[:, 6+2*kpt: 6+2*(kpt+1)][t[:,6+2*kpt: 6+2*(kpt+1)] !=0] -= gij[t[:,6+2*kpt: 6+2*(kpt+1)] !=0]
                tkpt.append(t[:, 6:-1])

            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        # tcls      分类索引列表
        # tbox      目标坐标偏移值
        # tkpt      关键点数据信息
        # indeices  真实目标对应的图像索引、anchor所以、网格坐标
        # anch      对应的anchor值(真实的anchor大小)
        return tcls, tbox, tkpt, indices, anch
