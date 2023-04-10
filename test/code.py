import torch 
import math

def logger(tip,gain, _is = True):
    if _is:
        print(tip, gain)

def build_targets(p):
    na = 3
    nt = 63
    tcls, tbox, indices, anch = [], [], [], []

    targets = torch.rand([63, 6]).float()
    gain = torch.ones(7)
    logger("gain:", gain, False)

    ai = torch.arange(na).float().view(na, 1).repeat(1, nt)
    logger("ai:", ai, False)

    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    logger("targets:", targets, False)

    g = 0.5
    off = torch.tensor([
        [0, 0],[1, 0],[0, 1], [-1, 0], [0,-1]
    ]).float() * g
    logger("off:", off, False)

    anchors = torch.tensor([
        [
            [12,16],[19,36],[40,28]
        ],
        [
            [36,75],[76,55],[72,146]
        ],
        [
            [142,110],[192,243],[459,401]
        ]
    ])
    logger("anchors", anchors, False)

    for i in range(3):
        anchor = anchors[i]

        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]
        logger("logger[2:6]:", gain, False)

        t = targets * gain
        logger("t:", t, False)

        if nt:
            r = t[:, :, 4:6] / anchor[:, None] 
            logger("r:", r, False)

            j = torch.max(r, 1. / r).max(2)[0] < torch.tensor([15, 30, 60]).view(na, 1)
            logger("j:", j, False)

            t = t[j]
            logger("t:", t.shape, False)

            gxy = t[:, 2:4]
            logger("gxy:", gxy, False)

            gxi = gain[[2,3]] - gxy 
            logger("gxi:", gxi, False)
            logger("gain[[2,3]]: ", gain[[2,3]], False)

            j,k = ((gxy % 1. < g) & (gxy > 1.)).T
            logger("j:", k, False)
            logger("k:", k, False)

            l,m = ((gxi % 1. < g) * (gxi > 1.)).T
            logger("l:", l, False)
            logger("m:", m, False)

            j = torch.stack((torch.ones_like(j), j, k, l , m))
            logger("j:", j, False)

            t = t.repeat((5, 1, 1))[j]
            logger("t:", t.shape, False)

            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            logger("offsets: ", offsets, False)

        else:
            t = targets[0]
            offsets = 0
        
        b,c = t[:, :2].long().T 
        logger("b:", b, False)
        logger("c:", c, False)

        gxy = t[:, 2:4]
        gwh = t[:, 4:6]
        logger("gxy: ", gxy, False)
        logger("gwh:", gwh, False)

        gij = (gxy - offsets).long()
        logger("gij:", gij, False)

        gi, gj = gij.T 
        logger("gi:", gi , False)
        logger("gj: ", gj, False)

        a = t[:, 6].long()
        logger("a:", a, False)

        indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gj.clamp_(0, gain[2].long() - 1)))

        tbox.append(torch.cat((gxy - gij, gwh), 1))

        anch.append(anchor[a])
        logger("anch:", anchor[a], False)

        tcls.append(c)

    return tcls, tbox, indices, anch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def call(p):
    # 初始化损失值
    lcls,lbox,lobj = torch.zeros(1), torch.zeros(1), torch.zeros(1)
    tcls, tbox, indices, anchors = build_targets(p)

    for i,pi in enumerate(p):
        b,a,gj,gi = indices[i]
        tobj = torch.zeros_like(pi[..., 0])
        n = b.shape[0]
        if n:
            # 取出对应
            ps = pi[b, a, gj, gi]
            logger("ps:", ps.shape, False)

            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            logger("pwh:", pwh, False)

            pbox = torch.cat((pxy, pwh), 1)
            logger("pxy:", pbox.T.shape, False)
            logger("tbox[i]:", tbox[i].T.shape)

            logger("pbox:", pbox.T.shape, True)
            iou = bbox_iou(pbox.T, tbox[i],x1y1x2y2=False, CIoU=True)
            logger("iou:", iou, False)

            lbox += (1.0 - iou).mean()
            logger("lbox:", (1.0 - iou).mean(), False)
            logger("full_like:", torch.full_like(iou, 3), False)

            logger("detach:", iou.detach().clamp(0, 0.05), False)
            tobj[b, a, gj, gi] = (1.0 - 1.0) + 1.0 * iou.detach().clamp(0).type(tobj.dtype)
            logger("tobj:", tobj[b, a, gj, gi], False)


feature_map1 = torch.rand([16, 3, 80, 80, 5 + 80])
feature_map2 = torch.rand([16, 3, 40, 40, 5 + 80])
feature_map3 = torch.rand([16, 3, 20, 20, 5 + 80])
pred = [feature_map1, feature_map2, feature_map3]
call(pred)