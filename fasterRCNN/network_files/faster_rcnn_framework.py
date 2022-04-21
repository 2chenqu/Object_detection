import torch
from torch import nn
from collections import OrderedDict
from network_files.rpn_function import AnchorsGenerator, RPNHead, RegionProposalNetwork
from network_files.roi_head import RoIHeads
from torchvision.ops import MultiScaleRoIAlign
from torch.jit.annotations import Tuple, List, Dict, Optional
from torch import Tensor
import torch.nn.functional as F
import warnings
from network_files.transform import GeneralizedRCNNTransform


class FasterRCNNBase(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):  骨干迁移网络
        rpn (nn.Module):区域建议网络部分
        roi_heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it. 结构图中除了transform,backbone和rpn剩下的就是roi_heads
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model   主要是对输入图像的张量进行归一化处理
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        # used only on torchscript mode
        self._has_warned = False

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        '数据集为训练集时self.trainning为True,当数据集为验证集时self.trainning为False'
        # type: (Dict[str, Tensor], List[Dict[str, Tensor]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        if self.training:
            return losses
        return detections

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]])
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            '在训练时需要传入含有图片目标信息的target文件来计算loss'
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], []) #初始化用来存储原始图片信息的变量
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        # original_image_sizes = [img.shape[-2:] for img in images]

        images, targets = self.transform(images, targets) #进行归一化处理
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):  # 若只在一层特征层上预测，将feature放入有序字典中，并编号为‘0’
            features = OrderedDict([('0', features)])  # 若在多层特征层上预测，传入的就是一个有序字典

        # 将特征层以及标注target信息传入rpn中  这里的rpn为什么要输入images参数？
        proposals, proposal_losses = self.rpn(images, features, targets)

        # 将rpn生成的数据以及标注target信息传入fast rcnn后半部分  这里为什么传进images.image_sizes参数，这个参数具体是什么？
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)

        # 对网络的预测结果进行后处理（主要将bboxes还原到原图像尺度上）
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

        # if self.training:
        #     return losses
        #
        # return detections


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)#展平成两个维度

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    box_predictor模块
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): 经过Two MLPhead输出的一维向量的维度大小
        num_classes (int): 数据集中要检测的目标的类别数 (VOC2012数据集中有20个类别再加上一个背景类别总数也就是21)
    """
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


class FasterRCNN(FasterRCNNBase):
    """
    Implements Faster R-CNN.

    The input to the model is expected to be a list of tensors, each of shape [C, H, W], one for each
    image, and should be in 0-1 range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box

    The model returns a Dict[Tensor] during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between
          0 and H and 0 and W
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores or each prediction

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            It should contain a out_channels attribute, which indicates the number of output
            channels that each feature map has (and it should be the same for all feature maps).
            The backbone should return a single Tensor or and OrderedDict[Tensor].
        num_classes (int): number of output classes of the model (including the background).
            If box_predictor is specified, num_classes should be None.
        min_size (int): minimum size of the image to be rescaled before feeding it to the backbone
        max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        image_mean (Tuple[float, float, float]): mean values used for input normalization.
            They are generally the mean values of the dataset on which the backbone has been trained
            on
        image_std (Tuple[float, float, float]): std values used for input normalization.
            They are generally the std values of the dataset on which the backbone has been trained on
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        box_roi_pool (MultiScaleRoIAlign): the module which crops and resizes the feature maps in
            the locations indicated by the bounding boxes
        box_head (nn.Module): module that takes the cropped feature maps as input
        box_predictor (nn.Module): module that takes the output of box_head and returns the
            classification logits and box regression deltas.
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_detections_per_img (int): maximum number of detections per image, for all classes.
        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_batch_size_per_image (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals in a mini-batch during training
            of the classification head
        bbox_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes

    """

    def __init__(self, backbone, num_classes=None,
                 # transform parameter
                 min_size=800, max_size=1333, #预处理时resize设置的最小尺寸和最大尺寸
                 image_mean=None, image_std=None, #预处理时normalize时使用的均值和方差
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,   #网络结构
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000, #rpn在nms处理前保留的proposal数
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000, #rpn在nms处理后保留的proposal数
                 rpn_nms_thresh=0.7, #rpn在nms处理时使用的iou阈值（类别分数最高框和其余框的iou）
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3, #rpn计算损失时，规划正负样本时设置的iou阈值(根据真实框和其余框之间的iou)
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5, #rpn计算损失时采样的样本数，以及正样本数占总样本数的比例
                 # Bpx parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 # 移除低目标概率 fast rcnn中进行nms处理的阈值 对预测结果根据score排序取前面100个目标
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 # faster rcnn计算损失时，规划正负样本时设置的iou阈值(根据真实框和其余框之间的iou)
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 #faster rcnn计算损失时采样的样本数，以及正样本所占所有样本的比例
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None):

        if not hasattr(backbone, "out_channels"):
            '判断bockbone是否有输出通道这个变量，在训练脚本内赋值这个属性'
            raise ValueError(
                "backbone should contain an attribute out_channels"
                "specifying the number of output channels  (assumed to be the"
                "same for all the levels"
            )
        #断言rpn_anchor_generator是不是为AnchorsGenerator或者none
        #断言box_roi_pool是不是MutiScaleRoiAlign或者none
        assert isinstance(rpn_anchor_generator, (AnchorsGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            '这里在train脚本内传入了num_classes变量不为空 然而predoctor为None所以直接跳过这段程序不会报错'
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor "
                                 "is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        # 预测特征层的channels
        out_channels = backbone.out_channels

        # 若anchor生成器为空，则自动生成针对resnet50_fpn的anchor生成器
        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes) #将该高宽比元组复制5次
            #AnchorsGenerator是在rpn_function.py内定义的
            rpn_anchor_generator = AnchorsGenerator(
                anchor_sizes, aspect_ratios
            )

        # 生成RPN通过滑动窗口预测网络部分
        if rpn_head is None:
            'rpn_head为faterRcnnbase内定义的RPNHead'
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
            )

        #rpn中采用非极大值抑制前后保留的训练集和测试集图片数量
        # 默认rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
        # 默认rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        # 定义整个RPN框架
        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #  box_roi_pool也就是结构图中Roipooling层
        if box_roi_pool is None:
            '将box_roi_pool设置为torchvision内的 MultiScaleRoIAlign可以直接调用' \
            '也就是使用resnet50作为back_bone的时候（因为在nn父类继承下）'
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 在哪些特征层进行预测
                output_size=[7, 7],
                sampling_ratio=2)

        # fast RCNN中roi pooling后的两个全连接层部分
        if box_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认等于7
            representation_size = 1024
            box_head = TwoMLPHead(
                out_channels * resolution ** 2, #通道数为backbone的输出通道数，roi不改变通道数
                representation_size  #经过两个全连接层输出的一维向量的维度为1024
            )

        # 在box_head的输出上预测部分
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                representation_size,
                num_classes)

        # 将roi pooling, box_head以及box_predictor结合在一起
        roi_heads = RoIHeads(
            # box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        #使用imagenet内的图像归一化处理的均值和方差值
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        # 对数据进行标准化，缩放，打包成batch等处理部分
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        #将继承父类faster rcnnbase内的backbone, rpn, roi_heads, transform等类属性重新定义初始化方法
        super(FasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

