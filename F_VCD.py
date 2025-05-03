from .util.utils import *
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import InputLayer, LambdaLayer
from tensorlayer.layers import ConcatLayer

# ------------------------ conv block ------------------------
def conv_block(layer, n_filter, kernel_size, is_train=True, activation=tf.nn.relu, is_in=False, border_mode="SAME", name='conv2d'):
    if is_in:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, padding=border_mode, name=name+'_conv2d')
        s = instance_norm(s, name=name+'in', is_train=is_train)
        s.outputs = activation(s.outputs)
    else:
        s = conv2d(layer, n_filter=n_filter, filter_size=kernel_size, stride=1, act=activation, padding=border_mode, name=name)
    return s

# ------------------------ MultiResBlock ------------------------
def MultiResBlock(layer, out_channel=None, is_train=True, alpha=1.0, name='MultiRes_block'):
    filter_num = out_channel * alpha
    n1_ = int(filter_num * 0.25)
    n2_ = int(filter_num * 0.25)
    n3_ = int(filter_num * 0.5)
    with tf.variable_scope(name):
        short_cut = conv_block(layer, n_filter=n1_+n2_+n3_, kernel_size=1, is_train=is_train, is_in=True)
        conv3x3 = conv_block(layer, n_filter=n1_, kernel_size=3, is_train=is_train, is_in=True, name='conv_block1')
        conv5x5 = conv_block(conv3x3, n_filter=n2_, kernel_size=3, is_train=is_train, is_in=True, name='conv_block2')
        conv7x7 = conv_block(conv5x5, n_filter=n3_, kernel_size=3, is_train=is_train, is_in=True, name='conv_block3')
        out = tf.concat([conv3x3.outputs, conv5x5.outputs, conv7x7.outputs], axis=-1, name='concat')
        out = InputLayer(out, name='concat_layer')
        out = instance_norm(out, is_train=is_train, name='in')
        out = merge([out, short_cut], name='merge_last')
        if out.outputs.get_shape().as_list()[-1] != out_channel:
            out = conv2d(out, n_filter=out_channel, filter_size=1, name='reshape_channel')
        out = LReluLayer(out, name='relu_last')
        out = instance_norm(out, name='batch_last')
    return out

# ------------------------ Residual Block ------------------------
def res_block(input, name='res_block'):
    c_num = input.outputs.get_shape()[-1].value
    with tf.variable_scope(name):
        n = conv2d(input, n_filter=c_num, filter_size=3, name='conv1')
        n = conv2d(n, n_filter=c_num, filter_size=3, name='conv2')
        # 直接对 Layer 对象做 merge，ElementwiseLayer 会自动跟踪 prev_layers
        n = merge([input, n], name='res-add')
    return n

# ------------------------ View Attention ------------------------
def ViewAttenionBlock(layer, out_channel, reduction=2, name='ViewAttenionBlock'):
    if not hasattr(layer, 'outputs'):
        layer = InputLayer(layer)
    c_in = layer.outputs.get_shape().as_list()[-1]
    feature_num = c_in // reduction
    with tf.variable_scope(name):
        n1 = GlobalMeanPool2d(layer, name='pooling1')
        n1 = tl.layers.DenseLayer(n1, n_units=feature_num, act=tf.nn.relu, name='FC1_1')
        n1 = tl.layers.DenseLayer(n1, n_units=c_in, act=tf.identity, name='FC1_2')

        n2 = GlobalMaxPool2d(layer, name='pooling2')
        n2 = tl.layers.DenseLayer(n2, n_units=feature_num, act=tf.nn.relu, name='FC2_1')
        n2 = tl.layers.DenseLayer(n2, n_units=c_in, act=tf.identity, name='FC2_2')

        n = merge([n1, n2], name='merge')
        n.outputs = tf.nn.sigmoid(n.outputs)
        n.outputs = tf.reshape(n.outputs, [-1, 1, 1, c_in])

        n = tl.layers.ElementwiseLayer([layer, n], combine_fn=tf.multiply, name='Attention')
        n = res_block(n, name='res_block1')
        n = res_block(n, name='res_block2')
    return n

# ------------------------ F_Denoise ------------------------
def F_Denoise(LFP, output_size, angRes=11, sr_factor=7, upscale_mode='one_step', is_train=True, reuse=False, name='F_Denoise', **kwargs):
    view_num = 3
    channels_interp = kwargs.get('channels_interp', 64)
    dialate_rate = [1, 2, 4]

    with tf.variable_scope(name, reuse=reuse):
        if hasattr(LFP, 'outputs'):
            n = LFP
        else:
            n = InputLayer(LFP, name='input_layer')
        n = LambdaLayer(n, lambda x: tf.transpose(x, perm=[0, 3, 1, 2]), name='transpose_input')

        with tf.variable_scope('Feature_extra'):
            n = conv2d(n, n_filter=channels_interp, filter_size=3, padding='SAME', name='conv0')
            # 先构造多个 layer
            aspp_pyramid = [
                conv2d_dilate(n, n_filter=channels_interp, filter_size=3, dilation=d_r,
                             name='dialate_pyramid_%d' % ii)
                for ii, d_r in enumerate(dialate_rate)
            ]
            # 用 ConcatLayer 保持 Layer.prev_layer 链
            n = ConcatLayer(aspp_pyramid, concat_dim=-1, name='dialation_concat')
            n = conv2d(n, n_filter=channels_interp, filter_size=1, padding='SAME', name='Pyramid_conv')
            n = res_block(n, name='res_1')

        with tf.variable_scope('Attention'):
            n = LambdaLayer(n, lambda x: tf.transpose(x, perm=[0, 2, 3, 1]), name='transpose_to_NHWC')
            fv0 = n
            fv1 = ViewAttenionBlock(fv0, out_channel=view_num, name='Atten_B1')
            fv2 = ViewAttenionBlock(merge([fv0, fv1], 'add_2'), out_channel=view_num, name='Atten_B2')
            FVA = merge([fv0, fv1, fv2], 'add_V')
            fs0 = n
            fs1 = ViewAttenionBlock(fs0, out_channel=channels_interp, name='spAtten_B1')
            fs2 = ViewAttenionBlock(merge([fs0, fs1], 'spadd_2'), out_channel=channels_interp, name='spAtten_B2')
            FSA = merge([fs0, fs1, fs2], 'spadd_V')

        with tf.variable_scope('fuse'):
            # FVA, FSA 本身就是 Layer 对象
            # 用 LambdaLayer 对 FSA 做 resize，保持 prev_layer
            FSA = LambdaLayer(
                FSA,
                lambda x, target=FVA.outputs: tf.image.resize(
                    x,
                    tf.stack([tf.shape(target)[1], tf.shape(target)[2]]),
                    method=tf.image.ResizeMethod.BILINEAR
                ),
                name='resize_FSA'
            )
            # 如果通道不匹配，用 conv2d 直接在 FSA Layer 上调整
            if FSA.outputs.get_shape().as_list()[-1] != FVA.outputs.get_shape().as_list()[-1]:
                FSA = conv2d(FSA,
                             n_filter=FVA.outputs.get_shape().as_list()[-1],
                             filter_size=1,
                             name='adjust_FSA_channels')

            # 融合时也用 ConcatLayer
            fuse = ConcatLayer([FVA, FSA], concat_dim=-1, name='SV_concat')
            fuse = conv2d(fuse, n_filter=channels_interp // 2, filter_size=1, name='conv1')
            fuse = res_block(fuse, name='res_1')
            fuse = conv2d(fuse, n_filter=channels_interp // 2, filter_size=1, name='conv2')
            fuse = res_block(fuse, name='res_2')

        with tf.variable_scope('upscale'):
            n = conv2d(fuse, n_filter=3, filter_size=1, name='conv2_out')
            n = LambdaLayer(n, lambda x: tf.nn.sigmoid(x), name='sigmoid_out')
            n = LambdaLayer(n, lambda x: tf.image.resize(x, [output_size[0], output_size[1]]), name='resize_out')

    return n

# ========================== F_Recon ==========================
def F_Recon(lf_extra, n_slices, output_size, is_train=True, reuse=False, name='F_Recon', **kwargs):
    channels_interp = kwargs.get('channels_interp', 64)
    interp_channels = channels_interp // 3
    dialate_rate = [1, 2, 4]

    with tf.variable_scope(name, reuse=reuse):
        # 如果传入的是 TensorLayer 的 Layer，就直接接着用；否则再包装成 InputLayer
        if hasattr(lf_extra, 'outputs'):
            n = lf_extra
        else:
            n = InputLayer(lf_extra, name='input_recon_layer')

        with tf.variable_scope('feature_extraction'):
            n = conv2d(n, n_filter=channels_interp, filter_size=3,
                       padding='SAME', name='conv0')
            aspp_pyramid = [
                conv2d_dilate(n, n_filter=channels_interp,
                             filter_size=3, dilation=d_r,
                             name='dialate_pyramid_%d' % ii)
                for ii, d_r in enumerate(dialate_rate)
            ]
            # 用 ConcatLayer 保持 prev_layer 链
            n = ConcatLayer(aspp_pyramid, concat_dim=-1, name='dialation_concat')
            n = conv2d(n, n_filter=channels_interp,
                       filter_size=1, padding='SAME', name='Pyramid_conv')
            n = res_block(n, name='res_block1')

        with tf.variable_scope('interp_layer'):
            n = tl.layers.UpSampling2dLayer(n, size=[9, 9], is_scale=True, name='upscale_before_interp')
            n = conv2d(n, n_filter=interp_channels, filter_size=3, padding='SAME', name='conv_interp')

        with tf.variable_scope('upscale'):
            if n.outputs.shape[1] != output_size[0] or n.outputs.shape[2] != output_size[1]:
                n = tl.layers.UpSampling2dLayer(n, size=(int(output_size[0]), int(output_size[1])), is_scale=False, name='resize_final')

        n = conv2d(n, n_filter=n_slices, filter_size=3, stride=1, name='conv_final')

        n.outputs = tf.clip_by_value(n.outputs, 0.0, 1.0)

        # crop_or_pad 的 Assert 节点只在 CPU 上有内核，必须切回 CPU
        h = tf.minimum(tf.shape(n.outputs)[1], output_size[0])
        w = tf.minimum(tf.shape(n.outputs)[2], output_size[1])
        with tf.device('/cpu:0'):
            n.outputs = tf.image.resize_with_crop_or_pad(n.outputs, h, w)
        # —— 调试打印：F_Recon 最终输出张量的静态 shape —— 
        print(f"[DEBUG] F_Recon.outputs tensor = {n.outputs}, static shape = {n.outputs.get_shape().as_list()}")

        return n
