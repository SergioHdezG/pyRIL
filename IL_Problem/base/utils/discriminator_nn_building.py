from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
from termcolor import colored
import numpy as np

def read_disc_net_params(net_architecture):
    id_1 = 'state_conv_layers'
    id_2 = 'state_kernel_num'
    id_3 = 'state_kernel_size'
    id_4 = 'state_kernel_strides'
    id_5 = 'state_conv_activation'
    id_6 = 'state_dense_lay'
    id_7 = 'state_n_neurons'
    id_8 = 'state_dense_activation'

    id_9 = 'action_dense_lay'
    id_10 = 'action_n_neurons'
    id_11 = 'action_dense_activation'

    id_12 = 'common_dense_lay'
    id_13 = 'common_n_neurons'
    id_14 = 'common_dense_activation'
    id_19 = 'last_layer_activation'

    id_15 = 'use_custom_network'
    id_16 = 'state_custom_network'
    id_17 = 'action_custom_network'
    id_18 = 'common_custom_network'
    id_20 = 'define_custom_output_layer'
    id_21 = 'tf_custom_model'
    id_22 = 'use_tf_custom_model'

    # TODO: formular correctamente los warnings
    if (id_1 and id_2 and id_3 and id_4 and id_5 in net_architecture) \
            and (net_architecture[id_1] and net_architecture[id_2] and net_architecture[id_3] and net_architecture[id_4]
                 and net_architecture[id_5] is not None):

        state_n_conv_layers = net_architecture[id_1]
        state_kernel_num = net_architecture[id_2]
        state_kernel_size = net_architecture[id_3]
        state_strides = net_architecture[id_4]
        state_conv_activation = net_architecture[id_5]
        # print('Convolutional layers for state net selected: {state_conv_layers:\t\t', state_n_conv_layers,
        #       '\n\t\t\t\t\t\t\t    state_kernel_num:\t\t\t', state_kernel_num, '\n\t\t\t\t\t\t\t    '
        #                                                                        'state_kernel_size:\t\t',
        #       state_kernel_size, '\n\t\t\t\t\t\t\t    state_kernel_strides:\t\t',
        #       state_strides, '\n\t\t\t\t\t\t\t    state_conv_activation:\t', state_conv_activation, '}')
    else:
        state_n_conv_layers = None
        state_kernel_num = None
        state_kernel_size = None
        state_strides = None
        state_conv_activation = None
        # print(colored('WARNING: If you want to specify convolutional layers for state net you must set all the values '
        #               'for the following keys: state_conv_layers, state_kernel_num, state_kernel_size, '
        #               'state_kernel_strides and state_conv_activation', 'yellow'))

    if (id_6 and id_7 and id_8 in net_architecture) and (net_architecture[id_6] and net_architecture[id_7]
                                                         and net_architecture[id_8] is not None):
        state_n_dense_layers = net_architecture[id_6]
        state_n_neurons = net_architecture[id_7]
        state_dense_activation = net_architecture[id_8]
        # print('Dense layers for state net selected: {state_dense_lay:\t\t\t', state_n_dense_layers, '\n\t\t\t\t\t    '
        #                                                                                             'state_n_neurons:\t\t\t',
        #       state_n_neurons, '\n\t\t\t\t\t    dense_activation:\t', state_dense_activation,
        #       '}')
    else:
        state_n_dense_layers = None
        state_n_neurons = None
        state_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for state net you must set all the values for the '
        #               'following keys: state_dense_lay, state_n_neurons and state_dense_activation', 'yellow'))

    if (id_9 and id_10 and id_11 in net_architecture) and (net_architecture[id_9] and net_architecture[id_10]
                                                           and net_architecture[id_11] is not None):
        action_n_dense_layers = net_architecture[id_9]
        action_n_neurons = net_architecture[id_10]
        action_dense_activation = net_architecture[id_11]
        # print('Dense layers for action net selected: {action_dense_lay:\t\t\t', action_n_dense_layers, '\n\t\t\t\t\t   '
        #                                                                                                'action_n_neurons:\t\t\t',
        #       action_n_neurons, '\n\t\t\t\t\t    action_activation:\t',
        #       action_dense_activation, '}')
    else:
        action_n_dense_layers = None
        action_n_neurons = None
        action_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for action net you must set all the values for the '
        #               'following keys: action_dense_lay, action_n_neurons and action_dense_activation', 'yellow'))

    if (id_12 and id_13 and id_14 in net_architecture) and (net_architecture[id_12] and net_architecture[id_13]
                                                            and net_architecture[id_14] is not None):
        common_n_dense_layers = net_architecture[id_12]
        common_n_neurons = net_architecture[id_13]
        common_dense_activation = net_architecture[id_14]
        # print('Dense layers for common net selected: {common_dense_lay:\t\t\t', common_n_dense_layers, '\n\t\t\t\t\t   '
        #                                                                                                'common_n_neurons:\t\t\t',
        #       common_n_neurons, '\n\t\t\t\t\t    common_activation:\t',
        #       common_dense_activation, '}')
    else:
        common_n_dense_layers = None
        common_n_neurons = None
        common_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for common net you must set all the values for the '
        #               'following keys: common_dense_lay, common_n_neurons and common_dense_activation', 'yellow'))

    if (id_15 and id_16 and id_17 and id_18 and id_20 in net_architecture) and \
            (net_architecture[id_15] and net_architecture[id_18]
             is not None):
        use_custom_net = net_architecture[id_15]
        state_custom_net = net_architecture[id_16]
        action_custom_net = net_architecture[id_17]
        common_custom_net = net_architecture[id_18]
        define_custom_output_layer = net_architecture[id_20]
        # print('Custom network option selected: {use_custom_network: ', use_custom_net, ', state_custom_network: ',
        #       state_custom_net, ', action_custom_network: ', action_custom_net, ', common_custom_network: ',
        #       common_custom_net, '}')
    else:
        use_custom_net = False
        state_custom_net = None
        action_custom_net = None
        common_custom_net = None
        define_custom_output_layer = False
        # print(colored('WARNING: If you want to use a custom neural net you must set the values at least for the '
        #               'following keys: use_custom_network and common_custom_network, and additionally for: '
        #               'state_custom_network and action_custom_network', 'yellow'))

    if (id_19 in net_architecture) and (net_architecture[id_19] is not None):
        last_layer_activation = net_architecture[id_19]
        # print('Last layer activation: {last_layer_activation: ', last_layer_activation, '}')
    else:
        last_layer_activation = 'sigmoid'
        # print(colored('WARNING: Last layer activation function was not specified, sigmoid activation is selected by '
        #               'default.', 'yellow'))

    if (id_21 and id_22 in net_architecture) and (net_architecture[id_21] and net_architecture[id_22] is not None):
        use_custom_net_tf = net_architecture[id_22]
        if use_custom_net_tf:
            use_custom_net = True
            common_custom_net = net_architecture[id_21]
    else:
        use_custom_net_tf = False

    return state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
           state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
           action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
           common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
           last_layer_activation, define_custom_output_layer, use_custom_net_tf


def read_disc_net_params_legacy(net_architecture):
    id_1 = 'state_conv_layers'
    id_2 = 'state_kernel_num'
    id_3 = 'state_kernel_size'
    id_4 = 'state_kernel_strides'
    id_5 = 'state_conv_activation'
    id_6 = 'state_dense_lay'
    id_7 = 'state_n_neurons'
    id_8 = 'state_dense_activation'

    id_9 = 'action_dense_lay'
    id_10 = 'action_n_neurons'
    id_11 = 'action_dense_activation'

    id_12 = 'common_dense_lay'
    id_13 = 'common_n_neurons'
    id_14 = 'common_dense_activation'
    id_19 = 'last_layer_activation'

    id_15 = 'use_custom_network'
    id_16 = 'state_custom_network'
    id_17 = 'action_custom_network'
    id_18 = 'common_custom_network'
    id_20 = 'define_custom_output_layer'
    id_21 = 'tf_custom_model'
    id_22 = 'use_tf_custom_model'

    if (id_1 and id_2 and id_3 and id_4 and id_5 in net_architecture) \
            and (net_architecture[id_1] and net_architecture[id_2] and net_architecture[id_3] and net_architecture[id_4]
                 and net_architecture[id_5] is not None):

        state_n_conv_layers = net_architecture[id_1]
        state_kernel_num = net_architecture[id_2]
        state_kernel_size = net_architecture[id_3]
        state_strides = net_architecture[id_4]
        state_conv_activation = net_architecture[id_5]
        # print('Convolutional layers for state net selected: {state_conv_layers:\t\t', state_n_conv_layers,
        #       '\n\t\t\t\t\t\t\t    state_kernel_num:\t\t\t', state_kernel_num, '\n\t\t\t\t\t\t\t    '
        #                                                                        'state_kernel_size:\t\t',
        #       state_kernel_size, '\n\t\t\t\t\t\t\t    state_kernel_strides:\t\t',
        #       state_strides, '\n\t\t\t\t\t\t\t    state_conv_activation:\t', state_conv_activation, '}')
    else:
        state_n_conv_layers = None
        state_kernel_num = None
        state_kernel_size = None
        state_strides = None
        state_conv_activation = None
        # print(colored('WARNING: If you want to specify convolutional layers for state net you must set all the values '
        #               'for the following keys: state_conv_layers, state_kernel_num, state_kernel_size, '
        #               'state_kernel_strides and state_conv_activation', 'yellow'))

    if (id_6 and id_7 and id_8 in net_architecture) and (net_architecture[id_6] and net_architecture[id_7]
                                                         and net_architecture[id_8] is not None):
        state_n_dense_layers = net_architecture[id_6]
        state_n_neurons = net_architecture[id_7]
        state_dense_activation = net_architecture[id_8]
        # print('Dense layers for state net selected: {state_dense_lay:\t\t\t', state_n_dense_layers, '\n\t\t\t\t\t    '
        #                                                                                             'state_n_neurons:\t\t\t',
        #       state_n_neurons, '\n\t\t\t\t\t    dense_activation:\t', state_dense_activation,
        #       '}')
    else:
        state_n_dense_layers = None
        state_n_neurons = None
        state_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for state net you must set all the values for the '
        #               'following keys: state_dense_lay, state_n_neurons and state_dense_activation', 'yellow'))

    if (id_9 and id_10 and id_11 in net_architecture) and (net_architecture[id_9] and net_architecture[id_10]
                                                           and net_architecture[id_11] is not None):
        action_n_dense_layers = net_architecture[id_9]
        action_n_neurons = net_architecture[id_10]
        action_dense_activation = net_architecture[id_11]
        # print('Dense layers for action net selected: {action_dense_lay:\t\t\t', action_n_dense_layers, '\n\t\t\t\t\t   '
        #                                                                                                'action_n_neurons:\t\t\t',
        #       action_n_neurons, '\n\t\t\t\t\t    action_activation:\t',
        #       action_dense_activation, '}')
    else:
        action_n_dense_layers = None
        action_n_neurons = None
        action_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for action net you must set all the values for the '
        #               'following keys: action_dense_lay, action_n_neurons and action_dense_activation', 'yellow'))

    if (id_12 and id_13 and id_14 in net_architecture) and (net_architecture[id_12] and net_architecture[id_13]
                                                            and net_architecture[id_14] is not None):
        common_n_dense_layers = net_architecture[id_12]
        common_n_neurons = net_architecture[id_13]
        common_dense_activation = net_architecture[id_14]
        # print('Dense layers for common net selected: {common_dense_lay:\t\t\t', common_n_dense_layers, '\n\t\t\t\t\t   '
        #                                                                                                'common_n_neurons:\t\t\t',
        #       common_n_neurons, '\n\t\t\t\t\t    common_activation:\t',
        #       common_dense_activation, '}')
    else:
        common_n_dense_layers = None
        common_n_neurons = None
        common_dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers for common net you must set all the values for the '
        #               'following keys: common_dense_lay, common_n_neurons and common_dense_activation', 'yellow'))

    if (id_15 and id_16 and id_17 and id_18 and id_20 in net_architecture) and \
            (net_architecture[id_15] and net_architecture[id_18]
             is not None):
        use_custom_net = net_architecture[id_15]
        state_custom_net = net_architecture[id_16]
        action_custom_net = net_architecture[id_17]
        common_custom_net = net_architecture[id_18]
        define_custom_output_layer = net_architecture[id_20]
        # print('Custom network option selected: {use_custom_network: ', use_custom_net, ', state_custom_network: ',
        #       state_custom_net, ', action_custom_network: ', action_custom_net, ', common_custom_network: ',
        #       common_custom_net, '}')
    else:
        use_custom_net = False
        state_custom_net = None
        action_custom_net = None
        common_custom_net = None
        define_custom_output_layer = False
        # print(colored('WARNING: If you want to use a custom neural net you must set the values at least for the '
        #               'following keys: use_custom_network and common_custom_network, and additionally for: '
        #               'state_custom_network and action_custom_network', 'yellow'))

    if (id_19 in net_architecture) and (net_architecture[id_19] is not None):
        last_layer_activation = net_architecture[id_19]
        # print('Last layer activation: {last_layer_activation: ', last_layer_activation, '}')
    else:
        last_layer_activation = 'sigmoid'
        # print(colored('WARNING: Last layer activation function was not specified, sigmoid activation is selected by '
        #               'default.', 'yellow'))

    if (id_21 and id_22 in net_architecture) and (net_architecture[id_21] and net_architecture[id_22] is not None):
        use_custom_net = net_architecture[id_22]
        custom_net = net_architecture[id_21]

    return state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
           state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
           action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
           common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
           last_layer_activation, define_custom_output_layer

def build_disc_conv_net(net_architecture, input_shape, n_actions, use_expert_actions=True):
    state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
    state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
    action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
    common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
    last_layer_activation, define_custom_output_layer, use_custom_net_tf = read_disc_net_params(net_architecture)

    if use_custom_net:
        if use_custom_net_tf:
            if use_expert_actions:
                return common_custom_net([input_shape, n_actions])
            else:
                return common_custom_net((input_shape))
        else:
            if state_custom_net is not None:
                state_model = state_custom_net(input_shape)
                state_out_size = [int(x) for x in state_model.output.shape[1:].dims]
            else:
                state_model = Flatten(input_shape=input_shape)
                state_out_size = [input_shape[-3] * input_shape[-2] * input_shape[-1]]
            if use_expert_actions:
                if action_custom_net is not None:
                    action_model = action_custom_net((n_actions,))
                    action_out_size = action_model.output.shape[-1]
                else:
                    action_model = _dummy_model(n_actions)
                    action_out_size = n_actions

                state_input = tf.keras.layers.Input(input_shape)
                action_input = tf.keras.layers.Input(n_actions)

                common_size = state_out_size[0] + action_out_size
                common_model = common_custom_net((common_size,))

                concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model(state_input), action_model(action_input)])
                output = common_model(concat)
                model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=output)

            else:
                state_input = tf.keras.layers.Input(input_shape)
                common_model = common_custom_net(state_out_size)
                output = common_model(state_model(state_input))
                model = tf.keras.models.Model(inputs=state_input, outputs=output)

    else:
        # build state network
        state_model = Sequential()
        state_model.add(Conv2D(state_kernel_num[0], kernel_size=state_kernel_size[0], input_shape=input_shape,
                               strides=(state_strides[0], state_strides[0]), padding='same',
                               activation=state_conv_activation[0]))
        for i in range(1, state_n_conv_layers):
            state_model.add(Conv2D(state_kernel_num[i], kernel_size=state_kernel_size[i],
                                   strides=(state_strides[i], state_strides[i]), padding='same',
                                   activation=state_conv_activation[i]))
        state_model.add(Flatten())
        state_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0],
                              name='disc_state_dense_0'))

        for i in range(1, state_n_dense_layers):
            state_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                  name='disc_state_dense_'+str(i)))

        # build action network
        if use_expert_actions:
            action_model = Sequential()
            action_model.add(Dense(state_n_neurons[0], input_dim=n_actions, activation=state_dense_activation[0],
                                   name='disc_action_dense_0'))

            for i in range(1, state_n_dense_layers):
                action_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                       name='disc_action_dense_'+str(i)))

        # input_common = action_model.output.shape[-1] + state_model.output.shape[-1]
        else:
            action_model = None
        # TODO: ver si es mejor opcion hacer que reciba dos vectores en lugar de uno como concatenacion de state+actions
        # build common network
        common_model = Sequential()
        common_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0], name='disc_common_dense_0'))

        for i in range(1, state_n_dense_layers):
            common_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                   name='disc_common_dense_'+str(i)))

        if use_expert_actions:
            concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model.outputs, action_model.outputs])
        else:
            concat = state_model.outputs

        output = common_model(concat)

        if use_expert_actions:
            model = tf.keras.models.Model(inputs=[state_model.inputs, action_model.inputs], outputs=output)
        else:
            model = tf.keras.models.Model(inputs=state_model.inputs, outputs=output)
    return model
    # return tf.keras.models.Sequential(model)

def build_disc_stack_nn_net(net_architecture, input_shape, n_actions, use_expert_actions=True):
    state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
    state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
    action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
    common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
    last_layer_activation, define_custom_output_layer, use_custom_net_tf = read_disc_net_params(net_architecture)

    if use_custom_net:
        if use_custom_net_tf:
            if use_expert_actions:
                return common_custom_net([input_shape, n_actions])
            else:
                return common_custom_net((input_shape))
        else:
            if state_custom_net is not None:
                state_model = state_custom_net(input_shape)
                state_out_size = tuple([int(x) for x in state_model.output.shape[1:].dims])
            else:
                state_model = Flatten(input_shape=input_shape)
                state_out_size = input_shape[-2] * input_shape[-1]
            if use_expert_actions:
                if action_custom_net is not None:
                    action_model = action_custom_net((n_actions,))
                    action_out_size = action_model.output.shape[-1]
                else:
                    action_model = _dummy_model(n_actions)
                    action_out_size = n_actions

                state_input = tf.keras.layers.Input(input_shape)
                action_input = tf.keras.layers.Input(n_actions)

                common_size = state_out_size + action_out_size
                common_model = common_custom_net((common_size,))

                concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model(state_input), action_model(action_input)])
                output = common_model(concat)
                model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=output)

            else:
                state_input = tf.keras.layers.Input(input_shape)
                common_model = common_custom_net((state_out_size,))
                output = common_model(state_model(state_input))
                model = tf.keras.models.Model(inputs=state_input, outputs=output)

    else:
        # build state network
        state_model = Sequential()
        state_model.add(Flatten(input_shape=input_shape))
        state_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0],
                              name='disc_state_dense_0'))

        for i in range(1, state_n_dense_layers):
            state_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                  name='disc_state_dense_'+str(i)))

        # build action network
        if use_expert_actions:
            action_model = Sequential()
            action_model.add(Dense(state_n_neurons[0], input_dim=n_actions, activation=state_dense_activation[0],
                                   name='disc_action_dense_0'))

            for i in range(1, state_n_dense_layers):
                action_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                       name='disc_action_dense_'+str(i)))

        # input_common = action_model.output.shape[-1] + state_model.output.shape[-1]
        else:
            action_model = None

        # build common network
        common_model = Sequential()
        common_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0], name='disc_common_dense_0'))

        for i in range(1, state_n_dense_layers):
            common_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                   name='disc_common_dense_'+str(i)))

        if use_expert_actions:
            concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model.outputs, action_model.outputs])
        else:
            concat = state_model.outputs

        output = common_model(concat)

        if use_expert_actions:
            model = tf.keras.models.Model(inputs=[state_model.inputs, action_model.inputs], outputs=output)
        else:
            model = tf.keras.models.Model(inputs=state_model.inputs, outputs=output)
    return model
    # return tf.keras.models.Sequential(model)

def build_disc_nn_net(net_architecture, input_shape, n_actions, use_expert_actions=True):
    state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
    state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
    action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
    common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
    last_layer_activation, define_custom_output_layer, use_custom_net_tf = read_disc_net_params(net_architecture)

    if use_custom_net:
        if use_custom_net_tf:
            if use_expert_actions:
                return common_custom_net([input_shape, n_actions])
            else:
                return common_custom_net((input_shape))

        else:
            if state_custom_net is not None:
                state_model = state_custom_net(input_shape)
                state_out_size = state_model.output.shape[-1]
            else:
                state_model = _dummy_model(input_shape)
                state_out_size = input_shape[-1]
            if use_expert_actions:
                if action_custom_net is not None:
                    action_model = action_custom_net((n_actions,))
                    action_out_size = action_model.output.shape[-1]
                else:
                    action_model = _dummy_model(n_actions)
                    action_out_size = n_actions

                state_input = tf.keras.layers.Input(input_shape)
                action_input = tf.keras.layers.Input(n_actions)

                common_size = state_out_size + action_out_size
                common_model = common_custom_net((common_size,))

                concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model(state_input), action_model(action_input)])
                output = common_model(concat)
                model = tf.keras.models.Model(inputs=[state_input, action_input], outputs=output)

            else:
                state_input = tf.keras.layers.Input(input_shape)
                common_model = common_custom_net((state_out_size,))
                output = common_model(state_model(state_input))
                model = tf.keras.models.Model(inputs=state_input, outputs=output)

    else:
        # build state network
        state_model = Sequential()

        state_model.add(Dense(state_n_neurons[0], input_dim=input_shape, activation=state_dense_activation[0],
                              name='disc_state_dense_0'))

        for i in range(1, state_n_dense_layers):
            state_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                  name='disc_state_dense_'+str(i)))

        # build action network
        if use_expert_actions:
            action_model = Sequential()
            action_model.add(Dense(state_n_neurons[0], input_dim=n_actions, activation=state_dense_activation[0],
                                   name='disc_action_dense_0'))

            for i in range(1, state_n_dense_layers):
                action_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                       name='disc_action_dense_'+str(i)))

        # input_common = action_model.output.shape[-1] + state_model.output.shape[-1]
        else:
            action_model = None

        # build common network
        common_model = Sequential()
        common_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0], name='disc_common_dense_0'))

        for i in range(1, state_n_dense_layers):
            common_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                   name='disc_common_dense_'+str(i)))

        if use_expert_actions:
            concat = tf.keras.layers.Concatenate(axis=1, name='disc_concatenate')([state_model.outputs, action_model.outputs])
        else:
            concat = state_model.outputs

        output = common_model(concat)

        if use_expert_actions:
            model = tf.keras.models.Model(inputs=[state_model.inputs, action_model.inputs], outputs=output)
        else:
            model = tf.keras.models.Model(inputs=state_model.inputs, outputs=output)
    return model
    # return tf.keras.models.Sequential(model)

def build_disc_nn_net_legacy(net_architecture, state_shape, n_actions, img_input=False, use_expert_actions=True):
    state_n_conv_layers, state_kernel_num, state_kernel_size, state_strides, state_conv_activation, \
    state_n_dense_layers, state_n_neurons, state_dense_activation, use_custom_net, action_n_dense_layers, \
    action_n_neurons, action_dense_activation, common_n_dense_layers, common_n_neurons, \
    common_dense_activation, use_custom_net, state_custom_net, action_custom_net, common_custom_net, \
    last_layer_activation, define_custom_output_layer = read_disc_net_params_legacy(net_architecture)

    if not img_input:
        stack = len(state_shape) > 1
    else:
        stack = False

    if use_custom_net:
        if state_custom_net is not None:
            state_model = state_custom_net(state_shape)
            state_out_size = state_model.output.shape[-1]
        else:
            if stack:
                state_model = Flatten(input_shape=state_shape)
                state_out_size = state_shape[-2] * state_shape[-1]
            else:
                state_model = _dummy_model
                state_out_size = state_shape[-1]
        if use_expert_actions:
            if action_custom_net is not None:
                action_model = action_custom_net((n_actions,))
                action_out_size = action_model.output.shape[-1]
            else:
                action_model = _dummy_model
                action_out_size = n_actions
            common_size = state_out_size + action_out_size
        else:
            action_model = None
            common_size = state_out_size

        common_model = common_custom_net((common_size,))
    else:
        if not stack and not img_input:
            # Extract an integer from a tuple
            state_shape = state_shape[0]

        # build state network
        state_model = Sequential()
        if img_input:
            state_model.add(Conv2D(state_kernel_num[0], kernel_size=state_kernel_size[0], input_shape=state_shape,
                                   strides=(state_strides[0], state_strides[0]), padding='same',
                                   activation=state_conv_activation[0]))
            for i in range(1, state_n_conv_layers):
                state_model.add(Conv2D(state_kernel_num[i], kernel_size=state_kernel_size[i],
                                       strides=(state_strides[i], state_strides[i]), padding='same',
                                       activation=state_conv_activation[i]))
            state_model.add(Flatten())

        elif stack:
            state_model.add(Flatten(input_shape=state_shape))

        state_model.add(Dense(state_n_neurons[0], input_dim=state_shape, activation=state_dense_activation[0],
                              name='disc_state_dense_0'))

        for i in range(1, state_n_dense_layers):
            state_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                  name='disc_state_dense_'+str(i)))

        # build action network
        if use_expert_actions:
            action_model = Sequential()
            action_model.add(Dense(state_n_neurons[0], input_dim=n_actions, activation=state_dense_activation[0],
                                   name='disc_action_dense_0'))

            for i in range(1, state_n_dense_layers):
                action_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                       name='disc_action_dense_'+str(i)))

        # input_common = action_model.output.shape[-1] + state_model.output.shape[-1]
        else:
            action_model = None

        # build common network
        common_model = Sequential()
        common_model.add(Dense(state_n_neurons[0], activation=state_dense_activation[0], name='disc_common_dense_0'))

        for i in range(1, state_n_dense_layers):
            common_model.add(Dense(state_n_neurons[i], activation=state_dense_activation[i],
                                   name='disc_common_dense_'+str(i)))

    return state_model, action_model, common_model, last_layer_activation, define_custom_output_layer

def _dummy_model(input_shape):
    dummy_input = tf.keras.layers.Input(input_shape)
    # dummy.add(tf.keras.layers.Lambda(lambda x: x))
    # dummy.add(tf.keras.layers.Dense(10))
    dummy = tf.keras.models.Model(inputs=dummy_input, outputs=dummy_input)
    return dummy

# def _dummy_model(input):
#     return input
