def irl_discriminator_net(state_conv_layers=None, state_kernel_num=None, state_kernel_size=None,
                          state_kernel_strides=None, state_conv_activation=None, state_dense_lay=None,
                          state_n_neurons=None, state_dense_activation=None,
                          action_dense_lay=None, action_n_neurons=None, action_dense_activation=None,
                          common_dense_lay=None, common_n_neurons=None, common_dense_activation=None,
                          use_custom_network=None, state_custom_network=None, action_custom_network=False,
                          common_custom_network=None, last_layer_activation=None, define_custom_output_layer=False,
                          tf_custom_model=None, use_tf_custom_model=False):
    """
    Here you can define the architecture of your model from input layer to last hidden layer. The output layer wil be
    created by the agent depending on the number of outputs and the algorithm used.
    :param state_conv_layers:       int for example 3. Number of convolutional layers on agent net.
    :param state_kernel_num:        array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    agent net.
    :param state_kernel_size:       array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    agent net.
    :param state_kernel_strides:    array of ints, for example [4, 2, 2]. Stride for each conv layer on agent net.
    :param state_conv_activation:   array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on agent net.
    :param state_dense_lay:      int, for example 2. Number of dense layers on agent net.
    :param state_n_neurons:         array of ints, for example [1024, 1024]. Number of neurons for each dense layer on
                                    agent net.
    :param state_dense_activation:  array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on agent net.
    :param action_dense_lay:      int for example 3. Number of convolutional layers on critic net.
    :param action_n_neurons:       array od ints, for example [32, 32, 64]. Number of conv kernel for each layer on
                                    critic net.
    :param action_dense_activation:      array of ints, for example [7, 5, 3]. Size of each conv kernel for each layer on
                                    critic net.
    :param common_dense_lay:   array of ints, for example [4, 2, 2]. Stride for each conv layer on critic net.
    :param common_n_neurons:  array of string, for example ['relu', 'relu', 'relu']. Activation function for each
                                    conv layer on critic net.
    :param common_dense_activation:     int, for example 2. Number of dense layers on critic net.
    :param use_custom_network:          boolean. Set True if you are going to use a custom external network with your own
                                    architecture. Use this together with actor_custom_network and critic_custom_network.
                                    Default values = False
    :param state_custom_network: array of string, for example ['relu', 'relu']. Activation function for each dense
                                    layer on critic net.
    :param action_custom_network:
    :param common_custom_network:    Model to be used for actor. Value based agent use keras models, for the other agents
                                    you have to return the network as a tensor flow graph. These models have to be an
                                    object returned by a function.
    :param last_layer_activation:   Model to be used for critic. Value based agent use keras models, for the other
                                    agents you have to return the network as a tensor flow graph. These models have to
                                    be an object returned by a function.
    :param define_custom_output_layer: True if the custom model defines the outputs layer for the discriminator in common_custom_network.
    :param tf_custom_model:         Function to create the custom tensorflow model. The model must inherit from
                                    RL_Agent.base.utils.networks.networks_interface.RLNetModel. The function must
                                    receive the input shape of the model.
    :param use_tf_custom_model:     boolean. Set True if you are going to use a custom external tensorflow model with
                                    your own architecture inheriting from
                                    RL_Agent.base.utils.networks.networks_interface.RLNetModel.
                                    Use this together with tf_custom_model.
                                    Default values = False
                                    tf_custom_model and use_custom_network can not have True value at same time
    :return: dictionary
    """
    if use_custom_network and use_tf_custom_model:
        raise Exception("use_custom_network and use_tf_custom_model can not have True value at the same time. Please, "
                        "set at least one of those to False.")
    net_architecture = {
           "state_conv_layers": state_conv_layers,
           "state_kernel_num": state_kernel_num,
           "state_kernel_size": state_kernel_size,
           "state_kernel_strides": state_kernel_strides,
           "state_conv_activation": state_conv_activation,

           "state_dense_lay": state_dense_lay,
           "state_n_neurons": state_n_neurons,
           "state_dense_activation": state_dense_activation,

           "action_dense_lay": action_dense_lay,
           "action_n_neurons": action_n_neurons,
           "action_dense_activation": action_dense_activation,

           "common_dense_lay": common_dense_lay,
           "common_n_neurons": common_n_neurons,
           "common_dense_activation": common_dense_activation,
           "last_layer_activation": last_layer_activation,

           "use_custom_network": use_custom_network,
           "state_custom_network": state_custom_network,
           "action_custom_network": action_custom_network,
           "common_custom_network": common_custom_network,
           "define_custom_output_layer": define_custom_output_layer if use_custom_network else False,
           "tf_custom_model": tf_custom_model,
           "use_tf_custom_model": use_tf_custom_model
           }
    return net_architecture