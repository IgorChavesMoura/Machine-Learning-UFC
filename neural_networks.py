import numpy as np
import linear_regression_models as lin
import activation_functions

def mse_loss(y, pred):
    return np.mean((y - pred)**2)

def logistic_loss(y, pred):
    return np.mean(- y * np.log(pred) - (1 - y) * np.log(1 - pred)) 

def softmax_loss(y, pred):
    return - np.sum(y * np.log(pred)) / y.shape[1]

def mlp_forward(W_list, x,
                activation_function, out_activation_function,
                output_only=True):
    
    num_hidden_layers = len(W_list) - 1
    layer_u_list = []
    layer_z_list = []
    for h in range(num_hidden_layers+1):
        if h == 0:
            u = W_list[h] @ x
            z = np.vstack((np.ones((1, u.shape[1])), activation_function(u)))

        elif h < num_hidden_layers:
            u = W_list[h] @ layer_z_list[h-1]
            z = np.vstack((np.ones((1, u.shape[1])), activation_function(u)))

        else:
            u = W_list[h] @ layer_z_list[h-1]
            z = out_activation_function(u)

        layer_u_list.append(u)
        layer_z_list.append(z)
    
    if output_only:
        return layer_z_list[-1]
    else:
        return (layer_u_list, layer_z_list)
    
def mlp_backward(W_list, y,
                 grad_activation_function, out_grad_activation_function,
                 layer_u_list, layer_z_list):
    
    num_hidden_layers = len(W_list) - 1
    layer_delta_list = [None] * (num_hidden_layers + 1)
    for h in range(num_hidden_layers, -1, -1):
        if h < num_hidden_layers:
            delta = grad_activation_function(layer_u_list[h]) * \
                        (W_list[h+1][:,1:].T @ layer_delta_list[h+1])

        else:
            error = y - layer_z_list[-1]
            delta = error * out_grad_activation_function(layer_u_list[-1])

        layer_delta_list[h] = delta
        
    return layer_delta_list

def select_activation_function(activation):
    
    if activation == "relu":
        activation_function = activation_functions.relu
        grad_activation_function = activation_functions.grad_relu
    elif activation == "sigmoid":
        activation_function = activation_functions.sigmoid
        grad_activation_function = activation_functions.grad_sigmoid
    elif activation == "tanh":
        activation_function = activation_functions.tanh
        grad_activation_function = activation_functions.grad_tanh        
    
    return (activation_function, grad_activation_function)

def select_output_type(output, y):
    
    if output == "regression":
        loss = mse_loss
        out_activation_function = activation_functions.identity
        out_grad_activation_function = activation_functions.grad_identity
    elif output == "classification":
        loss = logistic_loss
        out_activation_function = activation_functions.sigmoid
        out_grad_activation_function = activation_functions.grad_sigmoid
        if y.shape[1] > 1:
            loss = softmax_loss
            out_activation_function = activation_functions.softmax
            out_grad_activation_function = activation_functions.grad_softmax
     
    return (out_activation_function, out_grad_activation_function, loss)

def initialize_weights(x_matrix, y, num_hidden_nodes, activation, output):
    
    num_hidden_layers = len(num_hidden_nodes)
    
    W_list = []
    w_init = (1/np.sqrt(x_matrix.shape[1])) * \
                np.random.normal(size=(num_hidden_nodes[0], x_matrix.shape[1]))
    w_init[:,0] = 1
    W_list.append(w_init)
    for h in range(1, num_hidden_layers):
        w_init = np.random.normal(size=(num_hidden_nodes[h], num_hidden_nodes[h-1] + 1))
        if activation == "relu":
            w_init *= (2/np.sqrt(num_hidden_nodes[h-1] + 1))
            w_init[:,0] = 0.01
        else:
            w_init *= (1/np.sqrt(num_hidden_nodes[h-1] + 1))
            w_init[:,0] = 0
        W_list.append(w_init)
    w_init = (1/np.sqrt(W_list[num_hidden_layers-1].shape[0] + 1)) * \
                np.random.normal(size=(y.shape[1], num_hidden_nodes[-1] + 1))
    if output == 'regression':
        w_init[:,0] = np.mean(y, axis=0)
    else:
        w_init[:,0] = 0
    W_list.append(w_init)
    
    return W_list

def mlp_train(x, y, x_validation=None, y_validation=None,
              num_hidden_nodes=10, activation="relu", output="regression",
              num_epochs=100, alpha=1, mini_batch_size=1, momentum=0, weight_decay=0,
              build_regressors=True, compute_loss=True):
    
    # Preprocess the input and output data
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
        if x_validation is not None:
            x_validation = lin.build_poly_regressors(x_validation)
    else:
        x_matrix = x  
        
    y = y.copy()         
    if len(y.shape) == 1:   
        y = y[:,None]
        
    if y_validation is not None:
        y_validation = y_validation.copy()         
        if len(y_validation.shape) == 1:   
            y_validation = y_validation[:,None]
    
    # Select the hidden and output activation functions    
        
    activation_function, grad_activation_function = select_activation_function(activation)
        
    out_activation_function, out_grad_activation_function, loss = select_output_type(output, y)   
    
    if type(num_hidden_nodes) is not list:
        num_hidden_nodes = [num_hidden_nodes]
    
    num_hidden_layers = len(num_hidden_nodes)
    
    # Initialize the weights
    W_list = initialize_weights(x_matrix, y, num_hidden_nodes, activation, output)
    
    loss_history = []
    validation_loss_history = []
    past_updates = [0] * (num_hidden_layers + 1)
    for epoch in range(num_epochs):
        
        random_permutation = np.array_split(np.random.permutation(y.shape[0]),
                                            np.ceil(y.shape[0] / mini_batch_size))
        
        for i in random_permutation:  
            
            xi = x_matrix[i].T
            yi = y[i].T
            
            # Forward pass
            layer_u_list, layer_z_list = mlp_forward(W_list, xi,
                                                     activation_function, out_activation_function,
                                                     output_only=False)
            
            # Backward pass
            layer_delta_list = mlp_backward(W_list, yi,
                                            grad_activation_function, out_grad_activation_function,
                                            layer_u_list, layer_z_list)
                
            # Update the weights
            for h in range(num_hidden_layers+1):    
                if h == 0:                    
                    layer_input = xi.T
                else:
                    layer_input = layer_z_list[h-1].T
                    
                delta_weight = (alpha / xi.shape[1]) * layer_delta_list[h] @ layer_input + \
                                momentum * past_updates[h] - alpha * weight_decay * W_list[h]
                    
                W_list[h] += delta_weight
                    
                past_updates[h] = delta_weight
        
        if compute_loss:
            
            model_output = mlp_forward(W_list, x_matrix.T,
                                       activation_function, out_activation_function,
                                       output_only=True)
            loss_history.append(loss(y.T, model_output) + 0.5 * weight_decay * np.array([(W**2).sum() for W in W_list]).sum())
            
            if x_validation is not None:
                model_output_validation = mlp_forward(W_list, x_validation.T,
                                                      activation_function, out_activation_function,
                                                      output_only=True)
                validation_loss_history.append(loss(y_validation.T, model_output_validation) + \
                                               0.5 * weight_decay * np.array([(W**2).sum() for W in W_list]).sum())
    
    return {'W_list': W_list, 'loss_history': loss_history, 'validation_loss_history': validation_loss_history,
            'output': output, 'activation_function': activation_function, 'out_activation_function': out_activation_function}
    
def mlp_predict(model, x, build_regressors=True, return_class=True):
        
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x
        
    model_output = mlp_forward(model['W_list'], x_matrix.T,
                               model['activation_function'], model['out_activation_function'],
                               output_only=True)
    
    if return_class and model['output'] != 'regression': 
        if model_output.shape[0] == 1:
            return np.maximum(0, np.sign(model_output[0] - 0.5))
        else:
            return np.argmax(model_output, axis=0)
    else:
        return model_output

def train_perceptron(x, y, num_epochs=100, alpha=1, w_initial=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x        
        
    y = y.copy()
    y[y == 0] = -1    
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial.copy()
    
    loss_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):
            error = yi - np.sign(xi @ w)      
            w += alpha * xi[:,None] @ error[:,None].T                
        loss_history.append(np.sum(np.maximum(0, -(y * np.sign(x_matrix @ w)))))
        
    return {'w': w, 'loss_history': loss_history}

def train_adaline(x, y, num_epochs=100, alpha=0.001, w_initial=None, build_regressors=True):
    
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x   
        
    if len(y.shape) == 1:   
        y = y[:,None]
    
    if w_initial is None:
        w = np.zeros((x_matrix.shape[1], y.shape[1]))
    else:
        w = w_initial.copy()
    
    loss_history = []
    for epoch in range(num_epochs):
        random_permutation = np.random.permutation(y.shape[0])
        for xi, yi in zip(x_matrix[random_permutation], y[random_permutation]):
            error = yi - xi @ w      
            w += alpha * xi[:,None] @ error[:,None].T   
        loss_history.append(np.mean((y - x_matrix @ w)**2))
        
    return {'w': w, 'loss_history': loss_history}

def linear_class_predic(model, x, build_regressors=True):
        
    if build_regressors:
        x_matrix = lin.build_poly_regressors(x)    
    else:
        x_matrix = x
    
    if model['w'].shape[1] > 1:    
        return np.argmax(x_matrix @ model['w'], axis=1)
    else:
        return np.maximum(0, np.sign(x_matrix @ model['w']))
    