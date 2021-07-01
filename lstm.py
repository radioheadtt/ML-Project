import torch
batch_size, num_steps = 32, 35
def get_lstm_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape) * 0.01

    def three():
        return (normal(
            (num_inputs, num_hiddens)), normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens))

    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()  
    W_xo, W_ho, b_o = three() 
    W_xc, W_hc, b_c = three()  
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs)
    params = [
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
        W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
def init_lstm_state(batch_size, num_hiddens):
    return (torch.zeros((batch_size, num_hiddens)),
            torch.zeros((batch_size, num_hiddens)))