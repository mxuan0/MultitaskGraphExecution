import torch

import collections

import model as arch

def processor_init(model, model_state):
    if isinstance(model, arch.NeuralExecutor2Transfer) or isinstance(model, arch.NeuralExecutor3Transfer):
        model.reset_parameters()
        proc_model_state = collections.OrderedDict()
        term_model_state = collections.OrderedDict()
        term_mpnn_model_state = collections.OrderedDict()
        for k, v in model_state.items():
            if k.startswith('processor'):
                proc_model_state[k[10:]] = v
            if  k.startswith('termination'):
                term_model_state[k[12:]] = v
            if  k.startswith('termination_mpnn'):
                term_model_state[k[17:]] = v
        model.processor.processors[0].load_state_dict(proc_model_state, strict=False)
        model.processor.processors[0].requires_grad_(False)
        model.termination.load_state_dict(term_model_state, strict=False)
        model.termination_mpnn.load_state_dict(term_mpnn_model_state, strict=False)
    else:
        new_model_state = collections.OrderedDict()
        for k, v in model_state.items():
            if k.startswith('processor') or k.startswith('termination'):
                new_model_state[k] = v
        model.load_state_dict(new_model_state, strict=False)
        if isinstance(model, arch.NeuralExecutor2Freeze) or isinstance(model, arch.NeuralExecutor3Freeze):
            model.processor.requires_grad_(False)
    return model

def merge_processor_init(init_model, target_model, joint_model):
    if isinstance(joint_model, arch.NeuralExecutor2Transfer):
        joint_model.reset_parameters()
        ## target model enc/dec + proc
        joint_model.load_state_dict(target_model.state_dict(),strict=False)
        proc_model_state = collections.OrderedDict()
        for k, v in target_model.state_dict().items():
            if k.startswith('processor'):
                proc_model_state[k[10:]] = v
        joint_model.processor.processors[1].load_state_dict(proc_model_state, strict=False)
        joint_model.processor.processors[1].requires_grad_(True)
        ## init model proc + term (freeze)
        proc_model_state = collections.OrderedDict()
        term_model_state = collections.OrderedDict()
        term_mpnn_model_state = collections.OrderedDict()
        for k, v in init_model.state_dict().items():
            if k.startswith('processor'):
                proc_model_state[k[10:]] = v
            if  k.startswith('termination'):
                term_model_state[k[12:]] = v
            if  k.startswith('termination_mpnn'):
                term_model_state[k[17:]] = v
        joint_model.processor.processors[0].load_state_dict(proc_model_state, strict=False)
        joint_model.processor.processors[0].requires_grad_(False)
        joint_model.termination.load_state_dict(term_model_state, strict=False)
        joint_model.termination_mpnn.load_state_dict(term_mpnn_model_state, strict=False)
    else:
        raise NotImplementedError
    return joint_model

def distill_processor(source_model, target_model):
    if isinstance(target_model, arch.NeuralExecutor2):
        target_model.reset_parameters()
        encdec_model_state = collections.OrderedDict()
        for k, v in source_model.state_dict().items():
            if 'encoder' in k or 'decoder' in k or 'termination' in k or 'predecessor' in k:
                encdec_model_state[k] = v
        target_model.load_state_dict(encdec_model_state, strict=False)
        target_model.node_encoder.requires_grad_(False)
        target_model.edge_encoder.requires_grad_(False)
        target_model.decoder.requires_grad_(False)
        target_model.predecessor[0].requires_grad_(False)
        target_model.termination.requires_grad_(False)
        target_model.termination_mpnn.requires_grad_(False)
    else:
        raise NotImplementedError
    return target_model
    

