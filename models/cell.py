"""
linear attention

Cell
inter: (L_inter, C)
inner: (L_inner, C)

ecosystem
cells: List[Cell]
envir: (..., C)
"""


import torch
from torch import nn
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


class Cell:
    def __init__(self,
                 inner_state : torch.Tensor,
                 inter_state : torch.Tensor,
                 inner_name_dim_dict : Dict[str, int]):
        """
        inner_name_dim_dict: {name1: dim1, ...}
        """
        
        super().__init__()
        self.inner_state = inner_state # (B, L_inner, C)
        self.inter_state = inter_state # (B, L_inter, C)
        assert self.inner_state.shape[2] == self.inter_state.shape[2]
        assert sum(list(inner_name_dim_dict.values())) == self.inner_state.shape[1]


def inner_attn(cells: List[Cell],
            inner_decoder_layer: nn.Module):
    new_cells = []
    for cell in cells:
        temp_data = torch.cat([cell.inter_state, cell.inner_state], dim=1) # (B, L_inner+L_inter, C)
        new_cells.append(inner_decoder_layer(temp_data))
    return new_cells


from configs.cell_cls import CellCls
class Cells(nn.Module):
    """
    本质是一个embedding
    """
    def __init__(self,
                 structure: List[Tuple[str, int]],
                 embed_dim: int,
                 head_dim: int):
        super().__init__()
        
        init_state = []
        for name, N in structure:
            assert name in CellCls.cell_cls.keys(), f"{name} not in cell_cls"
            L = CellCls.cls_L_dict[name]
            init_state.append(
                nn.Parameter(torch.rand(N, L, embed_dim))
            )
        
        self.init_state = nn.ParameterList(init_state)
        self.structure = structure
        self.embed_dim = embed_dim
        
        self.L_inter_all = sum([N * CellCls.cls_L_inter_dict[name] for name, N in self.structure])

        # 更新它吧
        # self.state = init_state # List[(N, L, C)]
        # 不行，我们需要 B

        # cls_head
        head_dict = nn.ModuleDict()
        for name, _ in self.structure:
            token_head_dict = nn.ModuleDict()
            for token_name, range, dim in CellCls.cls_token_output_dim_dict[name]:
                token_head_dict[token_name] = nn.Linear(head_dim, dim)
            head_dict[name] = token_head_dict
        self.head_dict = head_dict

    def build_state(self, B):
        state = []
        for i, (name, N) in enumerate(self.structure):
            init = self.init_state[i]  # (N, L, C)

            # 扩展 batch
            s = init.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            # (B, N, L, C)

            state.append(s)
        return state

    def get_flatten_inter(self, state):
        """
        (B, L_inter_all, C)
        """
        inter_lst = []
        for i, (name, N) in enumerate(self.structure):
            start, end = CellCls.get_range(name, "inter")
            inter = state[i][:, :, start:end, :]
            inter_lst.append(inter.flatten(1,2)) # (B, N * L, C)
        result = torch.cat(inter_lst, dim=1)
        assert result.shape[1] == self.L_inter_all
        assert result.dim() == 3
        return result

    def update_inter(self,
                     new_inter,
                     state):
        """
        new_inter: (B, L_inter_all, C)
        state: List[(B, N, L, C)]
        update state
        """
        B = new_inter.shape[0]
        assert new_inter.shape[1] == self.L_inter_all
        assert new_inter.dim() == 3
        start = 0
        for i, (cls_name, N) in enumerate(self.structure):
            L_inter = CellCls.cls_L_inter_dict[cls_name]
            end = start + N * L_inter
            inter_flatted = new_inter[:, start:end, :]
            inter = inter_flatted.reshape(B, N, L_inter, -1)
            start = end
            
            storage_start, storage_end = CellCls.get_range(cls_name, "inter")
            state[i][:, :, storage_start:storage_end, :] = inter
    def extract_output(self, state):
        """
        从state: List[(B, N, L, C)]
        通过head提取输出
        return: Dict[cls_name, Dict[token_name, (B, N, 1, dim)]]
        List B [ Dict[cls, Dict[token, (N, dim)]] ]
        """
        output_dict = {}
        
        for i, (cls_name, N) in enumerate(self.structure):
            output_dict[cls_name] = {}
            
            token_range_dim_dict = CellCls.cls_token_output_dim_dict[cls_name]
            state_i = state[i]
            for token_name, range, dim in token_range_dim_dict:
                token_head = self.head_dict[cls_name][token_name]
                token_output = token_head(state_i[:, :, range[0]:range[1], :]) # (B, N, L, dim)
                assert token_output.shape[2] == 1
                output_dict[cls_name][token_name] = token_output[:,:,0,:] # (B, N, dim)

        B = token_output.shape[0]
                
        output_list = []
        for b in range(B):
            temp_dict = {}
            for cls_name, v1 in output_dict.items():
                for token_name, v2 in v1.items():
                    temp_dict[cls_name][token_name] = v2[b, :, :]
            output_list.append(temp_dict)
                
        return output_list

    def inner_decode(self,
                      decoder,
                      state):
        new_cell_states = []
        for this_state in state:
            B, N, L, C = this_state.shape
            state_flat = this_state.flatten(0,1) # (B * N, L, C)
            state_flat = decoder(state_flat) # (B * N, L, C)
            new_cell_states.append(state_flat.reshape(B, N, L, -1))
        return new_cell_states
        
    def infer(self,
              output_dict_dict,
              threshold):
        """
        output_dict_dict: Dict cls_name Dict token_name (N, dim)
        """
        result = {}
        for cls_name, output_dict in output_dict_dict.items():
            result[cls_name] = {}
            if cls_name in CellCls.not_need_match_cls:
                assert output_dict['exist'].numel()==1
                exist_prob = torch.sigmoid(output_dict['exist'][0,0])
                if exist_prob > threshold:
                    result[cls_name] = output_dict
                    result[cls_name]['exist'] = exist_prob
                    if result[cls_name].get("sustain"):
                        result[cls_name]['sustain'] = torch.exp(result[cls_name]['sustain']) * CellCls.sustain_ref
                    else:
                        pass
                else:
                    result[cls_name]['exist'] = exist_prob
            else:
                exist_prob = torch.sigmoid(output_dict['exist'][:, 0])
                choice = exist_prob > threshold
                for token_name, output in output_dict.items():
                    if token_name == "sustain":
                        result[cls_name][token_name] = torch.exp(output[choice, :]) * CellCls.sustain_ref
                    elif token_name == "anchor":                 
                        temp_out = output[choice, :]
                        temp_out[:, 1] = torch.exp(temp_out[:, 1]) * CellCls.sustain_ref
                        result[cls_name][token_name] = temp_out
                    else:
                        result[cls_name][token_name] = output[choice, :]
                result[cls_name]["exist"] = exist_prob[choice]
        return result
        
    def get_sample_loss(self,
                        output_dict_dict,
                        target_dict_dict,
                        cost_weight,
                        loss_weight):
        loss = 0
        for cls_name in target_dict_dict.keys():
            if cls_name in CellCls.not_need_match_cls:
                loss += self.get_sample_noMatch_clsName_loss(cls_name,
                                                             output_dict_dict,
                                                             target_dict_dict,
                                                             loss_weight)
            else:
                loss += self.get_sample_clsName_loss(cls_name,
                                                     output_dict_dict,
                                                     target_dict_dict,
                                                     cost_weight,
                                                     loss_weight)
        return loss


    def get_sample_noMatch_clsName_loss(self,
                                cls_name,
                                output_dict_dict,
                                target_dict_dict,
                                loss_weight):
        """
        target_dict_dict: Dict Dict [ token_name, (1, fea_dim) ]
        """
        loss = 0
        exist = target_dict_dict[cls_name]['exist'][0,0] == 1
        if not exist:
            loss_func = CellCls.posterior_cls_token_loss_dict[cls_name]["exist"]
            out = output_dict_dict[cls_name]['exist'][0,0]
            tar = target_dict_dict[cls_name]['exist'][0,0]
            loss = loss_func(out, tar)
            return loss
        for token_name in target_dict_dict[cls_name].keys():
            loss_func = CellCls.posterior_cls_token_loss_dict[cls_name][token_name]
            out = output_dict_dict[cls_name][token_name]
            assert out.shape[0]==1
            assert out.dim()==2
            out = out[0, :]
            tar = target_dict_dict[cls_name][token_name]
            assert tar.shape[0]==1
            assert tar.dim()==2
            tar = tar[0, :]
            temp_loss = loss_func(out, tar)
            loss += loss_weight[cls_name][token_name] * temp_loss
        return loss

    def get_sample_clsName_loss(self,
                                cls_name,
                                output_dict_dict,
                                target_dict_dict,
                                cost_weight,
                                loss_weight):
        gt_idx, pred_idx, cost_dict = self.match_event(output_dict_dict, target_dict_dict, cls_name, cost_weight)
        loss = 0
        for token_name in cost_dict.keys():
            if token_name not in CellCls.posterior_cls_token_loss_dict[cls_name][token_name].keys():
                loss_func = CellCls.posterior_cls_token_loss_dict[cls_name][token_name]
                if token_name == "exist":
                    temp_loss = loss_func(output_dict_dict[cls_name][token_name], gt_idx)
                    loss += loss_weight[cls_name][token_name] * temp_loss
                else:
                    raise NotImplementedError("wtf")
            else:
                cost = cost_dict[token_name]
                temp_loss = cost[gt_idx, pred_idx].mean()
                loss += loss_weight[cls_name][token_name] * temp_loss
        return loss

    def match_event(self,
                    output_dict_dict,
                    target_dict_dict,
                    cls_name,
                    cost_weight):
        """
        output: Dict[ cls_name, Dict[ fea_name, (Q, C_fea) ] ]
        target: Dict[ cls_name, Dict[ fea_name, (N, C_fea) ] ]
        cost_weight: Dict[ cls_name, Dict[ fea_name, float ] ]
        对每个 cls_name 进行一次匹配
        """
        output = output_dict_dict[cls_name]
        target = target_dict_dict[cls_name]
        
        cost_dict = {}
        
        cost = 0
        for fea_name in output.keys():
            assert fea_name in target.keys(), f"{fea_name} not in target"
            assert output[fea_name].shape[1] == target[fea_name].shape[1], f"{fea_name} dim not match"
            cost_func = CellCls.cls_token_cost_dict[cls_name][fea_name]
            if fea_name == "exist":
                temp_cost = cost_func(output[fea_name])
            else:
                temp_cost = cost_func(output[fea_name], target[fea_name]) # (N, Q)
            cost += cost_weight[cls_name][fea_name] * temp_cost
            cost_dict[fea_name] = temp_cost

        cost_np = cost.detach().cpu().numpy()
        assert cost_np.shape[1] >= cost_np.shape[0], f"too less of query, got {cost_np.shape[1]}, but have {cost_np.shape[0]} event"
        gt_idx, pred_idx = linear_sum_assignment(cost_np)
        return gt_idx, pred_idx, cost_dict



class CognitiveMediation(nn.Module):
    """
    1、cell 的 inter 部分 和 modal 作注意力
    2、更新后的 inter 和 inner 作注意力
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.num_layers = cfg.num_layers
        self.d_model_list = cfg.d_model_list
        self.dim_up = cfg.dim_up
    def forward(self,
                cells:Cells,
                modal_dict:Dict[str, torch.Tensor],
                law:nn.Module):
        """
        modal_dict: 外部模态信息
        cells: 细胞感知的信息
        
        一边加工 modal_dict, 一边总结认知(更改cells)
        """
        for i in range(self.num_layers):
            cells, modal_dict = self.inter_attn(cells, modal_dict, law, i)
            cells = self.inner_attn(cells, law, i)
        return cells, modal_dict
    def inter_attn(self,
                   cells:Cells,
                   modal_dict:Dict[str, torch.Tensor],
                   law:nn.Module,
                   i:int):
        pass        