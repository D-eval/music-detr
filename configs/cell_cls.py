"""
List[Tuple[str, int, Optional[List[Tuple[str, int]]]]]
name, num_tokens, head_dim_list[name, num_fea]
"""


def get_cls_L_dict(cell_cls, name=None):
    cls_L_dict = {}
    for key, value in cell_cls.items():
        L = 0
        for v in value:
            if name is None or v[0] == name:
                L += v[1]
            else:
                continue
        cls_L_dict[key] = L
    return cls_L_dict


def get_output_dim_dict(cell_cls):
    output_dim_dict = {}
    for key, value in cell_cls.items():
        output_dim_dict[key] = {}
        start = 0
        for token_name, num_tokens, head_dim_list in value:
            if head_dim_list is None:
                start += num_tokens
                continue
            output_dim = 0
            for dim_name, dim in head_dim_list:
                output_dim += dim
            output_dim_dict[key][token_name] = ((start, start+num_tokens), output_dim)
            start += num_tokens
    return output_dim_dict

from .costoss import *



class CellCls:
    cell_cls = {
        "chord": [("inter", 8, None),
                ("anchor", 1, [("start", 1), ("sustain", 1)]),
                ("exist", 1, [("exist", 1)]),
                ("root", 1, [("root", 13)]),
                ("chord", 1, [("chord", 12)]),
                ("tonic", 1, [("tonic", 13)])], # disentangle 模式，用 inter 感知外部，用 attn 融合内部
        "chord_before": [("inter", 8, None),
                        ("sustain", 1, [("sustain", 1)]),
                        ("exist", 1, [("exist", 1)]),
                        ("root", 1, [("root", 13)]),
                        ("chord", 1, [("chord", 12)]),
                        ("tonic", 1, [("tonic", 13)])], # 在 segment_start 前就响了
        # "_chord": [("inter", 8, None),
        #         ("chord", 1, [
        #             ("start", 1),
        #             ("sustain", 1),
        #             ("exist", 1),
        #             ("root", 13),
        #             ("chord", 12),
        #             ("tonic", 13)
        #         ])], # entangle 模式，用 inter 感知外部，用 ffn 融合内部，用于消融实验
        "beat": [("inter", 2, None),
                ("beat", 1, [("beat", 1)]),
                ("is_downbeat", 1, [("is_downbeat", 1)]),
                ("exist", 1, [("exist", 1)]),
                ],
        "metronome": [("inter", 8, None),
                    ("bpm", 1, [("bpm", 1)]),
                    ("offset", 1, [("offset", 1)]),
                    ("is_4beat", 1, [("is_4beat", 1)]),
                    ("exist", 1, [("exist", 1)]), # 只有bpm全片段一致，才为 1
                    ],
    }
    cls_L_dict = get_cls_L_dict(cell_cls)
    cls_L_inter_dict = get_cls_L_dict(cell_cls, "inter")
    cls_token_output_dim_dict = get_output_dim_dict(cell_cls)
    
    # hungarian matching & loss
    cls_token_cost_dict = cls_token_cost_dict
    not_need_match_cls = not_need_match_cls
    not_need_match_cls_token_loss_dict = not_need_match_cls_token_loss_dict
    posterior_cls_token_loss_dict = posterior_cls_token_loss_dict
    
    # infer_params
    sustain_ref = sustain_ref
    @staticmethod
    def get_range(cls, cls_name, token_name):
        assert cls_name in CellCls.cell_cls.keys(), f"{cls_name} not in cell_cls"
        
        start = 0
        for name, num_tokens, head_dim_list in cls.cell_cls[cls_name]:
            if name == token_name:
                end = start + num_tokens
                return start, end
            start += num_tokens
        assert False, f"{token_name} not in {cls_name}"


def validate_cell_cls(cell_cls: dict):
    assert isinstance(cell_cls, dict), "cell_cls 必须是 dict"

    for key, value in cell_cls.items():
        assert isinstance(key, str), f"{key} 不是 str"

        assert isinstance(value, list), f"{key} 的 value 必须是 list"
        
        assert "inter" in [item[0] for item in value], f"{key} 中必须包含 inter"
        
        for i, item in enumerate(value):
            assert isinstance(item, tuple), f"{key}[{i}] 不是 tuple"
            assert len(item) == 3, f"{key}[{i}] 长度必须为 3"

            name, num_tokens, head_dim_list = item

            # 1️⃣ name
            assert isinstance(name, str), f"{key}[{i}].name 必须是 str"

            # 2️⃣ num_tokens
            assert isinstance(num_tokens, int) and num_tokens > 0, \
                f"{key}[{i}].num_tokens 必须是正整数"

            # 3️⃣ head_dim_list
            if head_dim_list is not None:
                assert isinstance(head_dim_list, list), \
                    f"{key}[{i}].head_dim_list 必须是 list 或 None"

                for j, sub in enumerate(head_dim_list):
                    assert isinstance(sub, tuple) and len(sub) == 2, \
                        f"{key}[{i}][{j}] 必须是 (name, dim)"

                    sub_name, sub_dim = sub

                    assert isinstance(sub_name, str), \
                        f"{key}[{i}][{j}].name 必须是 str"

                    assert isinstance(sub_dim, int) and sub_dim > 0, \
                        f"{key}[{i}][{j}].dim 必须是正整数"
                        
validate_cell_cls(CellCls.cell_cls)