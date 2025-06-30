import warnings
import torch
from torch.utils.data import Dataset, Subset


def kfold(dataset: Dataset, k: int, shuffle: bool = True, random_state: int = 42):
    """
    对给定的 PyTorch 数据集进行 K 折交叉验证划分。

    参数:
    dataset (Dataset): 输入的 PyTorch 数据集对象。
    k (int): 折数。
    shuffle (bool): 是否在划分前打乱数据，默认为 True。
    random_state (int): 随机种子，用于保证结果的可重复性，默认为 42。

    返回:
    list: 包含 k 个元组的列表，每个元组为 (训练数据集, 验证数据集)。
    """
    try:
        from sklearn.model_selection import KFold
    except ImportError:
        warnings.warn("请安装 sklearn 库以使用 KFold 功能。可以使用 'pip install scikit-learn' 命令安装。")
    else:
        kf = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
        folds = []
        for train_index, val_index in kf.split(range(len(dataset))):
            train_dataset = Subset(dataset, train_index)
            val_dataset = Subset(dataset, val_index)
            folds.append((train_dataset, val_dataset))
        return folds


def get_gpu_usage():
    try:
        from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
    except ImportError:
         warnings.warn("pynvml库未安装，获取GPU显存功能不可用。使用pip install nvidia-ml-py3安装。")
    else:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        return f"GPU显存使用情况: {info.used//1024**3} GB"


def get_model_info(model):
    """
    返回模型的层级信息，兼容DDP/DeepSpeed包装，包含参数统计和显存估算。

    Args:
        model (torch.nn.Module): PyTorch模型（支持DDP/DeepSpeed包装）

    Returns:
        str: 格式化模型信息
    """
    # 解包DDP/DeepSpeed模型
    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    original_model = model
    while hasattr(original_model, 'module'):
        original_model = original_model.module

    model_info = []
    total_trainable = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    total_non_trainable = sum(p.numel() for p in original_model.parameters() if not p.requires_grad)
    total_size_bytes = sum(p.numel() * p.element_size() for p in original_model.parameters())

    # 构建树状结构
    tree = {}
    modules = list(original_model.named_modules())
    root_name = modules[0][0]  # 获取根模块名称
    root_cls = original_model.__class__.__name__

    # 递归构建树结构
    def build_tree(node, path):
        if not path:
            return
        head, *tail = path
        if head not in node:
            node[head] = {"children": {}, "module": None}
        if tail:
            build_tree(node[head]["children"], tail)
        else:
            node[head]["module"] = module

    # 收集所有模块
    for name, module in modules:
        if name == "":  # 根模块特殊处理
            tree[root_name] = {"children": {}, "module": module}
            tree[root_name]["display_name"] = root_cls
            continue
        path = name.split('.')
        build_tree(tree[root_name]["children"], path)

    # 生成树状前缀
    def traverse(node, prefix=[], is_last=False):
        nonlocal total_trainable, total_non_trainable, total_size_bytes

        # 生成当前节点信息
        params = list(node["module"].parameters()) if node["module"] else []
        num_params = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        non_trainable = num_params - trainable
        module_size = sum(p.numel() * p.element_size() for p in params)

        # 生成缩进前缀
        display_prefix = ""
        for p in prefix:
            display_prefix += "    " if p else "│   "
        if prefix:
            display_prefix += "└── " if is_last else "├── "

        model_info.append({
            "prefix": display_prefix,
            "name": node["display_name"] if "display_name" in node else list(node["children"].keys())[0],
            "type": type(node["module"]).__name__ if node["module"] else "",
            "params": num_params,
            "trainable": trainable,
            "non_trainable": non_trainable,
            "size": module_size
        })

        # 递归处理子节点
        children = list(node["children"].items())
        for i, (child_name, child_node) in enumerate(children):
            is_last_child = i == len(children) - 1
            new_prefix = prefix + [is_last]
            child_node["display_name"] = child_name
            traverse(child_node, new_prefix, is_last_child)

    # 遍历根节点
    traverse(tree[root_name], prefix=[], is_last=False)

    # 内存单位转换
    total_size_mb = total_size_bytes / (1024 ** 2)
    total_size_gb = total_size_bytes / (1024 ** 3)

    # 构建表格
    header = "| {:<50} | {:<20} | {:<15} | {:<12} | {:<14} |".format(
        "Name", "Type", "Params", "Trainable", "Non-Trainable"
    )
    separator = "-" * len(header)
    rows = [separator, header, separator]

    for info in model_info:
        full_name = f"{info['prefix']}{info['name']}"
        full_name = full_name[:50].ljust(50)

        row = "| {:<50} | {:<20} | {:<15} | {:<12} | {:<14} |".format(
            full_name,
            info["type"][:20],
            f"{info['params']:,}",
            f"{info['trainable']:,}",
            f"{info['non_trainable']:,}"
        )
        rows.append(row)
    rows.append(separator)

    # 汇总信息
    summary = [
        "Model SUMMARY:",
        f"Total Trainable: {total_trainable:,}",
        f"Total Non-Trainable: {total_non_trainable:,}",
        f"Total Parameters: {total_trainable + total_non_trainable:,}",
        f"Estimated Size: {total_size_mb:.2f} MB ({total_size_gb:.2f} GB)"
    ]

    return "\n".join(rows + summary) + '\n' + separator


def freeze_network(model):
    for param in model.parameters():
        param.requires_grad = False
    return


def get_model_params_num(model):
    num = sum(p.numel() for p in model.parameters())

    return round(num / (1024**2), 2)


def get_batch_n(train_data_loader):
    import math

    dataset_n = len(train_data_loader.dataset)
    batch_size = train_data_loader.batch_size
    # 多GPU训练时，单GPU的batch_n，所以要先知道用了几张GPU
    batch_n = math.ceil(dataset_n / batch_size)
    import torch.distributed as dist

    if dist.is_initialized():
        batch_n = batch_n / dist.get_world_size()

    return batch_n


def mean_pooling(token_embeddings, attention_mask):
    """
    global-meaning-pooling
    this method just mean the sequence in the sequence dim based on mask

    @param token_embeddings:
    @param attention_mask: (l, B)
    @return:
    """
    attention_mask = ~attention_mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float() # [B, l, d]
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim = 1) # [B, d]
    sum_mask = torch.clamp(input_mask_expanded.sum(dim = 1), min=1e-9) # [B, d]
    return sum_embeddings / sum_mask
