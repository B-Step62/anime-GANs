import torch

from .statistics import calc_mean_std

def adaptive_instance_normalization(content, style):
    assert(content.size()[:2] == style.size()[:2])
    size = content.size()
    style_mean, style_std = calc_mean_std(style)
    content_mean, content_std = calc_mean_std(content)
    
    normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
    output = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return output
