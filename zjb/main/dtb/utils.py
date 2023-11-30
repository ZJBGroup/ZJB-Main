import re

# 希腊字母字符串列表
STR_GREEK_LETTERS = [
    'alpha', 'beta', 'gamma', 'delta', 'epsilon',
    # 交换theta和eta以避免theta的eta先被替换
    # lambda是python保留字，所以使用lamda代替
    'zeta', 'theta', 'eta', 'lota', 'kappa', 'lamda',
    'mu', 'nu', 'xi', 'omicron', 'pi', 'rho', 'sigma',
    'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega'
]
# 希腊字母对应的Unicode字符
UNICODE_GREEK_LETTERS = '𝛼𝛽𝛾𝛿𝜀𝜁𝜃𝜂𝜄𝜅𝜆𝜇𝜈𝜉𝜊𝜋𝜌𝜎𝜏𝜐𝜑𝜒𝜓𝜔'
# 大写希腊字母对应的Unicode字符
UNICODE_GREEK_UPPER_LETTERS = '𝛢𝛣𝛤𝛥𝛦𝛧𝛩𝛨𝛪𝛫𝛬𝛭𝛮𝛯𝛰𝛱𝛲𝛴𝛵𝛶𝛷𝛸𝛹𝛺'

# 字母字符串列表
STR_LETTERS = 'abcdefghijklmnopqrstuvwxyz'
# 字母对应的Unicode字符
UNICODE_LETTERS = '𝑎𝑏𝑐𝑑𝑒𝑓𝑔ℎ𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑤𝑥𝑦𝑧'
# 大写字母对应的Unicode字符
UNICODE_UPPER_LETTERS = '𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐽𝐾𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑊𝑋𝑌𝑍'

# 内置函数列表
BUILTIN_FUNCTIONS = [
    'exp', 'ln', 'log2', 'log10',
    'sin', 'sinc', 'sinh', 'cos', 'cosh',
    'tan', 'tanh', 'arcsin', 'arcsinh',
    'arccos', 'arccosh', 'arctan',
    'arctan2', 'arctanh'
]

import numpy as np

np.tan

# 字符串转Unicode字符的映射字典
STR2UNICODE_LETTERS = {
    s.encode()[0]: u
    for s, u in zip(STR_LETTERS, UNICODE_LETTERS)
} | {
    s.capitalize().encode()[0]: u
    for s, u in zip(STR_LETTERS, UNICODE_UPPER_LETTERS)
}

# 上标字符
STR_SUP_CHARS = '0123456789+-=()ni'
UNICODE_SUP_CHARS = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ'

# 下标字符
STR_SUB_CHARS = '0123456789+-=()aehijklmnoprstuvx'
UNICODE_SUB_CHARS = '₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ'

# 特殊符号映射
STR2UNICODE = {
    '\\sum': '∑'
}

# Unicode字符转字符串的映射
UNICODE2STR = {
    u: s
    for s, u in STR2UNICODE.items()
}

# 正则表达式
RE_AxA = re.compile(r'(\w+)\s*\*\s*\1')
RE_SPACE = re.compile(r'\s+')


def expression2unicode(expression: str, rich=True):
    """
    将数学表达式字符串转换为包含Unicode字符和HTML标签的格式化字符串。

    Parameters
    ----------
    expression : str
        原始的数学表达式字符串。
    rich : bool, optional
        是否使用富文本格式，默认为True。

    Returns
    -------
    str
        转换后的格式化字符串。
    """
    exp = expression
    exp = exp.replace('np.', '')
    exp = exp.replace('**', '^')

    # 希腊字母
    for s, u, uu in zip(STR_GREEK_LETTERS, UNICODE_GREEK_LETTERS, UNICODE_GREEK_UPPER_LETTERS):
        exp = exp.replace(s, u)
        exp = exp.replace(s.capitalize(), uu)

    # 数学符号
    for s, u in STR2UNICODE.items():
        exp = exp.replace(s, u)

    # a*a => a^2
    exp, _ = RE_AxA.subn(r'\1^2', exp)

    # 字母（数学字体）
    exp = exp.translate(STR2UNICODE_LETTERS)

    if rich:
        # 处理下标
        for match in re.finditer(r'_(\w+)', exp):
            _chars = match.group()
            chars = match.group(1)
            start_index = match.start()
            exp = exp[:start_index] + exp[start_index:].replace(_chars, f'<sub>{chars}</sub>', 1)
        # 处理上标
        for match in re.finditer(r'\^\s*(\w+)', exp):
            exp = exp.replace(match.group(0), f'<sup>{match.group(1)}</sup>', 1)

    # 内建函数不使用数学字体
    for func in BUILTIN_FUNCTIONS:
        exp = exp.replace(
            func.translate(STR2UNICODE_LETTERS) + '(',
            func + '('
        )

    exp = exp.replace('*', ' ')

    # 清除重复空格
    exp, _ = RE_SPACE.subn(' ', exp)

    return exp
